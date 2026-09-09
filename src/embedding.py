"""Item2vec 商品向量训练基础层（CBOW / Skip-gram 共用）。

把数据集里每笔订单（InvoiceNo）看成「一个购物篮 = 一个句子」：
句子的词 = 该订单内去重后的商品 StockCode。篮内先后顺序本来没有意义，
所以用固定随机种子把每个篮内部的商品顺序打乱一次，消除原始行序带来的
位置偏差；再交给 gensim Word2Vec 训练，得到每个商品的稠密向量。

CBOW (sg=0)：用篮内其它商品(上下文)平均出向量，去预测中间那件商品。
Skip-gram (sg=1)：用当前商品去预测它周边的其它商品。
两者都只用「出现在同一订单」的商品对作为正样本（共同购买信号），
随机抽出的负样本让向量不至于全部挤在一起。

依赖：gensim 4.x（pip install gensim）。
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from gensim.models import Word2Vec, callbacks

from config.settings import COLUMNS

_INVOICE = COLUMNS["invoice"]
_ITEM = COLUMNS["item"]


# 训练超参数（与 config/settings.py 中 HYPERPARAMS["item2vec"] 保持一致）
HYPER = {
    "vector_size": 128,   # 每个商品向量的维数
    "window": 8,          # 上下文窗口半径（对打乱后的篮，近似随机挑“同篮商品”作上下文）
    "min_count": 5,       # 至少出现在 5 个订单里的商品才保留进词表
    "negative": 5,        # 负采样个数（见 README/Word 素材中“负样本设计”一节）
    "sample": 1e-3,       # 高频词下采样阈值（抑制 POST 之类服务性高频伪商品）
    "epochs": 12,         # 全量语料遍历 12 遍
    "workers": 4,         # 训练并行线程
    "seed": 42,           # 复现用随机种子
    "alpha": 0.025,       # 初始学习率
    "min_alpha": 0.0001,  # 学习率线性退火到的最小值
}


def build_basket_sentences(
    df: pd.DataFrame, seed: int = 42, min_len: int = 2
) -> list[list[str]]:
    """每笔订单 → 一个句子（去重商品码）。篮内顺序用 seed 打乱一次。

    Args:
        df: 清洗后交易（含 InvoiceNo / StockCode）。
        seed: 打乱每个篮内部顺序的随机种子。
        min_len: 至少含 min_len 种商品才保留（1 种商品无法构成“共同购买”）。

    Returns:
        [[StockCode, ...], ...]，列表长度 = 订单数。
    """
    grouped = df.groupby(_INVOICE, sort=False)[_ITEM].agg(
        lambda s: list(dict.fromkeys(str(x) for x in s))
    )
    rng = np.random.default_rng(seed)
    sentences: list[list[str]] = []
    for items in grouped:
        if len(items) >= min_len:
            rng.shuffle(items)  # 篮内顺序无意义：固定随机化，避免原始行序偏差
            sentences.append(items)
    return sentences


class _EpochLossRecorder(callbacks.CallbackAny2Vec):
    """每个 epoch 结束时记录一次「累计训练损失」（跨 epoch 求和）。"""

    def __init__(self) -> None:
        self.cumulative: list[float] = []

    def on_epoch_end(self, model) -> None:
        self.cumulative.append(float(model.get_latest_training_loss()))


def train_basket_embeddings(
    sentences: list[list[str]],
    *,
    sg: int,
    hyper: dict | None = None,
) -> tuple[Word2Vec, list[float]]:
    """训练一个 CBOW(sg=0) 或 Skip-gram(sg=1) 模型，返回 (model, 逐epoch损失)。

    gensim 在单次 train(epochs=N) 内部按 N 个 epoch 做线性学习率退火，
    用回调在每个 epoch 结束时记录累计损失，差分后得到逐 epoch 损失。
    """
    h = dict(HYPER)
    if hyper:
        h.update(hyper)
    h["sg"] = int(sg)

    model = Word2Vec(
        vector_size=int(h["vector_size"]),
        window=int(h["window"]),
        min_count=int(h["min_count"]),
        sg=h["sg"],
        negative=int(h["negative"]),
        sample=float(h["sample"]),
        epochs=int(h["epochs"]),
        workers=int(h["workers"]),
        seed=int(h["seed"]),
        alpha=float(h["alpha"]),
        min_alpha=float(h["min_alpha"]),
        hs=0,          # 关闭层次 softmax，只用负采样
        compute_loss=True,
    )
    model.build_vocab(sentences)

    rec = _EpochLossRecorder()
    model.train(
        sentences,
        total_examples=len(sentences),
        epochs=int(h["epochs"]),
        compute_loss=True,
        callbacks=[rec],
    )

    cum = rec.cumulative
    per_epoch = [cum[0]] + [cum[i] - cum[i - 1] for i in range(1, len(cum))]
    return model, per_epoch


def save_vectors(model: Word2Vec, path: str) -> None:
    """把词表与其稠密向量存成 npz（供离线相似度计算 / 后续可视化）。"""
    vecs = model.wv.vectors            # 行序 = model.wv.index_to_key
    keys = model.wv.index_to_key
    np.savez_compressed(str(path), vectors=vecs, keys=np.asarray(keys, dtype=str))


def load_vectors(path: str) -> tuple[list[str], np.ndarray]:
    """读回 save_vectors 保存的 (keys, vectors)，保持行序一致。"""
    z = np.load(str(path), allow_pickle=False)
    keys = [str(k) for k in z["keys"]]
    return keys, z["vectors"]


def top_k_similar_from_vectors(
    keys: list[str],
    vecs: np.ndarray,
    top_k: int = 10,
    *,
    desc_of: Callable[[str], str] | None = None,
    min_score: float = 0.0,
) -> dict[str, list[tuple[str, float]]]:
    """由稠密向量批量计算 Top-K 相似商品（余弦），排除自身。

    Returns:
        {code: [(neighbor_code, score), ...]}，score 保留 4 位小数。
    """
    n = vecs.shape[0]
    norms = np.linalg.norm(vecs, axis=1)
    norms[norms == 0] = 1.0
    unit = vecs / norms[:, None]
    sim = unit @ unit.T          # n × n 稠密余弦
    np.fill_diagonal(sim, -np.inf)

    out: dict[str, list[tuple[str, float]]] = {}
    for i in range(n):
        code = keys[i]
        order = np.argsort(-sim[i])
        hits: list[tuple[str, float]] = []
        for pos in order:
            score = float(sim[i, pos])
            if score <= min_score:
                break
            hits.append((str(keys[pos]), round(score, 4)))
            if len(hits) >= top_k:
                break
        out[str(code)] = hits
    return out
