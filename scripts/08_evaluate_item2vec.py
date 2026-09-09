"""阶段五公平对比：ItemCF（协同过滤） vs CBOW 嵌入 vs Skip-gram 嵌入。

与 scripts/05_evaluate_user_vs_item.py 完全相同的离线协议：
    每个用户「最后一笔订单(按时间)」= 测试篮，其余历史为训练；
    从训练历史为用户补 Top-N，衡量 recall@N / precision@N / hit@10。

区别：嵌入模型只用「训练期的订单」训练（不含测试篮，避免数据泄漏），
三种方法在同一批测试用户、同一批测试篮上评测，可直接对比。
全局热门作为第三条参照基线。

候选空间说明：ItemCF 可在全部训练期商品中推荐；嵌入方法只能在词表商品
（出现 ≥ min_count 次的商品）中推荐，测试篮里的冷门品它天然够不到——
这是方法本身的属性，指标会如实反映，不做回填。

运行：venv\\Scripts\\python.exe scripts\\08_evaluate_item2vec.py
（复用 scripts/06 的超参与 src/embedding.train_basket_embeddings）
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import COLUMNS, SEED
from src.embedding import HYPER, build_basket_sentences, train_basket_embeddings
from src.user_similarity import load_clean_transactions

_USER, _ITEM = COLUMNS["user"], COLUMNS["item"]
_TIME, _QTY = COLUMNS["time"], COLUMNS["qty"]
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def to_csr(sub: pd.DataFrame) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    s = sub.drop_duplicates([_USER, _ITEM])
    u_codes, u_ids = pd.factorize(s[_USER])
    i_codes, i_ids = pd.factorize(s[_ITEM])
    m = coo_matrix(
        (np.ones(len(s), dtype=np.float64), (u_codes, i_codes)),
        shape=(len(u_ids), len(i_ids)),
    ).tocsr()
    return m, u_ids.astype("int64"), i_ids


def main() -> None:
    t0 = time.time()
    df = load_clean_transactions()
    df = df.copy()
    df[_USER] = df[_USER].astype("int64")
    df[_ITEM] = df[_ITEM].astype(str)
    df[_TIME] = pd.to_datetime(df[_TIME])

    # ---------- 与 05 相同的训练/测试切分 ----------
    last_t = df.groupby(_USER)[_TIME].max().rename("last_t")
    df = df.merge(last_t, left_on=_USER, right_index=True)
    test = df[df[_TIME] >= df["last_t"]]
    train = df[df[_TIME] < df["last_t"]]

    B, train_users, train_items = to_csr(train)
    n_items = B.shape[1]
    code_of = {c: str(i) for c, i in enumerate(train_items)}
    col_of = {str(i): c for c, i in enumerate(train_items)}

    users = sorted(set(int(x) for x in train_users) & set(int(x) for x in test[_USER]))
    print(f"训练用户数 {B.shape[0]}  商品数 {n_items}  可分训练/测试的用户 {len(users)}")
    user_row = {int(u): r for r, u in enumerate(train_users)}

    own_codes: dict[int, set] = {u: set() for u in users}
    for u in users:
        own_codes[u] = {code_of[c] for c in B[user_row[u]].indices.tolist()}

    test_map: dict[int, set] = {}
    known_map: dict[int, list[int]] = {}
    for uid, g in test.groupby(_USER):
        uid = int(uid)
        basket = {str(s) for s in set(g[_ITEM])}
        test_map[uid] = basket
        known_map[uid] = [col_of[x] for x in basket if x in col_of]

    # 全局热门（训练期被多少用户买过，作基线）
    pop_cols = np.argsort(-np.asarray(B.sum(axis=0)).ravel()).tolist()
    pop_codes = [code_of[c] for c in pop_cols]

    # ---------- 只用训练期订单训练嵌入（无泄漏）----------
    sent = build_basket_sentences(train, seed=SEED, min_len=2)
    print(f"训练期购物篮语料：{len(sent)} 个订单")
    emb: dict[str, dict] = {}
    for key, sg in (("cbow", 0), ("skip", 1)):
        tk = time.time()
        model, _losses = train_basket_embeddings(sent, sg=sg, hyper={"seed": SEED})
        keys = model.wv.index_to_key
        vecs = model.wv.vectors
        norms = np.linalg.norm(vecs, axis=1)
        norms[norms == 0] = 1.0
        unit = vecs / norms[:, None]
        emb[key] = {"unit": unit, "keys": keys, "row": {k: i for i, k in enumerate(keys)}}
        print(f"  {key}: vocab={len(keys)}  耗时 {time.time()-tk:.0f}s")

    # ---------- ItemCF：相似商品聚合（与 05 相同）----------
    ITEM = B.T.tocsr()
    sim_cache: dict[int, np.ndarray] = {}

    def item_sim_vec(col: int) -> np.ndarray:
        v = sim_cache.get(col)
        if v is None:
            row = ITEM[col]
            v = cosine_similarity(row, ITEM).ravel() if row.nnz else np.zeros(n_items)
            v[col] = 0.0
            sim_cache[col] = v
        return v

    def rec_item(u: int) -> list[str]:
        cand: dict[str, float] = {}
        for col in known_map[u]:
            v = item_sim_vec(col)
            for j in np.argsort(-v)[:200]:
                sc = float(v[j])
                if sc <= 0:
                    break
                code = code_of[j]
                if code not in own_codes[u]:
                    cand[code] = cand.get(code, 0.0) + sc
        return [c for c, _ in sorted(cand.items(), key=lambda kv: -kv[1])]

    def rec_pop(u: int) -> list[str]:
        return [c for c in pop_codes if c not in own_codes[u]]

    # ---------- 嵌入：候选 = 词表商品，分数 = 训练期已购商品向量余弦之和 ----------
    def make_rec_embed(data: dict) -> object:
        unit, keys, row = data["unit"], data["keys"], data["row"]
        owned_rows: dict[int, list[int]] = {}

        def rec_embed(u: int) -> list[str]:
            if u not in owned_rows:
                owned_rows[u] = [row[c] for c in own_codes[u] if c in row]
            orows = owned_rows[u]
            if not orows:
                return []
            profile = unit[orows].mean(axis=0)                 # 用户“购买语义”中心
            scores = unit @ profile                            # 与每个词表商品的相似度累加
            for i in orows:
                scores[i] = -np.inf                            # 排除已购
            pos = np.flatnonzero(scores > 0)
            order = pos[np.argsort(-scores[pos])][:20]
            return [keys[i] for i in order]

        return rec_embed

    # ---------- 评测（与 05 相同指标）----------
    N_list = [5, 10]

    def metrics(rec_fn) -> dict[str, float]:
        r5, p5, r10, p10, h10 = [], [], [], [], []
        for u in users:
            basket = test_map[u]
            topn = rec_fn(u)
            for N, (rl, pl) in ((5, (r5, p5)), (10, (r10, p10))):
                hits = len(set(topn[:N]) & basket)
                rl.append(hits / len(basket))
                pl.append(hits / N)
            h10.append(1.0 if len(set(topn[:10]) & basket) else 0.0)
        return {"recall@5": float(np.mean(r5)), "precision@5": float(np.mean(p5)),
                "recall@10": float(np.mean(r10)), "precision@10": float(np.mean(p10)),
                "hit@10": float(np.mean(h10))}

    vocab = set(emb["cbow"]["keys"])

    def restrict_vocab(fn):
        """只允许推荐出现在嵌入词表里的商品（与嵌入方法同候选池，公平对比）。"""
        def r(u: int) -> list[str]:
            return [c for c in fn(u) if c in vocab]
        return r

    print("\n" + "=" * 92)
    print("同一留出协议下的离线对比（候选均排除训练期已购商品）")
    print("=" * 92)
    methods = [
        ("ItemCF (物品余弦聚合)", rec_item),
        ("ItemCF (限词表候选)", restrict_vocab(rec_item)),
        ("CBOW 嵌入 (已购向量累加)", make_rec_embed(emb["cbow"])),
        ("Skip-gram 嵌入 (已购向量累加)", make_rec_embed(emb["skip"])),
        ("全局热门 基线", rec_pop),
        ("全局热门 (限词表候选)", restrict_vocab(rec_pop)),
    ]
    for name, fn in methods:
        res = metrics(fn)
        line = f"  {name:<26}" + "  ".join(f"{k}={v:.4f}" for k, v in res.items())
        print(line)
    print("=" * 92)
    print(f"总耗时 {time.time()-t0:.0f}s。嵌入模型仅用训练期订单训练（无测试篮泄漏）；")
    print("“限词表候选”行 = 把该方法也限制在嵌入词表内（与嵌入同池），消除候选空间差异。")


if __name__ == "__main__":
    main()
