"""阶段六链路 · 序列落地 + 规范产物（指导书阶段二尾/三前）。

本脚本产出三样东西：
1. `data/processed/user_sequences.csv` —— 用户视角的按时间排序购物序列（长表）。
   每行 = 该用户在某张订单里买的一种商品；`pos_in_invoice` 是商品在订单内的行序，
   便于下游把「同订单」商品聚成一个 basket（Item2Vec 的“句子”）。
   CustomerID 已转 int64 写盘，避免 `17850.0` 这种浮点尾巴。
2. `models/item2vec.model` + `models/vector_index/` —— 指导书要求的规范命名。
   item2vec.model = 阶段五 Skip-gram 模型的规范副本（全量语料，可 Word2Vec.load）；
   vector_index = 全量 Skip-gram 商品向量索引（行归一化，keys.npy + vectors_unit.npy +
   meta.json），供 Flask /similar 与在线演示用。商品语义相似属静态知识，无用户评估泄漏。
3. `outputs/item2vec/annotation_candidates.json` —— 5 个高频商品的 Skip-gram Top-5 邻居
   （含英文/中文描述、类目、买家数），供 Word 里「5 组人工标注」逐条打 合理/存疑/失败。

运行：venv\\Scripts\\python.exe scripts\\10_user_sequences_and_index.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from config.settings import COLUMNS, PATHS, SEED, set_seed
from src import embedding
from src.item_similarity import build_item_catalog
from src.recommender import load_clean, load_products_zh

set_seed(SEED)

_USER, _ITEM, _TIME, _INVOICE = COLUMNS["user"], COLUMNS["item"], COLUMNS["time"], COLUMNS["invoice"]
PRICE_COL = COLUMNS["price"]

# 人工标注用的 5 个高频商品：按买家数在词表内排前 8（635~881 买家），
# 且分属不同类目（厨房/家居/派对/购物袋/装饰），保证标注样例多样。
ANCHORS = ["22423", "85123A", "47566", "85099B", "84879"]
TOP_K = 5


def build_user_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """长表用户序列：按 (用户, 时间, 单号, 行序) 排序，记录订单内行序。"""
    d = df.copy()
    d = d.assign(_r=np.arange(len(d)))
    d = d.sort_values([_USER, _TIME, _INVOICE, "_r"], kind="mergesort")
    d = d.assign(
        seq_position=d.groupby([_USER, _INVOICE], sort=False).cumcount().astype("int64")
    )
    out = d[[_USER, _INVOICE, _TIME, _ITEM, "seq_position"]].rename(columns={
        _USER: "customer_id", _INVOICE: "invoice_no",
        _TIME: "invoice_date", _ITEM: "stockcode",
    })
    # ISO 字符串：便于 Word/Excel/跨工具查看，不再带“月/日”歧义
    out["invoice_date"] = out["invoice_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return out


def save_vector_index(keys, vecs) -> None:
    """全量 Skip-gram 商品向量的行归一化索引（在线相似查询用）。"""
    unit = np.asarray(vecs, dtype=np.float32)
    unit = unit / np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
    d = PATHS["vector_index_dir"]
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "keys.npy", np.asarray(keys, dtype=str), allow_pickle=False)
    np.save(d / "vectors_unit.npy", unit, allow_pickle=False)
    meta = {
        "method": "skip-gram",
        "source_vectors": str(PATHS["models_dir"] / "item2vec_vectors_skip.npz"),
        "vocab_size": len(keys),
        "dim": int(vecs.shape[1]),
        "unit_normalized": True,
        "note": "全量语料商品语义索引（静态知识），供 /similar 与在线演示；离线评估用训练期-only 向量",
    }
    (d / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[向量索引] {d}  keys={len(keys)} dim={int(vecs.shape[1])} (unit)")


def write_annotation_candidates(df: pd.DataFrame, keys, vecs) -> None:
    """5 个高频商品 × Skip-gram Top-5 → annotation_candidates.json。"""
    zh = load_products_zh()
    catalog = build_item_catalog(df)            # 英文描述 / 单价 / 买家数 / 订单数
    key_set = set(keys)
    missing = [a for a in ANCHORS if a not in key_set]
    if missing:
        print(f"[警告] 锚点不在词表: {missing}")
    buyers = df.groupby(_ITEM)[_USER].nunique()
    orders = df.groupby(_ITEM)[_INVOICE].nunique()
    sims = embedding.top_k_similar_from_vectors(keys, vecs, top_k=TOP_K)

    def info(code: str) -> dict:
        c = catalog.get(code, {})
        z = zh.get(code, {})
        return {
            "code": code,
            "desc": c.get("desc", ""),
            "desc_zh": z.get("desc_zh", ""),
            "category": z.get("category", ""),
            "buyers": int(buyers.get(code, 0)),
            "orders": int(orders.get(code, 0)),
        }

    anchors = []
    for code in ANCHORS:
        if code not in key_set:
            anchors.append({"code": code, "in_vocab": False})
            continue
        entry = info(code)
        entry["neighbors"] = [
            dict(info(nb), score=round(sc, 4)) for nb, sc in sims.get(code, [])
        ]
        anchors.append(entry)

    out = {
        "task": "对『商品A→相似商品Top5』是否合理做人工标注：合理(主题成套/像搭配)/存疑/失败(无关)",
        "method": "skip-gram",
        "topk": TOP_K,
        "anchors": anchors,
    }
    path = PROJECT_ROOT / "outputs" / "item2vec" / "annotation_candidates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[标注候选] {path}")


def main() -> None:
    df = load_clean()
    print(f"清洗后交易 {len(df):,} 行 / {df[_USER].nunique():,} 用户 / {df[_INVOICE].nunique():,} 订单")

    # 1) 用户序列（CustomerID 已由 load_clean 转 int64）
    seq = build_user_sequences(df)
    PATHS["user_sequences"].parent.mkdir(parents=True, exist_ok=True)
    seq.to_csv(PATHS["user_sequences"], index=False, encoding="utf-8")
    print(f"[序列] {PATHS['user_sequences']}  行数={len(seq):,} "
          f"用户数={seq['customer_id'].nunique():,} 订单数={seq['invoice_no'].nunique():,}")

    # 2) 规范命名：item2vec.model = Skip-gram 全量模型副本
    skip_model_path = PATHS["models_dir"] / "item2vec_skip.model"
    m = Word2Vec.load(str(skip_model_path))
    m.save(str(PATHS["item2vec_model"]))
    print(f"[模型] {PATHS['item2vec_model']}  vocab={len(m.wv.index_to_key)} dim={m.wv.vector_size}")

    # 3) 向量索引 + 标注候选
    keys, vecs = embedding.load_vectors(str(PATHS["models_dir"] / "item2vec_vectors_skip.npz"))
    save_vector_index(keys, vecs)
    write_annotation_candidates(df, keys, vecs)


if __name__ == "__main__":
    main()
