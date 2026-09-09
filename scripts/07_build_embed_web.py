"""阶段五配套：由训练好的 CBOW / Skip-gram 商品向量计算「嵌入相似商品」Top-K，
并写成一个网页数据文件 web/data/embed_sim.js，供相似商品页新增的
「ItemCF | CBOW 嵌入 | Skip-gram 嵌入」切换栏使用。

同时在本脚本里做两类统计，作为 Word 文档「主观评价 / 与 ItemCF 对比」的素材：
    1) 锚点商品 Top-K 的三种方法并列对比
    2) 全商品层面：三种方法 Top10 列表之间的 Jaccard 重叠（中位数）

运行：venv\\Scripts\\python.exe scripts\\07_build_embed_web.py
（需先运行 scripts/06_train_item2vec.py）
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from config.settings import COLUMNS, PATHS
from src.embedding import load_vectors, top_k_similar_from_vectors
from src.item_similarity import build_item_catalog, top_k_similar_rows
from src.user_similarity import load_clean_transactions, load_purchase_matrix

TOPK = 10
OUT_JS = PROJECT_ROOT / "web" / "data" / "embed_sim.js"

ANCHORS = ["85123A", "85099B", "22423", "22138", "22633", "21730"]


def _js(name: str, payload: dict) -> str:
    return f"/* 由 scripts/07_build_embed_web.py 自动生成，请勿手改 */\nwindow.{name} = " + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ) + ";\n"


def main() -> None:
    df = load_clean_transactions().copy()
    df[COLUMNS["user"]] = df[COLUMNS["user"]].astype("int64")
    df[COLUMNS["item"]] = df[COLUMNS["item"]].astype(str)
    cat = build_item_catalog(df)

    # ---------- 两种嵌入方法：向量 → Top-K 相似商品 ----------
    recs: dict[str, dict[str, list]] = {}
    vocabs: dict[str, list[str]] = {}
    for key in ("cbow", "skip"):
        keys, vecs = load_vectors(str(PATHS["models_dir"] / f"item2vec_vectors_{key}.npz"))
        vocabs[key] = [str(k) for k in keys]
        top = top_k_similar_from_vectors(keys, vecs, top_k=TOPK)
        recs[key] = {str(c): [[str(b), s] for b, s in v] for c, v in top.items()}

    vocab_c = set(vocabs["cbow"])
    vocab_s = set(vocabs["skip"])
    assert vocab_c == vocab_s, "两个模型的词表应一致"
    vocab = vocab_c

    # ---------- ItemCF（同一张购买矩阵的物品余弦，与网页 ItemCF 一致）----------
    M, _, item_ids = load_purchase_matrix(df)
    MI = M.T.tocsr()
    item_top = top_k_similar_rows(MI, item_ids, top_k=TOPK, batch=128)
    itemcf = {str(c): [[str(b), s] for b, s in v] for c, v in item_top.items()}

    # ---------- 写网页数据 ----------
    hyper_json = json.loads(
        (PATHS["outputs_dir"] / "item2vec" / "hyperparams.json").read_text(encoding="utf-8")
    )
    payload = {
        "meta": {
            "label": "嵌入相似商品",
            "topK": TOPK,
            "hyper": hyper_json["hyper"],
            "records": {k: {kk: vv for kk, vv in v.items() if kk != "per_epoch_loss"}
                        for k, v in hyper_json["records"].items()},
            "vocabSize": len(vocab),
            "nItems": len(cat),
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        "vocab": sorted(vocab),
        "rec": {"cbow": recs["cbow"], "skip": recs["skip"]},
    }
    OUT_JS.write_text(_js("EMBED_SIM", payload), encoding="utf-8")
    print(f"已写出 {OUT_JS.relative_to(PROJECT_ROOT)}  {OUT_JS.stat().st_size/1024:.0f} KB")

    # ---------- 统计 1：锚点商品三方法 Top5 并列 ----------
    for a in ANCHORS:
        if a not in vocab:
            print(f"[skip] 锚点 {a} 不在词表")
            continue
        print("\n" + "=" * 96)
        print(f"锚点 {a}  {cat[a]['desc']}  (买家 {cat[a]['buyers']} 人)")
        print("=" * 96)
        def rows(name, data):
            lst = data.get(a, [])[:5]
            return " | ".join(f"{c}:{s:.3f}" for c, s in lst)
        print(f"  ItemCF        {rows('itemcf', itemcf)}")
        print(f"  CBOW          {rows('cbow', recs['cbow'])}")
        print(f"  Skip-gram     {rows('skip', recs['skip'])}")

    # ---------- 统计 2：全商品三方法 Top10 列表重叠（Jaccard 中位数）----------
    def jac(u: set, v: set) -> float:
        return len(u & v) / max(1, len(u | v))

    common = [c for c in itemcf if c in vocab]
    common = [c for c in common if itemcf[c] and recs["cbow"].get(c) and recs["skip"].get(c)]
    pair = {"ItemCF∩CBOW": [], "ItemCF∩Skip": [], "CBOW∩Skip": []}
    for c in common:
        s_itemcf = {b for b, _ in itemcf[c]}
        s_cbow = {b for b, _ in recs["cbow"][c]}
        s_skip = {b for b, _ in recs["skip"][c]}
        pair["ItemCF∩CBOW"].append(jac(s_itemcf, s_cbow))
        pair["ItemCF∩Skip"].append(jac(s_itemcf, s_skip))
        pair["CBOW∩Skip"].append(jac(s_cbow, s_skip))

    # 与 itemcf、cbow、skip 三方都有的商品个数
    print("\n" + "=" * 96)
    print(f"全商品统计（三种方法均有 Top{TOPK} 的商品 n = {len(common)}）")
    print("=" * 96)
    for k, vals in pair.items():
        print(f"  Top10 列表重叠中位 Jaccard  {k:<14}: {np.median(vals):.3f}")

    # Top1 相似度量级（嵌入余弦 vs 物品余弦）
    med1 = lambda d: float(np.median([v[0][1] for v in d.values() if v]))
    print(f"  Top1 相似度中位：ItemCF={med1(itemcf):.3f}  CBOW={med1(recs['cbow']):.3f}"
          f"  Skip-gram={med1(recs['skip']):.3f}")
    # 首位命中一致比例（CBOW 与 Skip-gram 的 Top1 相同）
    same_top1 = sum(
        1 for c in common
        if recs["cbow"][c][0][0] == recs["skip"][c][0][0]
    ) / len(common)
    print(f"  CBOW 与 Skip-gram Top1 完全一致的比例: {same_top1:.3f}")
    print(f"  说明：物品余弦(ItemCF)Top1 中位最高，但包含单买家满格 1.00 的假象；"
          f"嵌入分值更平缓、几乎不会出现 1.00。")


if __name__ == "__main__":
    main()
