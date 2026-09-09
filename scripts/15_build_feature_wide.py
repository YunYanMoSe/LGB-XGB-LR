"""阶段七 ①：多维度特征 → 全数值特征宽表 CSV（工程化实现落盘）。

口径见 src/feature_engineering.build_feature_wide 模块 docstring：
    特征期 < 2011-11-09，标签窗 = 尾随 30 天；正样本=标签窗首购，负样本=配对 1:3；
    全部特征仅用特征期计算（防泄漏），Item2Vec 用特征期订单篮重训并缓存。

产出（outputs/feature_stage/）：
    feature_wide.csv          宽表：user_id, item_id, 31 特征, label（全数值）
    id_map.csv                StockCode ↔ 数值 item_id
    category_map.csv          类目收拢 raw→collapsed
    features_meta.json        样本数 / 切分 / 类目桶 / 可复现信息
    vectors_feat_skip.npz     特征期重训 Skip-gram 商品向量（缓存，16 复用时不再重训）

运行：venv\\Scripts\\python.exe scripts\\15_build_feature_wide.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.settings import COLUMNS, SEED, set_seed
from src import feature_engineering as fe
from src import recommender as rec

set_seed(SEED)
_USER, _ITEM, _TIME, _INVOICE = COLUMNS["user"], COLUMNS["item"], COLUMNS["time"], COLUMNS["invoice"]

STAGE = PROJECT_ROOT / "outputs" / "feature_stage"
STAGE.mkdir(parents=True, exist_ok=True)
VEC_CACHE = STAGE / "vectors_feat_skip.npz"
WIDE_CSV = STAGE / "feature_wide.csv"
ID_MAP_CSV = STAGE / "id_map.csv"
CAT_MAP_CSV = STAGE / "category_map.csv"
META_JSON = STAGE / "features_meta.json"


def main() -> None:
    t0 = time.time()
    df = rec.load_clean()
    fp, lp = fe.split_windows(df)

    print("=" * 72)
    print("阶段七 ① · 特征工程：多维度特征 → 全数值宽表")
    print("=" * 72)
    n_orders = fp[_INVOICE].nunique()
    n_users = fp[_USER].nunique()
    n_items = fp[_ITEM].nunique()
    n_cand = int((fp.groupby(_USER)[_INVOICE].nunique() >= fe.MIN_FP_ORDERS).sum())
    print(f"特征期：{len(fp):,} 行 / 订单 {n_orders:,} / 用户 {n_users:,} / 商品 {n_items:,}")
    print(f"标签窗：{len(lp):,} 行 / 有购用户 {lp[_USER].nunique():,}（尾随 30 天）")
    print(f"候选用户（特征期订单数 ≥ {fe.MIN_FP_ORDERS}）：{n_cand:,}")

    wide = fe.build_feature_wide(df, VEC_CACHE)
    n_pos = int((wide["label"] == 1).sum())
    n_neg = int((wide["label"] == 0).sum())

    # ---- 商品码 ↔ 数值 item_id（样本内全部商品，升序排序，可复现）----
    id_map = wide.attrs["id_map"]
    id_df = pd.DataFrame({"stock_code": sorted(id_map, key=str)})
    id_df.insert(0, "item_id", id_df["stock_code"].map(id_map))
    id_df.to_csv(ID_MAP_CSV, index=False, encoding="utf-8")

    # ---- 类目收拢映射（raw 类目 → collapsed）----
    collapse_raw = wide.attrs["collapse_raw"]
    cat_df = pd.DataFrame({"raw_category": sorted(collapse_raw, key=str)})
    cat_df["collapsed_category"] = cat_df["raw_category"].map(collapse_raw)
    cat_df["bucket_code"] = cat_df["collapsed_category"].map(
        {b["name"]: b["code"] for b in wide.attrs["buckets"]})
    cat_df.to_csv(CAT_MAP_CSV, index=False, encoding="utf-8")

    # ---- meta（不含训练结果；16 训练后追加 AUC/重要性）----
    meta = {
        "stage": "stage7_feature_engineering",
        "protocol": "特征期=InvoiceDate<2011-11-09；标签窗=尾随30天[2011-11-09, 2011-12-09]；"
                     "正样本=标签窗首购；负样本=配对1:3（全数据集未购买∩特征期目录）",
        "feat_cut": str(fe.FEAT_CUT.date()),
        "seed": SEED, "min_fp_orders": fe.MIN_FP_ORDERS,
        "neg_ratio": fe.NEG_RATIO,
        "feature_period": {"rows": int(len(fp)), "orders": int(n_orders),
                           "users": int(n_users), "items": int(n_items)},
        "label_period": {"rows": int(len(lp)), "users": int(lp[_USER].nunique())},
        "n_candidates": n_cand,
        "n_pos": n_pos, "n_neg": n_neg,
        "actual_neg_per_pos": round(n_neg / max(1, n_pos), 3),
        "n_sample_users": int(wide["user_id"].nunique()),
        "n_item_map": int(len(id_map)),
        "n_feature_cols": len(fe.FEATURES),
        "features": [{"name": f, "zh": fe.FEAT_ZH[f], "group": fe.GROUP[f]}
                     for f in fe.FEATURES],
        "cat_buckets": wide.attrs["buckets"],
        "country_cols": fe._COUNTRY_COLS,
        "wide_csv": str(WIDE_CSV.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "id_map_csv": str(ID_MAP_CSV.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "category_map_csv": str(CAT_MAP_CSV.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    META_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 宽表落盘 + 自检 ----
    wide.to_csv(WIDE_CSV, index=False, encoding="utf-8")
    fe.verify_wide(WIDE_CSV, ID_MAP_CSV, meta)
    print(f"\n宽表 -> {WIDE_CSV}  ({WIDE_CSV.stat().st_size/1024/1024:.1f} MB)")
    print(f"id_map -> {ID_MAP_CSV}；category_map -> {CAT_MAP_CSV}；meta -> {META_JSON}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
