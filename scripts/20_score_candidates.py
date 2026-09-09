"""候选集打分任务 ③：对老师候选集 (user_id, item_id) 产出 3 列提交文件。

读候选集 2 列 → 用与训练完全相同的 33 列特征（features_for_pairs + 同一 cut/向量缓存）
载入 meta 中记录的模型（默认 cand-lgb-v1 即 LGB），score = predict_proba[:, 1]。

硬性约束（脚本内置断言）：
    * 输出恰 3 列：user_id, item_id, score（item_id 保持原始 StockCode 字符串原样）；
    * 行序与候选集输入一致；
    * 行数全等 ⇒ 覆盖 100% 候选行（漏行即错，故此处强制）。

用法：
    venv\\Scripts\\python.exe scripts\\20_score_candidates.py
    venv\\Scripts\\python.exe scripts\\20_score_candidates.py --input 某候选.csv --model lgb
"""
from __future__ import annotations

# 锁单线程数值库（与 scripts/19 一致），保证与训练同分布、逐位可复现。
import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from config.settings import SEED, set_seed
from src import candidate as cand

set_seed(SEED)

DEFAULT_INPUT = PROJECT_ROOT / "train_data" / "sample_candidates.csv"


def _resolve_cols(df: pd.DataFrame):
    """兼容两种列名习惯：user_id/CustomerID 与 item_id/StockCode/stock_code。"""
    lower = {c: str(c).lower() for c in df.columns}
    user_col = next((c for c in df.columns if lower[c] in {"user_id", "customerid"}), None)
    item_col = next((c for c in df.columns
                     if lower[c] in {"item_id", "stockcode", "stock_code"}), None)
    assert user_col is not None and item_col is not None, \
        f"候选集需含用户列与商品列，实际列：{list(df.columns)}"
    return user_col, item_col


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=None, help="缺省 = outputs/candidate/sample_submission_<version>.csv")
    ap.add_argument("--model", choices=["lgb", "lr"], default="lgb")
    ap.add_argument("--version", default=None, help="提交版本号，缺省读 meta.model_version")
    ap.add_argument("--cut", default=None,
                    help="打分特征切窗 YYYY-MM-DD（缺省 = meta 里的训练 cut）。"
                         "想用尽 train.csv 全部近况时传 2010-11-01（打分用向量会按该切窗单独重训缓存）。")
    args = ap.parse_args()

    # ---- meta / 模型 ----
    assert cand.CAND_META.exists(), "缺 models/candidate_meta.json，先跑 scripts/19"
    meta = json.loads(cand.CAND_META.read_text(encoding="utf-8"))
    version = args.version or meta["model_version"]
    feature_order = meta["feature_order"]
    assert feature_order == cand.FEATURES_CAND, "meta 特征列序 ≠ FEATURES_CAND"

    model_path = cand.LGB_PKL if args.model == "lgb" else cand.LR_PKL
    assert model_path.exists(), f"模型缺失：{model_path}，先跑 scripts/19"
    model = joblib.load(str(model_path))

    # ---- 数据 & 画像 ----
    # 打分 cut 可 ≠ 训练 cut：真榜在数据尾之后，打分特征切窗默认应尽量靠后（用尽全部历史）。
    cut = pd.Timestamp(args.cut) if args.cut else pd.Timestamp(meta["cut"])
    if args.cut and cut != pd.Timestamp(meta["cut"]):
        print(f"[切窗] 打分特征用 {cut.date()}（≠ 训练 {meta['cut']}），比训练多利用了后面的近况")
    vec_cache = cand.VEC_CACHE
    if cut != pd.Timestamp(meta["cut"]):
        vec_cache = cand.VEC_CACHE.parent / f"vectors_{cut:%Y%m%d}_skip.npz"
        print(f"[向量] 该切窗独立向量缓存 -> {vec_cache.name}")
    clean = cand.load_clean_train()
    bundle = cand.build_bundle(clean, cut=cut, vec_cache=vec_cache)

    # ---- 读候选集（item_id 强制字符串，保留原 token，含清洗删掉的服务码亦原样 round-trip）----
    raw = pd.read_csv(args.input, dtype={c: str for c in pd.read_csv(args.input, nrows=0).columns})
    user_col, item_col = _resolve_cols(raw)
    df = raw[[user_col, item_col]].copy()
    df.columns = ["user_id", "item_id"]
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("int64")
    df = df.dropna(subset=["user_id", "item_id"]).reset_index(drop=True)
    assert df["item_id"].astype(str).str.strip().ne("").all(), "候选集存在空 item_id"
    n_in = len(df)
    users = df["user_id"].to_numpy(dtype="int64")
    codes = df["item_id"].astype(str).to_numpy(dtype=object)
    print(f"[候选] {n_in:,} 行；用户 {df['user_id'].nunique():,}；商品 {df['item_id'].nunique():,}")

    # ---- 打分 ----
    X = cand.features_for_pairs(bundle, users, codes)
    assert list(X.columns) == feature_order, "打分特征列序 ≠ meta.feature_order"
    assert X.isna().sum().sum() == 0, "打分特征含 NaN"
    Xa = X.to_numpy(dtype=np.float64)
    score = model.predict_proba(Xa)[:, 1].astype(np.float64)
    assert np.isfinite(score).all(), "score 含非有限值"

    # ---- 覆盖断言：行数全等、顺序保持 ----
    assert len(score) == n_in == len(df), "输出行数与候选集不一致（漏行/多行）→ 会被拒评分"
    out = pd.DataFrame({"user_id": df["user_id"], "item_id": df["item_id"],
                        "score": np.round(score, 6)})
    assert out.isna().sum().sum() == 0, "输出含 NaN"

    out_path = (Path(args.output) if args.output
                else PROJECT_ROOT / "outputs" / "candidate" / f"sample_submission_{version}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")

    # ---- 汇总 ----
    cold_item = int((~df["item_id"].isin(set(bundle["catalog"].astype(str)))).sum())
    owned = sum(1 for u, c in zip(users, codes)
                if int(u) in bundle["owned_fp"] and str(c) in bundle["owned_fp"][int(u)])
    print(f"[覆盖] 输出 {len(out):,} 行 = 候选 {n_in:,} 行 ⇒ 覆盖率 100% ✅（行序保持输入序）")
    print(f"[口径] 其中候选商品不在特征期目录(冷商品) {cold_item:,} 行；fp 历史已购对 {owned:,} 行（~{owned / max(1, n_in):.1%}）")
    print(f"[分数] min={score.min():.4f}  mean={score.mean():.4f}  p50={np.median(score):.4f}  max={score.max():.4f}")
    print(f"[版本] 提交版本={version}（模型 {model_path.name}，打分特征 cut={cut.date()}，"
          f"{meta['n_feature_cols']} 特征；训练原版 {meta['model_version']} cut={meta['cut']}）→ {out_path}")
    print("\n前 5 行：")
    print(out.head().to_string(index=False))


if __name__ == "__main__":
    main()
