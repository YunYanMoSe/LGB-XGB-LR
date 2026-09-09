"""候选集打分 v2 提交：全回放(44 特征)重训 → cut=11-01 对 58,205 候选打分。

与 scripts/31 同一时间序对齐思路，仅特征从 33 → 44（+11 v2）：
训练样本 = wide_v2_2010-{06..10}.csv（script 32 建），列序 = FEATURES_CAND + V2_FEATS；
打分特征 = 同一套 features_for_pairs(33) ⊕ v2_features(11)，在 cut=11-01 leader_clean 上重算，
v2 特征宇宙 = 全量候选行（与训练定义同函数 ⇒ 无训练/打分分叉）。

用法：venv\\Scripts\\python.exe scripts\\33_build_submission_v2.py --weight macro|macro_sp --version cand-align-v2-macro
产物：outputs/candidate/sample_submission_<version>.csv + models/candidate_<version>.pkl + meta
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SEED, set_seed
from src import candidate as cand
from src.candidate_v2 import v2_features, V2_FEATS

set_seed(SEED)

REPLAY_DIR = PROJECT_ROOT / "outputs" / "candidate" / "replay"
LEADER_CLEAN = PROJECT_ROOT / "data" / "processed" / "leader_clean.csv"
CAND_INPUT = PROJECT_ROOT / "train_data" / "sample_candidates.csv"
SC_CUT = pd.Timestamp("2010-11-01")
SC_VEC = PROJECT_ROOT / "outputs" / "candidate" / "v9_vecs_20101101_skip.npz"
BASE_FEATS = cand.FEATURES_CAND
FEATS = list(BASE_FEATS) + list(V2_FEATS)
MONTHS = [f"2010-{m:02d}-01" for m in range(6, 11)]


def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="macro_sp", choices=["scale_pos", "macro", "macro_sp"])
    ap.add_argument("--version", default="cand-align-v2-macrosp")
    args = ap.parse_args()
    name = args.version

    # ---- 1. 训练：wide_v2 全 5 月，取 FEATS 列 ----
    tr = pd.concat([pd.read_csv(REPLAY_DIR / f"wide_v2_{m[:7]}.csv",
                                dtype={"item_id": str}) for m in MONTHS],
                   ignore_index=True)
    assert list(tr.columns[2: 2 + len(BASE_FEATS)]) == BASE_FEATS  # user_id,item_id 之后是 33 基础特征
    Xtr = tr[FEATS].copy()
    spw = float((tr["label"] == 0).sum() / max(1, int(tr["label"].sum())))
    sw = None
    use_spw = 1.0
    if args.weight == "scale_pos":
        use_spw = spw
    elif args.weight in ("macro", "macro_sp"):
        nrow_u = tr.groupby("user_id").size()
        macro = 1.0 / np.sqrt(nrow_u.reindex(tr["user_id"]).to_numpy(dtype="float64"))
        sw = macro if args.weight == "macro" else \
            (macro * np.where(tr["label"] == 1, spw, 1.0)).astype("float64")
    print(f"[1/3] 训练 {len(tr):,} 行 / {len(FEATS)} 特征 / weight={args.weight}", flush=True)

    from lightgbm import LGBMClassifier
    clf = LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=31, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=use_spw,
        random_state=SEED, verbosity=-1, deterministic=True, n_jobs=1)
    clf.fit(Xtr, tr["label"], sample_weight=sw)
    print(f"[2/3] LGB 完成（{clf.n_estimators} 树）", flush=True)

    # ---- 3. 打分（cut 11-01）----
    clean = cand.load_clean_train(LEADER_CLEAN)
    raw = pd.read_csv(CAND_INPUT, dtype={c: str for c in ["user_id", "item_id"]})
    df = raw[["user_id", "item_id"]].copy()
    df["user_id"] = pd.to_numeric(df["user_id"]).astype("int64")
    df["item_id"] = df["item_id"].astype(str)

    bundle = cand.build_bundle(clean, cut=SC_CUT, vec_cache=SC_VEC)
    Xb = cand.features_for_pairs(bundle, df["user_id"].to_numpy(), df["item_id"].to_numpy())
    Xv = v2_features(clean, SC_CUT, df)
    Xs = pd.concat([Xb.reset_index(drop=True), Xv.reset_index(drop=True)], axis=1)
    assert list(Xs.columns) == FEATS and Xs.isna().sum().sum() == 0
    score = clf.predict_proba(Xs)[:, 1].astype(np.float64)

    out = pd.DataFrame({"user_id": df["user_id"], "item_id": df["item_id"], "score": score})
    assert len(out) == len(raw) and out.isna().sum().sum() == 0
    out_path = PROJECT_ROOT / "outputs" / "candidate" / f"sample_submission_{name}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")

    import joblib
    joblib.dump(clf, str(PROJECT_ROOT / "models" / f"candidate_{name}.pkl"))
    meta = {"name": name, "desc": "candidate-aligned replay LGB v2 (33 base + 11 v2)",
            "train_months": MONTHS, "n_train_rows": int(len(tr)),
            "pos_rate": float(tr["label"].mean()), "weight": args.weight,
            "score_cut": str(SC_CUT.date()), "feature_order": FEATS,
            "n_feature_cols": len(FEATS),
            "lgb_params": {"n_estimators": int(clf.n_estimators), "learning_rate": 0.05,
                           "num_leaves": 31, "scale_pos_weight": float(use_spw)}}
    meta_path = PROJECT_ROOT / "models" / f"candidate_meta_{name}.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    owned = sum(1 for u, c in zip(df["user_id"], df["item_id"])
                if int(u) in bundle["owned_fp"] and str(c) in bundle["owned_fp"][int(u)])
    print(f"[版本] {name} → {out_path.name}")
    print(f"[覆盖] {len(out):,} 行；fp 已购 {owned:,} (~{owned/len(out):.2%})")
    print(f"[分数] min={score.min():.4f} mean={score.mean():.4f} max={score.max():.4f} "
          f"std={score.std():.4f}")
    print(f"[meta] {meta_path.name}；总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
