"""候选集打分 v7 提交：池化AUC 选权探针（复用 v6 模型，仅换融合权重）。

组长 3 探针测得老师评分 ≈ 全局池化 AUC。本地验证（scripts/54）：按池化AUC 选权，
冠军权从 v6 的 (lgb.1, xgb.3, lr.6) 漂移到 (lgb .075, xgb .55, lr .375)，OOF 池化AUC
.8872→.8883（gAUC_m 同步掉到 .8714）——两口径不再一致。

v7 ≡ v6 分量（同一 lgb_macro_eq / xgb_macro_eq / lr_eq_l2_sp 模型，直接读已存 pkl，
不重训），仅融合权重按池化AUC-argmax。上板与 v6(0.7782) 对照，检验组长口径。
若 v7 > v6 ⇒ 按池化口径继续深挖；否则回退 gAUC_m 口径。

打分 cut=11-01（leader_clean 全量），特征 44，58,205 全行覆盖。

用法：venv\\Scripts\\python.exe scripts\\55_build_submission_v7_pooled.py
产物：outputs/candidate/sample_submission_cand-align-v7-pooled.csv
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import joblib
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import set_seed, SEED
from src import candidate as cand
from src.candidate_v2 import v2_features, V2_FEATS

set_seed(SEED)

LEADER_CLEAN = PROJECT_ROOT / "data" / "processed" / "leader_clean.csv"
CAND_INPUT = PROJECT_ROOT / "train_data" / "sample_candidates.csv"
SC_CUT = pd.Timestamp("2010-11-01")
SC_VEC = PROJECT_ROOT / "outputs" / "candidate" / "v9_vecs_20101101_skip.npz"
BASE_FEATS = cand.FEATURES_CAND
FEATS = list(BASE_FEATS) + list(V2_FEATS)
NAME = "cand-align-v7-pooled"
# 池化AUC-argmax（scripts/54 对 v6 三分量 OOF 网格）：pooledAUC .8883 / gAUC_m .8714
W = {"lgb_macro_eq": 0.075, "xgb_macro_eq": 0.55, "lr_eq_l2_sp": 0.375}
V6_PKLS = {k: PROJECT_ROOT / "models" / f"candidate_cand-align-v6-fuse_{k}.pkl"
           for k in ("lgb", "xgb", "lr")}


def main() -> None:
    t0 = time.time()
    for p in V6_PKLS.values():
        if not p.exists():
            raise SystemExit(f"缺 v6 模型 {p.name}，先跑 scripts/49_build_submission_v6.py")
    clf_lgb = joblib.load(str(V6_PKLS["lgb"]))
    clf_xgb = joblib.load(str(V6_PKLS["xgb"]))
    lr = joblib.load(str(V6_PKLS["lr"]))

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
    print(f"[score] 特征就绪 {Xs.shape}（{time.time()-t0:.0f}s）", flush=True)

    s_lgb = clf_lgb.predict_proba(Xs)[:, 1].astype("float64")
    s_xgb = clf_xgb.predict_proba(Xs)[:, 1].astype("float64")
    s_lr = lr.predict_proba(Xs)[:, 1].astype("float64")
    N = len(df)
    fused = (W["lgb_macro_eq"] * rankdata(s_lgb) / N
             + W["xgb_macro_eq"] * rankdata(s_xgb) / N
             + W["lr_eq_l2_sp"] * rankdata(s_lr) / N)

    out = pd.DataFrame({"user_id": df["user_id"], "item_id": df["item_id"], "score": fused})
    assert len(out) == len(raw) and out.isna().sum().sum() == 0
    out_path = PROJECT_ROOT / "outputs" / "candidate" / f"sample_submission_{NAME}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")

    meta = {"name": NAME,
            "desc": "v6 models re-fused with pooled-AUC-argmax weights "
                    "(same comps as cand-align-v6-fuse, only weights differ)",
            "weights": W, "oof_pooled_auc": 0.8883, "oof_gAUC_m": 0.8714,
            "score_cut": str(SC_CUT.date()), "n_rows": N}
    (PROJECT_ROOT / "models" / f"candidate_meta_{NAME}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[4/4] {NAME} → {out_path.name}（{N:,} 行；{time.time()-t0:.0f}s）")


if __name__ == "__main__":
    main()
