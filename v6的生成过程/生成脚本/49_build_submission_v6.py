"""候选集打分 v6 提交：组等权三分量融合。

LGB(macro, 组等权, 全月)×.1 + XGB(macro, 组等权, 全月)×.3 + LR(macro_sp, 组等权, 最近2月)×.6
—— 全局百分位秩融合。依据 scripts/47/48 OOF：
    * 训练行权重 1/√组 → 1/组(组=(user,month)) 对齐 gAUC_m 口径：XGB 单模型 .8586→.8665，
      LR .8684→.8687，融合 .8710→.8721（gAUC_w .8576→.8580，AUC .8854→.8872）；
    * 冠军 (lgb_eq .1, xgb_eq .3, lr_eq .6) gAUC_m .8721，邻域 .8716-.8721 平台稳健。

打分 cut=11-01（leader_clean 全量），特征 44，58,205 全行覆盖。

用法：venv\\Scripts\\python.exe scripts\\49_build_submission_v6.py
产物：outputs/candidate/sample_submission_cand-align-v6-fuse.csv + models/candidate_*.pkl + meta
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
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
NAME = "cand-align-v6-fuse"
W = {"lgb_macro_eq": 0.1, "xgb_macro_eq": 0.3, "lr_eq_l2_sp": 0.6}  # scripts/48 mgAUC .8721
LR_LAST_N = 2


def load_months(months) -> pd.DataFrame:
    return pd.concat([pd.read_csv(REPLAY_DIR / f"wide_v2_{m[:7]}.csv",
                                  dtype={"item_id": str}) for m in months],
                     ignore_index=True)


def eq_group_weight(df: pd.DataFrame, *, with_sp: bool):
    """组(=(user,month))等权：w = 1 / n_group；可选 × spw（线性正例放大）。"""
    n_g = df.groupby(["user_id", "month"], sort=True).size()
    keys = pd.MultiIndex.from_arrays([df["user_id"], df["month"].astype(str)])
    w = 1.0 / n_g.reindex(keys).to_numpy(dtype="float64")
    if not with_sp:
        return w.astype("float64")
    spw = float((df["label"] == 0).sum() / max(1, int(df["label"].sum())))
    return (w * np.where(df["label"] == 1, spw, 1.0)).astype("float64")


def main() -> None:
    t0 = time.time()
    # ---- LGB 组等权 全月 ----
    tr_all = load_months(MONTHS)
    from lightgbm import LGBMClassifier
    clf_lgb = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                             min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=1.0, random_state=SEED, verbosity=-1,
                             deterministic=True, n_jobs=1)
    clf_lgb.fit(tr_all[FEATS], tr_all["label"],
                sample_weight=eq_group_weight(tr_all, with_sp=False))
    print(f"[1/4] LGB(macro, 组等权, 全月) 完成 {len(tr_all):,} 行", flush=True)

    # ---- XGB 组等权 全月 ----
    from xgboost import XGBClassifier
    clf_xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8,
                            tree_method="hist", n_jobs=1, random_state=SEED,
                            eval_metric="auc", verbosity=0)
    clf_xgb.fit(tr_all[FEATS], tr_all["label"],
                sample_weight=eq_group_weight(tr_all, with_sp=False))
    print(f"[2/4] XGB(macro, 组等权, 全月) 完成", flush=True)

    # ---- LR 组等权×spw 最近2月 ----
    tr_last = load_months(MONTHS[-LR_LAST_N:])
    lr = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=2000, C=1.0, random_state=SEED))
    lr.fit(tr_last[FEATS], tr_last["label"],
           logisticregression__sample_weight=eq_group_weight(tr_last, with_sp=True))
    print(f"[3/4] LR(macro_sp, 组等权, 最近{LR_LAST_N}月={MONTHS[-LR_LAST_N:][0][:7]}.."
          f"{MONTHS[-1][:7]}) 完成 {len(tr_last):,} 行", flush=True)

    # ---- 打分 cut=11-01 ----
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

    import joblib
    joblib.dump(clf_lgb, str(PROJECT_ROOT / "models" / f"candidate_{NAME}_lgb.pkl"))
    joblib.dump(clf_xgb, str(PROJECT_ROOT / "models" / f"candidate_{NAME}_xgb.pkl"))
    joblib.dump(lr, str(PROJECT_ROOT / "models" / f"candidate_{NAME}_lr.pkl"))
    meta = {"name": NAME,
            "desc": "fusion LGB(macro,eqgrp,all).1 + XGB(macro,eqgrp,all).3 + "
                    "LR(macro_sp,eqgrp,last2).6 global-percentile-rank",
            "weight_scheme": "eq-group (1/n per user-month) x spw for LR",
            "weights": W, "lr_last_n_months": LR_LAST_N,
            "train_months": MONTHS, "n_rows": len(tr_all), "n_rows_lr": int(len(tr_last)),
            "score_cut": str(SC_CUT.date()), "feature_order": FEATS,
            "n_feature_cols": len(FEATS)}
    (PROJECT_ROOT / "models" / f"candidate_meta_{NAME}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[4/4] {NAME} → {out_path.name}")
    print(f"[覆盖] {len(out):,} 行；score mean={fused.mean():.4f} min={fused.min():.4f} "
          f"max={fused.max():.4f}")
    print(f"[meta] candidate_meta_{NAME}.json；总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
