"""候选集打分 v4 提交：LGB(macro, 全月)×0.2 + LR(macro_sp, 最近2月)×0.8 —— 全局百分位秩融合。

依据（scripts/40/41/42 OOF）：
    * 老师口径 ≈ 逐用户列表；LR 组内 AUC 最高；
    * LR 训练窗口截断到最近 2 个月更好（gAUC_m .8669→.8684）；
    * 融合 (0.2 LGB-macro + 0.8 LR-last2) mgAUC=0.8703（v3-fuse 0.8682）。
LGB 对更远月数据仍有增益 ⇒ LGB 全月、LR 近窗——两种模型偏好相反，融合互偿。

打分 cut=11-01（leader_clean 全量），特征 44，58,205 全行覆盖。

用法：venv\\Scripts\\python.exe scripts\\43_build_submission_v4.py
产物：outputs/candidate/sample_submission_cand-align-v4-fuse.csv + models/candidate_*.pkl + meta
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
NAME = "cand-align-v4-fuse"
W = {"lgb_macro": 0.2, "lr": 0.8}            # 来自 fusion_v2b_report（mgAUC 0.8703）
LR_LAST_N = 2                                 # LR 只用最近 N 个训练月


def load_months(months) -> pd.DataFrame:
    return pd.concat([pd.read_csv(REPLAY_DIR / f"wide_v2_{m[:7]}.csv",
                                  dtype={"item_id": str}) for m in months],
                     ignore_index=True)


def macro_weight(df: pd.DataFrame, *, with_sp: bool):
    nrow_u = df.groupby("user_id").size()
    macro = 1.0 / np.sqrt(nrow_u.reindex(df["user_id"]).to_numpy(dtype="float64"))
    if not with_sp:
        return macro.astype("float64")
    spw = float((df["label"] == 0).sum() / max(1, int(df["label"].sum())))
    return (macro * np.where(df["label"] == 1, spw, 1.0)).astype("float64")


def main() -> None:
    t0 = time.time()
    # ---- LGB：全月 macro ----
    tr_all = load_months(MONTHS)
    from lightgbm import LGBMClassifier
    clf_lgb = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                             min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=1.0, random_state=SEED, verbosity=-1,
                             deterministic=True, n_jobs=1)
    clf_lgb.fit(tr_all[FEATS], tr_all["label"],
                sample_weight=macro_weight(tr_all, with_sp=False))
    print(f"[1/3] LGB(macro, 全月) 完成 {len(tr_all):,} 行", flush=True)

    # ---- LR：最近 LR_LAST_N 个月 macro_sp ----
    tr_last = load_months(MONTHS[-LR_LAST_N:])
    lr = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=2000, C=1.0, random_state=SEED))
    lr.fit(tr_last[FEATS], tr_last["label"],
           logisticregression__sample_weight=macro_weight(tr_last, with_sp=True))
    print(f"[2/3] LR(macro_sp, 最近{LR_LAST_N}月={MONTHS[-LR_LAST_N:][0][:7]}.."
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
    s_lr = lr.predict_proba(Xs)[:, 1].astype("float64")
    N = len(df)
    fused = W["lgb_macro"] * rankdata(s_lgb) / N + W["lr"] * rankdata(s_lr) / N

    out = pd.DataFrame({"user_id": df["user_id"], "item_id": df["item_id"], "score": fused})
    assert len(out) == len(raw) and out.isna().sum().sum() == 0
    out_path = PROJECT_ROOT / "outputs" / "candidate" / f"sample_submission_{NAME}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")

    import joblib
    joblib.dump(clf_lgb, str(PROJECT_ROOT / "models" / f"candidate_{NAME}_lgb.pkl"))
    joblib.dump(lr, str(PROJECT_ROOT / "models" / f"candidate_{NAME}_lr.pkl"))
    meta = {"name": NAME,
            "desc": "fusion LGB(macro,all)+LR(macro_sp,last2) global-percentile-rank",
            "weights": W, "lr_last_n_months": LR_LAST_N,
            "train_months": MONTHS, "n_rows": len(tr_all), "n_rows_lr": int(len(tr_last)),
            "score_cut": str(SC_CUT.date()), "feature_order": FEATS,
            "n_feature_cols": len(FEATS)}
    (PROJECT_ROOT / "models" / f"candidate_meta_{NAME}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[3/3] {NAME} → {out_path.name}")
    print(f"[覆盖] {len(out):,} 行；score mean={fused.mean():.4f} min={fused.min():.4f} "
          f"max={fused.max():.4f}")
    print(f"[meta] candidate_meta_{NAME}.json；总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
