"""按用户要求产出两个「OOF 判不出差但值得上板一试」的提交（一次脚本建两个）。

A. cand-align-v7-tune : 组等权重调参冠军（scripts/53；池化信息之前）：xb_d5(depth5)×.4
   + LR(C=3, 近2月)×.6，Jun..Oct 池，gAUC_m .8722（史上最高，仅 +.0001，噪声级故当时未发）。
B. cand-align-v8-ext   : 回放前伸冠军（scripts/57）：LGB(base)×.15 + XGB(base)×.225 +
   LR(C=1, 近2月)×.625，Feb..Oct 9 月池（serve 树吃 9 个月），gAUC_m .8720。

两者都与 v6 同一打分管线：cut=11-01（leader_clean 全量）、特征 44、58,205 全行覆盖、
全局百分位秩融合。仅当用户想上板试奇效时使用；OOF 显示均 ≈ v6（.8721），不保证更高。

用法：venv\\Scripts\\python.exe scripts\\58_build_submission_v7tune_v8ext.py
产物：outputs/candidate/sample_submission_cand-align-v7-tune.csv
      outputs/candidate/sample_submission_cand-align-v8-ext.csv（+ 各 meta json）
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
OUT_DIR = PROJECT_ROOT / "outputs" / "candidate"
SC_CUT = pd.Timestamp("2010-11-01")
SC_VEC = PROJECT_ROOT / "outputs" / "candidate" / "v9_vecs_20101101_skip.npz"
FEATS = list(cand.FEATURES_CAND) + list(V2_FEATS)

MA = [f"2010-{m:02d}-01" for m in range(6, 11)]       # A: Jun..Oct
MB = [f"2010-{m:02d}-01" for m in range(2, 11)]       # B: Feb..Oct


def load_months(months) -> pd.DataFrame:
    return pd.concat([pd.read_csv(REPLAY_DIR / f"wide_v2_{m[:7]}.csv",
                                  dtype={"item_id": str}) for m in months],
                     ignore_index=True)


def eq_group_weight(df: pd.DataFrame, *, with_sp: bool):
    n_g = df.groupby(["user_id", "month"], sort=True).size()
    keys = pd.MultiIndex.from_arrays([df["user_id"], df["month"].astype(str)])
    w = 1.0 / n_g.reindex(keys).to_numpy(dtype="float64")
    if not with_sp:
        return w.astype("float64")
    spw = float((df["label"] == 0).sum() / max(1, int(df["label"].sum())))
    return (w * np.where(df["label"] == 1, spw, 1.0)).astype("float64")


def fit_xgb(tr, *, depth):
    from xgboost import XGBClassifier
    clf = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=depth,
                        subsample=0.8, colsample_bytree=0.8,
                        tree_method="hist", n_jobs=1, random_state=SEED,
                        eval_metric="auc", verbosity=0)
    clf.fit(tr[FEATS], tr["label"], sample_weight=eq_group_weight(tr, with_sp=False))
    return clf


def fit_lgb(tr):
    from lightgbm import LGBMClassifier
    clf = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                         min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=1.0, random_state=SEED, verbosity=-1,
                         deterministic=True, n_jobs=1)
    clf.fit(tr[FEATS], tr["label"], sample_weight=eq_group_weight(tr, with_sp=False))
    return clf


def fit_lr(tr, *, C):
    lr = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=2000, C=C, random_state=SEED))
    lr.fit(tr[FEATS], tr["label"],
           logisticregression__sample_weight=eq_group_weight(tr, with_sp=True))
    return lr


def write(name: str, fused: np.ndarray, df: pd.DataFrame, raw_len: int,
          meta: dict) -> None:
    out = pd.DataFrame({"user_id": df["user_id"], "item_id": df["item_id"], "score": fused})
    assert len(out) == raw_len and out.isna().sum().sum() == 0
    path = OUT_DIR / f"sample_submission_{name}.csv"
    out.to_csv(path, index=False, encoding="utf-8")
    (OUT_DIR.parent.parent / "models" / f"candidate_meta_{name}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {name} → {path.name}（{len(out):,} 行）", flush=True)


def main() -> None:
    t_all = time.time()
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
    print(f"[score] 特征就绪 {Xs.shape}（{time.time()-t_all:.0f}s）", flush=True)
    N = len(df)

    # ============ A. cand-align-v7-tune：xb_d5×.4 + lr_C3×.6（Jun..Oct 池） ============
    t0 = time.time()
    trA = load_months(MA)
    xgbA = fit_xgb(trA, depth=5)
    lrA = fit_lr(load_months(MA[-2:]), C=3.0)
    s_xa = xgbA.predict_proba(Xs)[:, 1].astype("float64")
    s_la = lrA.predict_proba(Xs)[:, 1].astype("float64")
    fusedA = 0.4 * rankdata(s_xa) / N + 0.6 * rankdata(s_la) / N
    write("cand-align-v7-tune", fusedA, df, len(raw),
          {"desc": "xb_d5(eq,Jun..Oct)x.4 + lr(C3,eq,sp,last2)x.6 全局秩；53 冠军 gAUC_m .8722",
           "weights": {"xgb_d5": 0.4, "lr_C3": 0.6}, "oof_gAUC_m": 0.8722,
           "score_cut": str(SC_CUT.date())})
    print(f"[A] cand-align-v7-tune 完成（{time.time()-t0:.0f}s）", flush=True)

    # ============ B. cand-align-v8-ext：lgb.15+xgb.225+lr.625（Feb..Oct 池） ============
    t0 = time.time()
    trB = load_months(MB)
    lgbB = fit_lgb(trB)
    xgbB = fit_xgb(trB, depth=6)
    lrB = fit_lr(load_months(MB[-2:]), C=1.0)
    s_lb = lgbB.predict_proba(Xs)[:, 1].astype("float64")
    s_xb = xgbB.predict_proba(Xs)[:, 1].astype("float64")
    s_lrb = lrB.predict_proba(Xs)[:, 1].astype("float64")
    fusedB = (0.15 * rankdata(s_lb) / N + 0.225 * rankdata(s_xb) / N
              + 0.625 * rankdata(s_lrb) / N)
    write("cand-align-v8-ext", fusedB, df, len(raw),
          {"desc": "LGB(eq,Feb..Oct)x.15 + XGB(eq,Feb..Oct)x.225 + LR(eq,sp,last2)x.625 "
                   "全局秩；57 前伸冠军 gAUC_m .8720",
           "weights": {"lgb": 0.15, "xgb": 0.225, "lr": 0.625},
           "train_pool": "2010-02..10 (9 月)", "oof_gAUC_m": 0.8720,
           "score_cut": str(SC_CUT.date())})
    print(f"[B] cand-align-v8-ext 完成（{time.time()-t0:.0f}s）", flush=True)

    print(f"[done] 总耗时 {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
