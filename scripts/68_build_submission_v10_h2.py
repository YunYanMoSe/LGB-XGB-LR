"""候选集打分 v10（视界 H≥2）：三分量重训在「m月∪(m+1)月买」标签，权重保持 v6 (.1,.3,.6)。

隔离标签效应：除训练标签外一切与 v6 相同（44 特征、cut 11-01、三分量配方、融合权）。
若老师测试窗 >1 个月（如 Nov+Dec），H2 模型比 v6 下月模型更对齐 → 榜应升；若老师只测 1 个月，
H2 应略低于 v6（一个仍带信息的负结果，确认视界=1 月）。真伪交给真榜，本地健康检查见 scripts/67。

serve 训练池 = Jun..Sep 行贴 H2 标签（Jun∪Jul…Sep∪Oct，最后一窗 Sep∪Oct 全可标注；Oct 行需 Nov → 弃）。
树吃全部 4 个 H2 月；LR 吃最近 2 个 H2 标注月（Aug、Sep 行）。

用法：venv\\Scripts\\python.exe scripts\\68_build_submission_v10_h2.py
产物：outputs/candidate/sample_submission_cand-align-v10-h2.csv + meta
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SEED, set_seed
from src import candidate as cand
from src.candidate_v2 import v2_features, V2_FEATS

set_seed(SEED)

REPLAY = PROJECT_ROOT / "outputs" / "candidate" / "replay"
LEADER_CLEAN = PROJECT_ROOT / "data" / "processed" / "leader_clean.csv"
CAND_INPUT = PROJECT_ROOT / "train_data" / "sample_candidates.csv"
SC_CUT = pd.Timestamp("2010-11-01")
SC_VEC = PROJECT_ROOT / "outputs" / "candidate" / "v9_vecs_20101101_skip.npz"
BASE_FEATS = cand.FEATURES_CAND
FEATS = list(BASE_FEATS) + list(V2_FEATS)
H2_MONTHS = [f"2010-{m:02d}-01" for m in range(6, 10)]  # Jun..Sep
W = {"lgb": 0.1, "xgb": 0.3, "lr": 0.6}   # 与 v6 相同，隔离标签效应
LR_LAST_N = 2
USER = "CustomerID"; ITEM = "StockCode"; TIME = "InvoiceDate"


def monthly_purchase_sets(clean):
    df = clean[[USER, ITEM, TIME]].copy()
    df["m"] = df[TIME].dt.strftime("%Y-%m")
    return {m: set(zip(g[USER].to_numpy(), g[ITEM].astype(str).to_numpy()))
            for m, g in df.groupby("m")}


def load_h2(month, buy_sets):
    w = pd.read_csv(REPLAY / f"wide_v2_{month[:7]}.csv", dtype={"item_id": str})
    nxt = f"2010-{int(month[5:7]) + 1:02d}"
    s = buy_sets.get(nxt, set())
    u = w["user_id"].to_numpy(dtype="int64")
    i = w["item_id"].astype(str).to_numpy()
    bn = np.fromiter(((a, b) in s for a, b in zip(u, i)), dtype=bool, count=len(w))
    w = w.copy()
    w["label"] = (w["label"].to_numpy(dtype="int64") | bn.astype("int64"))
    return w


def eq_group_sw(tr, with_sp):
    n_g = tr.groupby(["user_id", "month"], sort=True).size()
    keys = pd.MultiIndex.from_arrays([tr["user_id"], tr["month"].astype(str)])
    w = 1.0 / n_g.reindex(keys).to_numpy(dtype="float64")
    if not with_sp:
        return w.astype("float64")
    spw = float((tr["label"] == 0).sum() / max(1, int(tr["label"].sum())))
    return (w * np.where(tr["label"] == 1, spw, 1.0)).astype("float64")


def main() -> None:
    t_all = time.time()
    clean = cand.load_clean_train(LEADER_CLEAN)
    clean[TIME] = pd.to_datetime(clean[TIME])
    clean[USER] = pd.to_numeric(clean[USER], errors="coerce").astype("int64")
    clean[ITEM] = clean[ITEM].astype(str)
    buys = monthly_purchase_sets(clean)
    wide = {m: load_h2(m, buys) for m in H2_MONTHS}
    tr_all = pd.concat([wide[m] for m in H2_MONTHS], ignore_index=True)
    tr_last = pd.concat([wide[m] for m in H2_MONTHS[-LR_LAST_N:]], ignore_index=True)

    from lightgbm import LGBMClassifier
    clf_lgb = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                             min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=1.0, random_state=SEED, verbosity=-1,
                             deterministic=True, n_jobs=1)
    clf_lgb.fit(tr_all[FEATS], tr_all["label"],
                sample_weight=eq_group_sw(tr_all, with_sp=False))
    print(f"[1/4] LGB(H2, 组等权, Jun..Sep) {len(tr_all):,} 行 完成", flush=True)

    from xgboost import XGBClassifier
    clf_xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8,
                            tree_method="hist", n_jobs=1, random_state=SEED,
                            eval_metric="auc", verbosity=0)
    clf_xgb.fit(tr_all[FEATS], tr_all["label"],
                sample_weight=eq_group_sw(tr_all, with_sp=False))
    print("[2/4] XGB(H2, 组等权, Jun..Sep) 完成", flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    lr = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=2000, C=1.0, random_state=SEED))
    lr.fit(tr_last[FEATS], tr_last["label"],
           logisticregression__sample_weight=eq_group_sw(tr_last, with_sp=True))
    print(f"[3/4] LR(H2, macro_sp, 最近{LR_LAST_N}月) {len(tr_last):,} 行 完成", flush=True)

    raw = pd.read_csv(CAND_INPUT, dtype={c: str for c in ["user_id", "item_id"]})
    df = raw[["user_id", "item_id"]].copy()
    df["user_id"] = pd.to_numeric(df["user_id"]).astype("int64")
    df["item_id"] = df["item_id"].astype(str)
    bundle = cand.build_bundle(clean, cut=SC_CUT, vec_cache=SC_VEC)
    Xb = cand.features_for_pairs(bundle, df["user_id"].to_numpy(), df["item_id"].to_numpy())
    Xv = v2_features(clean, SC_CUT, df)
    Xs = pd.concat([Xb.reset_index(drop=True), Xv.reset_index(drop=True)], axis=1)
    assert list(Xs.columns) == FEATS and Xs.isna().sum().sum() == 0

    N = len(df)
    fused = (W["lgb"] * rankdata(clf_lgb.predict_proba(Xs)[:, 1]) / N
             + W["xgb"] * rankdata(clf_xgb.predict_proba(Xs)[:, 1]) / N
             + W["lr"] * rankdata(lr.predict_proba(Xs)[:, 1]) / N)

    name = "cand-align-v10-h2"
    out = pd.DataFrame({"user_id": df["user_id"], "item_id": df["item_id"], "score": fused})
    assert len(out) == N and out.isna().sum().sum() == 0
    p = PROJECT_ROOT / "outputs" / "candidate" / f"sample_submission_{name}.csv"
    out.to_csv(p, index=False, encoding="utf-8")
    import joblib
    joblib.dump(clf_lgb, str(PROJECT_ROOT / "models" / f"candidate_{name}_lgb.pkl"))
    joblib.dump(clf_xgb, str(PROJECT_ROOT / "models" / f"candidate_{name}_xgb.pkl"))
    joblib.dump(lr, str(PROJECT_ROOT / "models" / f"candidate_{name}_lr.pkl"))
    meta = {"name": name,
            "desc": "视界 H≥2 赌注：三分量重训在 m∪(m+1) 月买标签，权重/特征/cut 与 v6 相同",
            "weights": W, "train_months_h2": [m[:7] for m in H2_MONTHS],
            "score_cut": str(SC_CUT.date()), "feature_order": FEATS}
    (PROJECT_ROOT / "models" / f"candidate_meta_{name}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[4/4] {name} → {p.name}；总耗时 {time.time()-t_all:.0f}s", flush=True)
    print(f"[done] H2 serve 完成", flush=True)


if __name__ == "__main__":
    main()
