"""v4 特征 OOF：在 wide_v4（48 特征 = 44 + 4 v4）上按组等权重训三分量并融合选权。

对照 v6（wide_v2 44 特征）单模型：lgb_macro_eq .8535 / xgb_macro_eq .8665 /
lr_eq_l2_sp .8687；融合冠军 (lgb_eq .1, xgb_eq .3, lr_eq .6) gAUC_m .8721。
v4 特征若给单模型/融合天花板带来 +，才产 v7 提交。

用法：venv\\Scripts\\python.exe scripts\\51_v4_features_oof.py
产物：outputs/candidate/replay/fusion_v4feat_report.txt + weights json
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools
import json
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SEED
from src import candidate as cand
from src.candidate_v2 import V2_FEATS
from src.candidate_v4 import V4_FEATS

REPLAY_DIR = PROJECT_ROOT / "outputs" / "candidate" / "replay"
FEATS = list(cand.FEATURES_CAND) + list(V2_FEATS) + list(V4_FEATS)
MONTHS = [f"2010-{m:02d}-01" for m in range(6, 11)]


def load_wide():
    return {m: pd.read_csv(REPLAY_DIR / f"wide_v4_{m[:7]}.csv", dtype={"item_id": str})
            for m in MONTHS}


def gauc_vec(df: pd.DataFrame, s: np.ndarray):
    sub = df[["user_id", "month"]].copy()
    sub["s"] = np.asarray(s, dtype="float64")
    sub["lbl"] = df["label"].to_numpy(dtype="int64")
    grp, _ = pd.factorize(sub["user_id"].astype(str) + "_" + sub["month"].astype(str))
    sub["g"] = grp
    sub["r"] = sub.groupby("g")["s"].rank(method="average").to_numpy()
    pos = sub[sub["lbl"] == 1]
    agg = sub.groupby("g").agg(n=("lbl", "size"), p=("lbl", "sum"))
    sp = pos.groupby("g")["r"].sum()
    agg = agg.join(sp.rename("sp"), how="left").fillna({"sp": 0.0})
    agg["nn"] = agg["n"] - agg["p"]
    ok = agg[(agg["p"] > 0) & (agg["nn"] > 0)].copy()
    ok["auc"] = (ok["sp"] - ok["p"] * (ok["p"] + 1.0) / 2.0) / (ok["p"] * ok["nn"])
    w = ok["n"]
    return float((w * ok["auc"]).sum() / w.sum()), float(ok["auc"].mean())


def eq_group_sw(tr, with_sp):
    n_g = tr.groupby(["user_id", "month"], sort=True).size()
    keys = pd.MultiIndex.from_arrays([tr["user_id"], tr["month"].astype(str)])
    w = 1.0 / n_g.reindex(keys).to_numpy(dtype="float64")
    if not with_sp:
        return w.astype("float64")
    spw = float((tr["label"] == 0).sum() / max(1, int(tr["label"].sum())))
    return (w * np.where(tr["label"] == 1, spw, 1.0)).astype("float64")


def gen_eq(wide, kind: str):
    mo = sorted(wide)
    parts = []
    for ti, tgt in enumerate(mo):
        if ti == 0:
            continue
        if kind in ("lgb_macro_eq", "xgb_macro_eq"):
            tr_m = mo[:ti]
            w_sp = False
        else:
            tr_m = mo[:ti][-2:] if ti >= 2 else mo[:ti]
            w_sp = True
        tr = pd.concat([wide[m] for m in tr_m], ignore_index=True)
        te = wide[tgt]
        sw = eq_group_sw(tr, with_sp=w_sp)
        if kind == "lgb_macro_eq":
            from lightgbm import LGBMClassifier
            clf = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                                 min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=1.0, random_state=SEED, verbosity=-1,
                                 deterministic=True, n_jobs=1)
            clf.fit(tr[FEATS], tr["label"], sample_weight=sw)
            s = clf.predict_proba(te[FEATS])[:, 1].astype("float64")
        elif kind == "xgb_macro_eq":
            from xgboost import XGBClassifier
            clf = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6,
                                subsample=0.8, colsample_bytree=0.8,
                                tree_method="hist", n_jobs=1, random_state=SEED,
                                eval_metric="auc", verbosity=0)
            clf.fit(tr[FEATS], tr["label"], sample_weight=sw)
            s = clf.predict_proba(te[FEATS])[:, 1].astype("float64")
        else:
            lr = make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=2000, C=1.0, random_state=SEED))
            lr.fit(tr[FEATS], tr["label"], logisticregression__sample_weight=sw)
            s = lr.predict_proba(te[FEATS])[:, 1].astype("float64")
        part = te[["user_id", "item_id", "label", "month"]].copy()
        part["score"] = s
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    wide = load_wide()
    mo = sorted(wide)
    cols = {}
    for k in ["lgb_macro_eq", "xgb_macro_eq", "lr_eq_l2_sp"]:
        pf = gen_eq(wide, k)
        cols[k] = pf["score"].to_numpy()
    # 与 gen_eq 内部 concat 同序重建键行（目标月 = mo[1:] Jul..Oct）
    parts = [wide[tgt][["user_id", "item_id", "label", "month"]] for tgt in mo[1:]]
    base_df = pd.concat(parts, ignore_index=True)
    for k, s in cols.items():
        base_df[k] = s
    base_df.to_csv(REPLAY_DIR / "oof_v4feat_preds.csv", index=False, encoding="utf-8")

    dfm = base_df[["user_id", "item_id", "label", "month"]].copy()
    label = dfm["label"].to_numpy()
    comps = ["lgb_macro_eq", "xgb_macro_eq", "lr_eq_l2_sp"]
    lines = ["# v4 特征 OOF（wide_v4 48 特征；对照 v6 44 特征 = .8535/.8665/.8687/.8721）", ""]
    rmat = np.column_stack([rankdata(base_df[c]) / len(base_df) for c in comps])
    for j, c in enumerate(comps):
        wg, mg = gauc_vec(dfm, rmat[:, j])
        lines.append(f"单 {c:14s}: AUC={roc_auc_score(label, rmat[:, j]):.4f} "
                     f"gAUC_w={wg:.4f} gAUC_m={mg:.4f}")
    lines.append("")

    step = 0.025
    n = int(round(1.0 / step)) + 1
    rows = []
    for a, b in itertools.product(range(n), repeat=2):
        if a + b > n - 1:
            continue
        w = np.array([a, b, n - 1 - a - b], dtype="float64") * step
        s = rmat @ w
        wg, mg = gauc_vec(dfm, s)
        rows.append((tuple(round(float(x), 3) for x in w),
                     roc_auc_score(label, s), wg, mg))
    res = pd.DataFrame(rows, columns=["w", "auc", "gAUC_w", "gAUC_m"])
    res = res.sort_values("gAUC_m", ascending=False).reset_index(drop=True)
    lines.append("== 三分量融合 top6（v6 冠军 .15/.275/.575≈.8721；超 .8721 → 产 v7）==")
    for _, r in res.head(6).iterrows():
        lines.append(f"w={r['w']}  AUC={r['auc']:.4f} gAUC_w={r['gAUC_w']:.4f} "
                     f"gAUC_m={r['gAUC_m']:.4f}")
    best = res.iloc[0]
    wdict = {c: float(w) for c, w in zip(comps, best["w"])}
    json.dump({"features": FEATS, "n_features": len(FEATS),
               "components": comps, "weights": wdict,
               "auc": float(best["auc"]), "gAUC_w": float(best["gAUC_w"]),
               "gAUC_m": float(best["gAUC_m"])},
              open(REPLAY_DIR / "fusion_v4feat_weights.json", "w"), indent=2)
    lines.append(f"== best w={wdict} gAUC_m={best['gAUC_m']:.4f} ==")
    txt = "\n".join(lines) + "\n"
    (REPLAY_DIR / "fusion_v4feat_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
