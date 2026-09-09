"""v5 融合选权：加入 XGBoost(macro) 作去相关树分量，对照 lgb_macro 与 lr_last2。

组件（wide_v2，时间序 OOF，目标月只吃更早月）：
    * lgb_macro / lgb_macro_sp（全前月，macro 加权）
    * xgb_macro（全前月，macro 加权）
    * lr_last2（最近 2 月，macro_sp 加权）
网格步 .05 在 (lgb_macro, xgb_macro, lr) 上按 mgAUC 选权；报告单模型与最佳融合。
若融合 mgAUC 显著 > 0.8703（v4）才值得产 v5 提交。

用法：venv\\Scripts\\python.exe scripts\\44_fusion_xgb_oof.py
产物：outputs/candidate/replay/oof_v2c_preds.csv + fusion_v2c_report.txt + weights json
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

REPLAY_DIR = PROJECT_ROOT / "outputs" / "candidate" / "replay"
FEATS = cand.FEATURES_CAND + V2_FEATS
MONTHS = [f"2010-{m:02d}-01" for m in range(6, 11)]


def load_wide():
    return {m: pd.read_csv(REPLAY_DIR / f"wide_v2_{m[:7]}.csv", dtype={"item_id": str})
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


def macro_sw(tr):
    nrow_u = tr.groupby("user_id").size()
    return (1.0 / np.sqrt(nrow_u.reindex(tr["user_id"]).to_numpy(dtype="float64"))).astype("float64")


def gen_preds(wide):
    mo = sorted(wide)
    preds = {}
    # LGB macro / macro_sp
    for weight in ("macro", "macro_sp"):
        from lightgbm import LGBMClassifier
        parts = []
        for ti, tgt in enumerate(mo):
            tr_m = mo[:ti]
            if not tr_m:
                continue
            tr = pd.concat([wide[m] for m in tr_m], ignore_index=True)
            te = wide[tgt]
            if weight == "macro":
                sw = macro_sw(tr)
            else:
                spw = float((tr["label"] == 0).sum() / max(1, int(tr["label"].sum())))
                sw = (macro_sw(tr) * np.where(tr["label"] == 1, spw, 1.0)).astype("float64")
            clf = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                                 min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=1.0, random_state=SEED, verbosity=-1,
                                 deterministic=True, n_jobs=1)
            clf.fit(tr[FEATS], tr["label"], sample_weight=sw)
            part = te[["user_id", "item_id", "label", "month"]].copy()
            part["score"] = clf.predict_proba(te[FEATS])[:, 1].astype("float64")
            parts.append(part)
        preds[f"lgb_{weight}"] = pd.concat(parts, ignore_index=True)
    # XGB macro
    from xgboost import XGBClassifier
    parts = []
    for ti, tgt in enumerate(mo):
        tr_m = mo[:ti]
        if not tr_m:
            continue
        tr = pd.concat([wide[m] for m in tr_m], ignore_index=True)
        te = wide[tgt]
        xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8,
                            tree_method="hist", n_jobs=1, random_state=SEED,
                            eval_metric="auc", verbosity=0)
        xgb.fit(tr[FEATS], tr["label"], sample_weight=macro_sw(tr))
        part = te[["user_id", "item_id", "label", "month"]].copy()
        part["score"] = xgb.predict_proba(te[FEATS])[:, 1].astype("float64")
        parts.append(part)
    preds["xgb_macro"] = pd.concat(parts, ignore_index=True)
    # LR last2
    parts = []
    for ti, tgt in enumerate(mo):
        tr_m = mo[:ti][-2:] if ti >= 2 else mo[:ti]
        if not tr_m:
            continue
        tr = pd.concat([wide[m] for m in tr_m], ignore_index=True)
        te = wide[tgt]
        spw = float((tr["label"] == 0).sum() / max(1, int(tr["label"].sum())))
        sw = (macro_sw(tr) * np.where(tr["label"] == 1, spw, 1.0)).astype("float64")
        lr = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, C=1.0, random_state=SEED))
        lr.fit(tr[FEATS], tr["label"], logisticregression__sample_weight=sw)
        part = te[["user_id", "item_id", "label", "month"]].copy()
        part["score"] = lr.predict_proba(te[FEATS])[:, 1].astype("float64")
        parts.append(part)
    preds["lr"] = pd.concat(parts, ignore_index=True)
    return preds


def main() -> None:
    wide = load_wide()
    preds = gen_preds(wide)
    base = preds["lgb_macro"][["user_id", "item_id", "label", "month"]].copy()
    for nm, pf in preds.items():
        base[nm] = pf["score"].to_numpy()
    base.to_csv(REPLAY_DIR / "oof_v2c_preds.csv", index=False, encoding="utf-8")

    df = base[["user_id", "item_id", "label", "month"]].copy()
    label = df["label"].to_numpy()
    lines = ["# v5 融合选权（wide_v2：lgb_macro / lgb_macro_sp / xgb_macro / lr_last2）", ""]
    for c in ["lgb_macro", "lgb_macro_sp", "xgb_macro", "lr"]:
        r = rankdata(base[c]) / len(base)
        wg, mg = gauc_vec(df, r)
        lines.append(f"单 {c:12s}: AUC={roc_auc_score(label, r):.4f} "
                     f"gAUC_w={wg:.4f} gAUC_m={mg:.4f}")
    lines.append("")

    def grid(comps, tag):
        """通用单纯形网格：前 len(comps)-1 个权重点 .05 整倍，末位=1-Σ前位≥0。"""
        sub = []
        rmat = np.column_stack([rankdata(base[c]) / len(base) for c in comps])
        m = len(comps)
        step = 0.05
        n = int(1 / step) + 1  # 每维点数（含 0 与 1）
        if m == 2:
            for a in range(n):
                b = n - 1 - a
                w = np.array([a, b], dtype="float64") * step
                s = rmat @ w
                wg, mg = gauc_vec(df, s)
                sub.append((tuple(round(float(x), 2) for x in w),
                            roc_auc_score(label, s), wg, mg))
        else:
            for a, b in itertools.product(range(n), repeat=m - 1):
                rest = a + b
                if rest > n - 1:
                    continue
                w = np.array([a, b, n - 1 - rest], dtype="float64") * step
                s = rmat @ w
                wg, mg = gauc_vec(df, s)
                sub.append((tuple(round(float(x), 2) for x in w),
                            roc_auc_score(label, s), wg, mg))
        res = pd.DataFrame(sub, columns=["w", "auc", "gAUC_w", "gAUC_m"])
        res = res.sort_values("gAUC_m", ascending=False).reset_index(drop=True)
        lines.append(f"== {tag}（{comps}）按 gAUC_m top5 ==")
        for _, r in res.head(5).iterrows():
            lines.append(f"w={r['w']}  AUC={r['auc']:.4f} gAUC_w={r['gAUC_w']:.4f} "
                         f"gAUC_m={r['gAUC_m']:.4f}")
        return res

    res1 = grid(["lgb_macro", "xgb_macro", "lr"], "三分量树×2")
    lines.append("")
    res2 = grid(["lgb_macro", "lr"], "两分量基准(对照 v4=.8703)")
    lines.append("")
    res = res1 if res1.iloc[0]["gAUC_m"] >= res2.iloc[0]["gAUC_m"] else res2
    best = res.iloc[0]
    best_w = best["w"]
    comps = ["lgb_macro", "xgb_macro", "lr"] if res is res1 else ["lgb_macro", "lr"]
    wdict = {c: float(w) for c, w in zip(comps, best_w)}
    json.dump({"components": comps, "weights": wdict},
              open(REPLAY_DIR / "fusion_v2c_weights.json", "w"), indent=2)
    lines.append(f"== 最佳 w={wdict}（若 gAUC_m > .8703 → 产 v5 提交）==")
    txt = "\n".join(lines) + "\n"
    (REPLAY_DIR / "fusion_v2c_report.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print("[save] oof_v2c_preds.csv / fusion_v2c_report.txt / fusion_v2c_weights.json")


if __name__ == "__main__":
    main()
