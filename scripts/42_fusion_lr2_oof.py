"""v4 融合选权：在 v2 组件上加入「LR 只用最近 2 个月训练」(mgAUC .8684) 重做网格。

组件：lgb_macro(all) / lgb_macro_sp(all) / lr_last2（wide_v2）。
对目标月 t：lgb 用全部 <t 月训练；lr 用最近 2 个 <t 月训练（与 scripts/41 一致）。
评估同 35：宏 gAUC（主选权）+ AUC/逐用户 top-k 对照。写回 best weights 供最终构建。

用法：venv\\Scripts\\python.exe scripts\\42_fusion_lr2_oof.py
产物：outputs/candidate/replay/oof_v2b_preds.csv + fusion_v2b_report.txt
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
from src.candidate import rank_metrics

REPLAY_DIR = PROJECT_ROOT / "outputs" / "candidate" / "replay"
FEATS = cand.FEATURES_CAND + V2_FEATS
MONTHS = [f"2010-{m:02d}-01" for m in range(6, 11)]
KS = (5, 10, 20)


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


def gen_preds(wide):
    mo = sorted(wide)
    preds = {}
    for weight in ("macro", "macro_sp"):
        from lightgbm import LGBMClassifier
        parts = []
        for ti, tgt in enumerate(mo):
            tr_m = mo[:ti]
            if not tr_m:
                continue
            tr = pd.concat([wide[m] for m in tr_m], ignore_index=True)
            te = wide[tgt]
            nrow_u = tr.groupby("user_id").size()
            macro = 1.0 / np.sqrt(nrow_u.reindex(tr["user_id"]).to_numpy(dtype="float64"))
            if weight == "macro":
                sw = macro.astype("float64")
                spw_p = 1.0
            else:
                spw_p = float((tr["label"] == 0).sum() / max(1, int(tr["label"].sum())))
                sw = (macro * np.where(tr["label"] == 1, spw_p, 1.0)).astype("float64")
            clf = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                                 min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=1.0, random_state=SEED, verbosity=-1,
                                 deterministic=True, n_jobs=1)
            clf.fit(tr[FEATS], tr["label"], sample_weight=sw)
            part = te[["user_id", "item_id", "label", "month"]].copy()
            part["score"] = clf.predict_proba(te[FEATS])[:, 1].astype("float64")
            parts.append(part)
        preds[f"lgb_{weight}"] = pd.concat(parts, ignore_index=True)

    # lr_last2
    parts = []
    for ti, tgt in enumerate(mo):
        tr_m = mo[:ti][-2:] if ti >= 2 else mo[:ti]
        if not tr_m:
            continue
        tr = pd.concat([wide[m] for m in tr_m], ignore_index=True)
        te = wide[tgt]
        nrow_u = tr.groupby("user_id").size()
        macro = 1.0 / np.sqrt(nrow_u.reindex(tr["user_id"]).to_numpy(dtype="float64"))
        spw_p = float((tr["label"] == 0).sum() / max(1, int(tr["label"].sum())))
        sw = (macro * np.where(tr["label"] == 1, spw_p, 1.0)).astype("float64")
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
    base.to_csv(REPLAY_DIR / "oof_v2b_preds.csv", index=False, encoding="utf-8")

    df = base[["user_id", "item_id", "label", "month"]].copy()
    comps = ["lgb_macro", "lgb_macro_sp", "lr"]
    rmat = np.column_stack([rankdata(base[c]) / len(base) for c in comps])
    label = df["label"].to_numpy()
    lines = ["# v4 融合选权（wide_v2；组件=lgb_macro/lgb_macro_sp/lr_last2）", ""]
    for j, c in enumerate(comps):
        wg, mg = gauc_vec(df, rmat[:, j])
        lines.append(f"单 {c}: AUC={roc_auc_score(label, rmat[:, j]):.4f} "
                     f"gAUC_w={wg:.4f} gAUC_m={mg:.4f}")
    lines.append("")

    rows = []
    step = 0.05
    n = int(1 / step) + 1
    for a, b in itertools.product(range(n), repeat=2):
        c = n - 1 - a - b
        if c < 0:
            continue
        w = np.array([a, b, c], dtype="float64") * step
        s = rmat @ w
        wg, mg = gauc_vec(df, s)
        rows.append((tuple(round(float(x), 2) for x in w),
                     roc_auc_score(label, s), wg, mg))
    res = pd.DataFrame(rows, columns=["w", "auc", "gAUC_w", "gAUC_m"])
    res = res.sort_values("gAUC_m", ascending=False).reset_index(drop=True)
    lines.append("== 按 gAUC_m top6 ==")
    for _, r in res.head(6).iterrows():
        lines.append(f"w={r['w']}  AUC={r['auc']:.4f} gAUC_w={r['gAUC_w']:.4f} "
                     f"gAUC_m={r['gAUC_m']:.4f}")
    # 对照 top-k
    lines.append("")
    best = res.iloc[0]
    best_w = np.array(best["w"])
    lines.append(f"== top1 组合 w={best['w']} 逐用户 top-k ==")
    df["s"] = rmat @ best_w
    r = rank_metrics(df.rename(columns={"item_id": "stock_code"}), "s", ks=KS)
    lines.append("  ".join(f"{k}={r[k]:.4f}" for k in
                           ("rec@5", "rec@10", "hit@5", "hit@10", "prec@10")))

    json.dump({"components": comps, "weights": {c: float(w) for c, w in zip(comps, best_w)}},
              open(REPLAY_DIR / "fusion_v2b_weights.json", "w"), indent=2)
    txt = "\n".join(lines) + "\n"
    (REPLAY_DIR / "fusion_v2b_report.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print("[save] oof_v2b_preds.csv / fusion_v2b_report.txt / fusion_v2b_weights.json")


if __name__ == "__main__":
    main()
