"""树侧/融合细选 OOF：在已存 oof_v2c_preds 基础上补 XGB 变体并细网格。

已存（scripts/44）：lgb_macro / lgb_macro_sp / xgb_macro(all) / lr(last2)。
本轮补训：xgb_macro_sp(all)、xgb_macro(last2)、xgb_macro_sp(last2)——只重训新列；
细网格（step .025）在 (lgb_macro, xgb_macro, lr) 上精调三分量权重（对照 v5 .15/.15/.7 = .8710）；
粗网格测含 xgb_macro_sp 的 4 分量有无增量。目标：找出比 .8710 更高的融合权重组合。

用法：venv\\Scripts\\python.exe scripts\\46_xgb_variants_oof.py
产物：outputs/candidate/replay/oof_v2d_preds.csv + fusion_v2d_report.txt + weights json
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


def macro_sw(tr, with_sp):
    nrow_u = tr.groupby("user_id").size()
    macro = 1.0 / np.sqrt(nrow_u.reindex(tr["user_id"]).to_numpy(dtype="float64"))
    if not with_sp:
        return macro.astype("float64")
    spw = float((tr["label"] == 0).sum() / max(1, int(tr["label"].sum())))
    return (macro * np.where(tr["label"] == 1, spw, 1.0)).astype("float64")


def gen_xgb(wide, weight: str, window: int | None):
    from xgboost import XGBClassifier
    mo = sorted(wide)
    parts = []
    for ti, tgt in enumerate(mo):
        if ti == 0:
            continue
        tr_m = mo[:ti][-window:] if window else mo[:ti]
        tr = pd.concat([wide[m] for m in tr_m], ignore_index=True)
        te = wide[tgt]
        xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8,
                            tree_method="hist", n_jobs=1, random_state=SEED,
                            eval_metric="auc", verbosity=0)
        xgb.fit(tr[FEATS], tr["label"],
                sample_weight=macro_sw(tr, with_sp=(weight == "macro_sp")))
        part = te[["user_id", "item_id", "label", "month"]].copy()
        part["score"] = xgb.predict_proba(te[FEATS])[:, 1].astype("float64")
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def grid_ok(df, label, comps, step):
    """通用单纯形细网格；step 任意（1/step 须为整数）。"""
    rmat = np.column_stack([rankdata(df[c]) / len(df) for c in comps])
    m = len(comps)
    n = int(round(1.0 / step)) + 1
    rows = []
    for idx in itertools.product(range(n), repeat=m - 1):
        if sum(idx) > n - 1:
            continue
        w = np.array(list(idx) + [n - 1 - sum(idx)], dtype="float64") * step
        s = rmat @ w
        wg, mg = gauc_vec(df, s)
        rows.append((tuple(round(float(x), 3) for x in w),
                     roc_auc_score(label, s), wg, mg))
    res = pd.DataFrame(rows, columns=["w", "auc", "gAUC_w", "gAUC_m"])
    return res.sort_values("gAUC_m", ascending=False).reset_index(drop=True)


def main() -> None:
    wide = load_wide()
    # 重载已存分量，避免重训
    base = pd.read_csv(REPLAY_DIR / "oof_v2c_preds.csv", dtype={"item_id": str})
    assert {"lgb_macro", "lgb_macro_sp", "xgb_macro", "lr"} <= set(base.columns)

    # 补训 XGB 变体
    newcols = {
        "xgb_macro_sp": gen_xgb(wide, "macro_sp", None),
        "xgb_macro_l2": gen_xgb(wide, "macro", 2),
        "xgb_macrosp_l2": gen_xgb(wide, "macro_sp", 2),
    }
    for nm, pf in newcols.items():
        assert (pf["user_id"].astype(str) + pf["item_id"].astype(str)
                == base["user_id"].astype(str) + base["item_id"].astype(str)).all()
        base[nm] = pf["score"].to_numpy()
    base.to_csv(REPLAY_DIR / "oof_v2d_preds.csv", index=False, encoding="utf-8")

    df = base[["user_id", "item_id", "label", "month"]].copy()
    label = df["label"].to_numpy()
    allcomp = ["lgb_macro", "lgb_macro_sp", "xgb_macro", "xgb_macro_sp",
               "xgb_macro_l2", "xgb_macrosp_l2", "lr"]
    lines = ["# v6 候选 OOF（wide_v2；补 XGB 变体 + 细网格）", ""]
    for c in allcomp:
        r = rankdata(base[c]) / len(base)
        wg, mg = gauc_vec(df, r)
        lines.append(f"单 {c:14s}: AUC={roc_auc_score(label, r):.4f} "
                     f"gAUC_w={wg:.4f} gAUC_m={mg:.4f}")
    lines.append("")

    res_fine = grid_ok(base, label, ["lgb_macro", "xgb_macro", "lr"], step=0.025)
    lines.append("== 细网格 (lgb_macro,xgb_macro,lr) step.025 top6（v5=.15/.15/.70→.8710）==")
    for _, r in res_fine.head(6).iterrows():
        lines.append(f"w={r['w']}  AUC={r['auc']:.4f} gAUC_w={r['gAUC_w']:.4f} "
                     f"gAUC_m={r['gAUC_m']:.4f}")
    lines.append("")

    # 含 lgb_macro_sp / xgb_macro_sp 的四分量粗扫（对照三分量 best 是否还有增量）
    best_fine = res_fine.iloc[0]
    comps_f = ["lgb_macro", "xgb_macro", "lr"]
    lines.append(f"== 三分量细网格 best w={best_fine['w']} gAUC_m={best_fine['gAUC_m']:.4f} ==")
    cand_4 = []
    for addcol, name in [("lgb_macro_sp", "三分量+lgb_macro_sp"),
                         ("xgb_macro_sp", "三分量+xgb_macro_sp")]:
        comps4 = comps_f + [addcol]
        r4 = grid_ok(base, label, comps4, step=0.1)
        cand_4.append(r4)
        lines.append(f"== {name}（step.1）top3 ==")
        for _, r in r4.head(3).iterrows():
            lines.append(f"w={r['w']}  AUC={r['auc']:.4f} gAUC_w={r['gAUC_w']:.4f} "
                         f"gAUC_m={r['gAUC_m']:.4f}")
        lines.append("")

    # 选权：fine 三分量 vs 两个 4 分量 top1，取 gAUC_m 最高
    cands = [(best_fine, comps_f)] + [(r4.iloc[0], comps_f + [addc])
                                      for (r4, addc) in
                                      zip(cand_4, ["lgb_macro_sp", "xgb_macro_sp"])]
    res_best, comps = max(cands, key=lambda t: t[0]["gAUC_m"])
    wdict = {c: float(w) for c, w in zip(comps, res_best["w"])}
    json.dump({"components": comps, "weights": wdict},
              open(REPLAY_DIR / "fusion_v2d_weights.json", "w"), indent=2)
    lines.append(f"== 全场 best gAUC_m={res_best['gAUC_m']:.4f} w={wdict}"
                 f"（> .8710 才产 v6 提交）==")
    txt = "\n".join(lines) + "\n"
    (REPLAY_DIR / "fusion_v2d_report.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print("[save] oof_v2d_preds.csv / fusion_v2d_report.txt / fusion_v2d_weights.json")


if __name__ == "__main__":
    main()
