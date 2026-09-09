"""行权重对齐 gAUC_m 的 OOF：把权重从 1/√组大小 换成 1/组大小（每组对损失等贡献）。

选型口径 gAUC_m = 逐 (user,month) 组 AUC 的宏平均（组等权）；此前训练权重 macro=1/√n
（脚本 40/41 定为最优）。本轮测「组等权」w=1/n_group（group=(user,month)）是否更贴口径、
把单模型/融合天花板抬过 .8710。只重训新 eqn 列，其余从已存 oof_v2d_preds 读取。

变体：
    * lgb_macro_eq    : LGB 全月, 组等权
    * xgb_macro_eq    : XGB 全月, 组等权
    * lr_eq_l2_sp     : LR 最近2月, 组等权 × scale_pos(spw)
    * lr_eq_l2        : LR 最近2月, 组等权（对照是否靠 spw）
单模型对照（已存）：lgb_macro .8556 / xgb_macro .8586 / lr(last2,sp) .8684。
融合对照（v5 三分量 fine-grid 已证 .8710 为该权系最优）。

用法：venv\\Scripts\\python.exe scripts\\47_eqgroup_weight_oof.py
产物：outputs/candidate/replay/oof_eqg_preds.csv + eqgroup_report.txt
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools
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


def eq_group_sw(tr, with_sp):
    """每 (user,month) 组内行等权 ⇒ w=1/n_group；×spw 放大正例（线性用）。"""
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
        else:  # lr 近 2 月
            tr_m = mo[:ti][-2:] if ti >= 2 else mo[:ti]
            w_sp = kind == "lr_eq_l2_sp"
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
    base = pd.read_csv(REPLAY_DIR / "oof_v2d_preds.csv", dtype={"item_id": str})
    newcols = {k: gen_eq(wide, k) for k in
               ["lgb_macro_eq", "xgb_macro_eq", "lr_eq_l2_sp", "lr_eq_l2"]}
    for nm, pf in newcols.items():
        assert (pf["user_id"].astype(str) + pf["item_id"].astype(str)
                == base["user_id"].astype(str) + base["item_id"].astype(str)).all()
        base[nm] = pf["score"].to_numpy()
    base.to_csv(REPLAY_DIR / "oof_eqg_preds.csv", index=False, encoding="utf-8")

    df = base[["user_id", "item_id", "label", "month"]].copy()
    label = df["label"].to_numpy()
    comps_all = ["lgb_macro", "xgb_macro", "lr",
                 "lgb_macro_eq", "xgb_macro_eq", "lr_eq_l2", "lr_eq_l2_sp"]
    lines = ["# 组等权 OOF（wide_v2；1/√n → 1/n_group 对照）", ""]
    for c in comps_all:
        r = rankdata(base[c]) / len(base)
        wg, mg = gauc_vec(df, r)
        lines.append(f"单 {c:14s}: AUC={roc_auc_score(label, r):.4f} "
                     f"gAUC_w={wg:.4f} gAUC_m={mg:.4f}")
    lines.append("")

    def grid(comps, tag, step=0.05):
        rmat = np.column_stack([rankdata(base[c]) / len(base) for c in comps])
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
        res = res.sort_values("gAUC_m", ascending=False).reset_index(drop=True)
        lines.append(f"== {tag} top4 ==")
        for _, r in res.head(4).iterrows():
            lines.append(f"w={r['w']}  AUC={r['auc']:.4f} gAUC_w={r['gAUC_w']:.4f} "
                         f"gAUC_m={r['gAUC_m']:.4f}")
        lines.append("")
        return res

    grid(["lgb_macro_eq", "xgb_macro_eq", "lr_eq_l2_sp"], "三分量组等权（对照 v5=.8710）")
    grid(["lgb_macro", "xgb_macro", "lr_eq_l2_sp"], "树保持 1/√n、LR 组等权")
    grid(["lgb_macro_eq", "xgb_macro_eq", "lr"], "树组等权、LR 保持 macro_sp last2")
    txt = "\n".join(lines) + "\n"
    (REPLAY_DIR / "eqgroup_report.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print("[save] oof_eqg_preds.csv / eqgroup_report.txt")


if __name__ == "__main__":
    main()
