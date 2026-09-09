"""回放前伸 OOF：训练池从 Jun 起扩到 Feb 起，验证折叠仍 Jul..Oct（可比 v6 .8721）。

scripts/56 已补 wide_v2_2010-02..05。本脚本用与 v6 完全相同的分量配方（lgb_macro_eq /
xgb_macro_eq / lr_eq_l2_sp，超参不变），唯一变化 = 每折树训练历史多吃了 Feb..May：
    * 旧：目标 Jul 树训 Jun 单月、…、目标 Oct 树训 Jun..Sep（serve 11 个月 vs 最深折 10 月）
    * 新：目标 Jul 树训 Feb..Jun、…、目标 Oct 树训 Feb..Sep（对应 serve 模型吃 Feb..Oct 全量 9 月）
LR 每折仍取最近 2 个历史月（serve 时仍 Sep+Oct，不受前伸影响）。
验证行 = wide_v2 Jul..Oct（逐位不变）⇒ gAUC_m 直接对照 v6 冠军 .8721。
超 .8721（且 >.0003 有实际余量）→ 产 v8（serve 模型同吃 Feb..Oct）。

用法：venv\\Scripts\\python.exe scripts\\57_oof_replay_ext.py
产物：outputs/candidate/replay/ext_report.txt + oof_ext_preds.csv
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools
import json
import sys
import time
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
# 全池 Feb..Oct；验证目标仍是 Jul..Oct（MONTHS[5:]）
MONTHS = [f"2010-{m:02d}-01" for m in range(2, 11)]
TGT = MONTHS[5:]


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
    n_g = tr.groupby(["user_id", "month"], sort=True).size()
    keys = pd.MultiIndex.from_arrays([tr["user_id"], tr["month"].astype(str)])
    w = 1.0 / n_g.reindex(keys).to_numpy(dtype="float64")
    if not with_sp:
        return w.astype("float64")
    spw = float((tr["label"] == 0).sum() / max(1, int(tr["label"].sum())))
    return (w * np.where(tr["label"] == 1, spw, 1.0)).astype("float64")


def gen(wide, kind: str):
    parts = []
    for ti, tgt in enumerate(wide):
        if tgt not in TGT:
            continue
        tr_m = [m for m in MONTHS[:ti]]          # 从 Feb 起的全部历史月
        if kind == "lr_eq_l2_sp" and len(tr_m) > 2:
            tr_m = tr_m[-2:]
        tr = pd.concat([wide[m] for m in tr_m], ignore_index=True)
        te = wide[tgt]
        sw = eq_group_sw(tr, with_sp=(kind == "lr_eq_l2_sp"))
        t0 = time.time()
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
        print(f"[{kind}] 折 {tgt[:7]} 训练 {len(tr):,} 行（历史 {tr_m[0][:7]}.."
              f"{tr_m[-1][:7]}）({time.time()-t0:.0f}s)", flush=True)
        part = te[["user_id", "item_id", "label", "month"]].copy()
        part["score"] = s
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    t_all = time.time()
    wide = load_wide()
    parts = [wide[m][["user_id", "item_id", "label", "month"]] for m in TGT]
    base = pd.concat(parts, ignore_index=True)
    print(f"[init] 验证 Jul..Oct 共 {len(base):,} 行 / 正例 {int(base['label'].sum()):,}"
          f"；全池 {len(MONTHS)} 月", flush=True)

    cols = {}
    for k in ["lgb_macro_eq", "xgb_macro_eq", "lr_eq_l2_sp"]:
        pf = gen(wide, k)
        assert (pf["user_id"].astype(str) + pf["item_id"].astype(str)
                == base["user_id"].astype(str) + base["item_id"].astype(str)).all()
        cols[k] = pf["score"].to_numpy()
        base[k] = pf["score"].to_numpy()
    base.to_csv(REPLAY_DIR / "oof_ext_preds.csv", index=False, encoding="utf-8")

    label = base["label"].to_numpy()
    lines = ["# 回放前伸 OOF（Feb..Oct 池；验证 Jul..Oct；对照 v6 Jun 起 = "
             ".8535/.8665/.8687/融合.8721）", ""]
    rmat = np.column_stack([rankdata(cols[k]) / len(base) for k in cols])
    for j, k in enumerate(cols):
        wg, mg = gauc_vec(base, rmat[:, j])
        lines.append(f"单 {k:14s}: AUC={roc_auc_score(label, rmat[:, j]):.4f} "
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
        wg, mg = gauc_vec(base, s)
        rows.append((tuple(round(float(x), 3) for x in w),
                     roc_auc_score(label, s), wg, mg))
    res = pd.DataFrame(rows, columns=["w", "auc", "gAUC_w", "gAUC_m"])
    res = res.sort_values("gAUC_m", ascending=False).reset_index(drop=True)
    lines.append("== 三分量融合 top6（v6 冠军 (0.1,0.3,0.6)=.8721；超 .8721 才产 v8）==")
    for _, r in res.head(6).iterrows():
        lines.append(f"w={r['w']}  AUC={r['auc']:.4f} gAUC_w={r['gAUC_w']:.4f} "
                     f"gAUC_m={r['gAUC_m']:.4f}")
    best = res.iloc[0]
    json.dump({"months": MONTHS, "components": list(cols), "weights": best["w"],
               "auc": float(best["auc"]), "gAUC_w": float(best["gAUC_w"]),
               "gAUC_m": float(best["gAUC_m"])},
              open(REPLAY_DIR / "ext_weights.json", "w"), indent=2)
    lines.append(f"== best gAUC_m={best['gAUC_m']:.4f} w={best['w']} ==")
    txt = "\n".join(lines) + "\n"
    (REPLAY_DIR / "ext_report.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"[done] 总耗时 {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
