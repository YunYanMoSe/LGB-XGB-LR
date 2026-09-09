"""CatBoost 去相关分量 OOF：wide_v2（44 特征）+ 组等权，测能否给融合再加增量。

组长 PDF 融合含 CatBoost(权重 .159) 作第三种树族；我们用 XGB 替代但未测 CatBoost 本身。
本轮：catboost_eq 单模型 mgAUC + 四分量融合 (lgb_eq, xgb_eq, cb_eq, lr_eq) 细网格，
对照 v6 三分量 .8721。超 .8721 → 产 v7。

用法：venv\\Scripts\\python.exe scripts\\52_catboost_oof.py
产物：outputs/candidate/replay/fusion_cb_report.txt + weights json
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


def eq_group_sw(tr):
    n_g = tr.groupby(["user_id", "month"], sort=True).size()
    keys = pd.MultiIndex.from_arrays([tr["user_id"], tr["month"].astype(str)])
    return (1.0 / n_g.reindex(keys).to_numpy(dtype="float64")).astype("float64")


def gen_catboost(wide, log=print):
    from catboost import CatBoostClassifier
    mo = sorted(wide)
    parts = []
    t_start = time.time()
    for ti, tgt in enumerate(mo):
        if ti == 0:
            continue
        tr_m = mo[:ti]
        tr = pd.concat([wide[m] for m in tr_m], ignore_index=True)
        te = wide[tgt]
        t0 = time.time()
        clf = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6,
                                 loss_function="Logloss", random_seed=SEED,
                                 thread_count=1, verbose=False, allow_writing_files=False)
        clf.fit(tr[FEATS], tr["label"], sample_weight=eq_group_sw(tr))
        t1 = time.time()
        log(f"[cb] 折 目标月={tgt[:7]} 训练 {len(tr):,} 行：fit {t1-t0:.0f}s", flush=True)
        part = te[["user_id", "item_id", "label", "month"]].copy()
        part["score"] = clf.predict_proba(te[FEATS])[:, 1].astype("float64")
        parts.append(part)
    log(f"[cb] 4 折 CatBoost 预测完成，累计 {time.time()-t_start:.0f}s", flush=True)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    t_all = time.time()
    wide = load_wide()
    base = pd.read_csv(REPLAY_DIR / "oof_eqg_preds.csv", dtype={"item_id": str})
    cb = gen_catboost(wide)
    assert (cb["user_id"].astype(str) + cb["item_id"].astype(str)
            == base["user_id"].astype(str) + base["item_id"].astype(str)).all()
    base["cb_macro_eq"] = cb["score"].to_numpy()
    base.to_csv(REPLAY_DIR / "oof_cb_preds.csv", index=False, encoding="utf-8")
    print("[cb] OOF 预测已存盘 oof_cb_preds.csv，开始融合网格", flush=True)

    df = base[["user_id", "item_id", "label", "month"]].copy()
    label = df["label"].to_numpy()
    comps = ["lgb_macro_eq", "xgb_macro_eq", "cb_macro_eq", "lr_eq_l2_sp"]
    lines = ["# CatBoost OOF（wide_v2 44 特征；对照 v6=.8535/.8665/.8687/融合.8721）", ""]
    for c in comps:
        r = rankdata(base[c]) / len(base)
        wg, mg = gauc_vec(df, r)
        lines.append(f"单 {c:14s}: AUC={roc_auc_score(label, r):.4f} "
                     f"gAUC_w={wg:.4f} gAUC_m={mg:.4f}")
    lines.append("")

    def grid(cs, tag, step=0.05):
        rmat = np.column_stack([rankdata(base[c]) / len(base) for c in cs])
        m = len(cs)
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
        lines.append(f"== {tag} top5 ==")
        for _, r in res.head(5).iterrows():
            lines.append(f"w={r['w']}  AUC={r['auc']:.4f} gAUC_w={r['gAUC_w']:.4f} "
                         f"gAUC_m={r['gAUC_m']:.4f}")
        lines.append("")
        return res

    res3 = grid(["lgb_macro_eq", "xgb_macro_eq", "lr_eq_l2_sp"],
                "三分量基准(对照 v6=.8721)")
    print(f"[grid] 三分量基准完成（累计 {time.time()-t_all:.0f}s）", flush=True)
    res4 = grid(comps, "四分量加CatBoost", step=0.05)
    print(f"[grid] 四分量粗网格完成（累计 {time.time()-t_all:.0f}s）", flush=True)

    cands = [("v6三分量", res3.iloc[0]), ("四分量", res4.iloc[0])]
    name, best = max(cands, key=lambda t: t[1]["gAUC_m"])
    comps_use = ["lgb_macro_eq", "xgb_macro_eq", "lr_eq_l2_sp"] if name == "v6三分量" else comps
    json.dump({"components": comps_use,
               "weights": {c: float(w) for c, w in zip(comps_use, best["w"])},
               "auc": float(best["auc"]), "gAUC_w": float(best["gAUC_w"]),
               "gAUC_m": float(best["gAUC_m"])},
              open(REPLAY_DIR / "fusion_cb_weights.json", "w"), indent=2)
    lines.append(f"== best {name}: gAUC_m={best['gAUC_m']:.4f} w={best['w']}"
                 f"（> .8721 才产 v7）==")
    txt = "\n".join(lines) + "\n"
    (REPLAY_DIR / "fusion_cb_report.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"[done] 总耗时 {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
