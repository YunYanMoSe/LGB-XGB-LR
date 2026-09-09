"""组等权下重调参 OOF：单模型超参在旧(1/√n)目标下冻结，eq 换目标后未跟着重调。

wide_v2（44 特征）同一 4 折叠 (Jul..Oct) 上逐类扫 XGB / LGB / LR 配置，单 gAUC_m
直接对照 v6 基线：lgb_macro_eq .8535 / xgb_macro_eq .8665 / lr_eq_l2_sp .8687 /
融合 (lgb_eq .1, xgb_eq .3, lr_eq_l2_sp .6) = .8721。
选各类最优配置 → 三分量 fine-grid 融合；gAUC_m 超 .8721（且稳）才值得产 v7。

行权重：w = 1/n_{user,month}（组等权），LR/可选树配置再 ×ratio**alpha 放大正例（alpha=0 ⇒ 纯组等权）。

用法：venv\\Scripts\\python.exe scripts\\53_eqgroup_tune.py
产物：outputs/candidate/replay/eqtune_report.txt + oof_eqtune_preds.csv（含各配置列）
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
MONTHS = [f"2010-{m:02d}-01" for m in range(6, 11)]

# ---- 组等权（alpha=0 ⇒ 纯 1/n_group；alpha>0 ⇒ 正例行再 ×ratio**alpha）----
def group_sw(tr: pd.DataFrame, alpha: float = 0.0) -> np.ndarray:
    n_g = tr.groupby(["user_id", "month"], sort=True).size()
    keys = pd.MultiIndex.from_arrays([tr["user_id"], tr["month"].astype(str)])
    w = 1.0 / n_g.reindex(keys).to_numpy(dtype="float64")
    if alpha > 0.0:
        ratio = float((tr["label"] == 0).sum() / max(1, int(tr["label"].sum())))
        w = w * np.where(tr["label"] == 1, ratio ** alpha, 1.0)
    return w.astype("float64")


def load_wide():
    return {m: pd.read_csv(REPLAY_DIR / f"wide_v2_{m[:7]}.csv", dtype={"item_id": str})
            for m in MONTHS}


def gauc_of(s: np.ndarray, df: pd.DataFrame):
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


# ---- 折内训练（全部目标月 mo[1:] 的按序预测拼成一列）----
def fold_tree(wide, kind: str, cfg: dict, log) -> np.ndarray:
    mo = sorted(wide)
    parts = []
    for ti, tgt in enumerate(mo):
        if ti == 0:
            continue
        tr = pd.concat([wide[m] for m in mo[:ti]], ignore_index=True)
        te = wide[tgt]
        sw = group_sw(tr, alpha=cfg.get("alpha", 0.0))
        t0 = time.time()
        if kind == "xgb":
            from xgboost import XGBClassifier
            clf = XGBClassifier(n_estimators=cfg["ne"], learning_rate=cfg["lr"],
                                max_depth=cfg["d"], subsample=cfg["ss"],
                                colsample_bytree=cfg["cs"], min_child_weight=cfg.get("mcw", 1),
                                tree_method="hist", n_jobs=1, random_state=SEED,
                                eval_metric="auc", verbosity=0)
            clf.fit(tr[FEATS], tr["label"], sample_weight=sw)
        else:
            from lightgbm import LGBMClassifier
            clf = LGBMClassifier(n_estimators=cfg["ne"], learning_rate=cfg["lr"],
                                 num_leaves=cfg["nl"], min_child_samples=cfg.get("mcs", 20),
                                 subsample=cfg["ss"], colsample_bytree=cfg["cs"],
                                 scale_pos_weight=1.0, random_state=SEED, verbosity=-1,
                                 deterministic=True, n_jobs=1)
            clf.fit(tr[FEATS], tr["label"], sample_weight=sw)
        s = clf.predict_proba(te[FEATS])[:, 1].astype("float64")
        log(f"  [{kind}] {cfg['name']} 折 {tgt[:7]} 训练 {len(tr):,} 行 "
            f"({time.time()-t0:.0f}s)", flush=True)
        parts.append(s)
    return np.concatenate(parts)


def fold_lr(wide, window: int, alpha: float, C: float, log) -> np.ndarray:
    mo = sorted(wide)
    parts = []
    for ti, tgt in enumerate(mo):
        if ti == 0:
            continue
        tr_m = mo[:ti]
        if window is not None and ti >= window:
            tr_m = mo[:ti][-window:]
        tr = pd.concat([wide[m] for m in tr_m], ignore_index=True)
        te = wide[tgt]
        sw = group_sw(tr, alpha=alpha)
        t0 = time.time()
        lr = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, C=C, random_state=SEED))
        lr.fit(tr[FEATS], tr["label"], logisticregression__sample_weight=sw)
        s = lr.predict_proba(te[FEATS])[:, 1].astype("float64")
        log(f"  [lr] w={window} a={alpha} C={C} 折 {tgt[:7]} 训练 {len(tr):,} 行 "
            f"({time.time()-t0:.0f}s)", flush=True)
        parts.append(s)
    return np.concatenate(parts)


XGB_CFGS = [
    dict(name="xb_base",  d=6, lr=0.05, ne=1000, ss=0.8, cs=0.8, alpha=0.0),   # 复现 .8665
    dict(name="xb_d5",    d=5, lr=0.05, ne=1000, ss=0.8, cs=0.8, alpha=0.0),
    dict(name="xb_d7_l03", d=7, lr=0.03, ne=1500, ss=0.8, cs=0.8, alpha=0.0),
    dict(name="xb_cs5_ss1", d=6, lr=0.05, ne=1000, ss=1.0, cs=0.5, alpha=0.0),
    dict(name="xb_spw05", d=6, lr=0.05, ne=1000, ss=0.8, cs=0.8, alpha=0.5),
]
LGB_CFGS = [
    dict(name="lb_base", nl=31, lr=0.05, ne=1000, mcs=20, ss=0.8, cs=0.8, alpha=0.0),  # 复现 .8535
    dict(name="lb_nl63", nl=63, lr=0.05, ne=1000, mcs=20, ss=0.8, cs=0.8, alpha=0.0),
    dict(name="lb_l03",  nl=31, lr=0.03, ne=2000, mcs=10, ss=0.8, cs=0.8, alpha=0.0),
]
LR_GRID = [(w, a, c) for w in (2, 3, None)
           for a in (0.5, 1.0) for c in (0.3, 1.0, 3.0)]  # window(None=全月)


def main() -> None:
    t_all = time.time()
    wide = load_wide()
    mo = sorted(wide)
    parts = [wide[m][["user_id", "item_id", "label", "month"]] for m in mo[1:]]
    df = pd.concat(parts, ignore_index=True)
    label = df["label"].to_numpy()
    print(f"[init] 4 折叠共 {len(df):,} 行 / 正例 {int(label.sum()):,} "
          f"({label.mean():.4%})", flush=True)

    scores: dict[str, np.ndarray] = {}
    rows = []   # (kind, key, AUC, gAUC_w, gAUC_m)

    def run_cfg(kind: str, key: str, fn):
        t0 = time.time()
        print(f"[run] {kind} {key}", flush=True)
        s = fn()
        r = rankdata(s) / len(s)
        auc = roc_auc_score(label, r)
        wg, mg = gauc_of(r, df)
        scores[key] = s
        rows.append((kind, key, auc, wg, mg))
        print(f"  -> {key}: AUC={auc:.4f} gAUC_w={wg:.4f} gAUC_m={mg:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    noop = lambda *a, **k: None
    for cfg in XGB_CFGS:
        run_cfg("xgb", f"xgb:{cfg['name']}",
                lambda c=cfg: fold_tree(wide, "xgb", c, print))
    for cfg in LGB_CFGS:
        run_cfg("lgb", f"lgb:{cfg['name']}",
                lambda c=cfg: fold_tree(wide, "lgb", c, print))
    for w, a, c in LR_GRID:
        run_cfg("lr", f"lr:w{w}_a{a}_C{c}",
                lambda w=w, a=a, c=c: fold_lr(wide, w, a, c, noop))

    res = pd.DataFrame(rows, columns=["kind", "key", "auc", "gAUC_w", "gAUC_m"])
    print("\n=== 各配置单模型 gAUC_m（按 kind 排序）===")
    print(res.sort_values(["kind", "gAUC_m"], ascending=[True, False]).to_string(index=False))

    # ---- 每类最优 + 基线参考 ----
    best = {}
    for kind in ("xgb", "lgb", "lr"):
        sub = res[res["kind"] == kind].sort_values("gAUC_m", ascending=False)
        best[kind] = sub.iloc[0]["key"]
    lgb_base = "lgb:lb_base"
    xgb_base = "xgb:xb_base"
    lr_base = "lr:w2_a1.0_C1.0"   # = lr_eq_l2_sp
    assert lr_base in scores and lgb_base in scores and xgb_base in scores

    lines = ["# 组等权重调参 OOF（wide_v2 44 特征；对照 v6 基线 "
             ".8535/.8665/.8687 → 融合 .8721）", ""]
    lines.append("单模型（新旧对照）:")
    for key in (lgb_base, xgb_base, lr_base,
                best["xgb"], best["lgb"], best["lr"]):
        r = res[res["key"] == key].iloc[0]
        lines.append(f"  {key:22s} AUC={r['auc']:.4f} gAUC_w={r['gAUC_w']:.4f} "
                     f"gAUC_m={r['gAUC_m']:.4f}")
    lines.append("")

    # ---- fine-grid 融合 ----
    def grid(comps, tag, step=0.025):
        rmat = np.column_stack([rankdata(scores[c]) / len(scores[c]) for c in comps])
        m = len(comps)
        n = int(round(1.0 / step)) + 1
        gl = []
        for idx in itertools.product(range(n), repeat=m - 1):
            if sum(idx) > n - 1:
                continue
            wgt = np.array(list(idx) + [n - 1 - sum(idx)], dtype="float64") * step
            s = rmat @ wgt
            wg, mg = gauc_of(s, df)
            gl.append((tuple(round(float(x), 3) for x in wgt),
                       roc_auc_score(label, s), wg, mg))
        gdf = pd.DataFrame(gl, columns=["w", "auc", "gAUC_w", "gAUC_m"])
        gdf = gdf.sort_values("gAUC_m", ascending=False).reset_index(drop=True)
        lines.append(f"== {tag} top6 ==")
        for _, r in gdf.head(6).iterrows():
            lines.append(f"w={r['w']}  AUC={r['auc']:.4f} gAUC_w={r['gAUC_w']:.4f} "
                         f"gAUC_m={r['gAUC_m']:.4f}")
        lines.append("")
        return gdf

    g_base = grid([lgb_base, xgb_base, lr_base], "三分量基线（应复现 .8721）")
    comps_new = [best["lgb"], best["xgb"], best["lr"]]
    g_new = grid(comps_new, f"三分量最优配置 {comps_new}")
    champ = ("基线", g_base.iloc[0]) if g_base.iloc[0]["gAUC_m"] >= g_new.iloc[0]["gAUC_m"] \
        else ("重调参", g_new.iloc[0])

    # ---- 落盘 OOF（最优配置列 + 基线列，供 v7 复用）----
    out = df.copy()
    out["xgb_eq_base"] = rankdata(scores[xgb_base]) / len(scores[xgb_base])
    out["lgb_eq_base"] = rankdata(scores[lgb_base]) / len(scores[lgb_base])
    out["lr_eq_l2_sp"] = rankdata(scores[lr_base]) / len(scores[lr_base])
    for key in (best["xgb"], best["lgb"], best["lr"]):
        out[key.replace(":", "_").replace(".", "p")] = rankdata(scores[key]) / len(scores[key])
    out.to_csv(REPLAY_DIR / "oof_eqtune_preds.csv", index=False, encoding="utf-8")

    lines.append(f"== 冠军[{champ[0]}]: gAUC_m={champ[1]['gAUC_m']:.4f} w={champ[1]['w']} "
                 f"（> .8721 才产 v7）==")
    json.dump({"winner": champ[0], "weights": dict(zip(comps_new, champ[1]["w"])),
               "auc": float(champ[1]["auc"]), "gAUC_w": float(champ[1]["gAUC_w"]),
               "gAUC_m": float(champ[1]["gAUC_m"]),
               "best_per_kind": best},
              open(REPLAY_DIR / "eqtune_weights.json", "w"), indent=2)
    txt = "\n".join(lines) + "\n"
    (REPLAY_DIR / "eqtune_report.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"[done] 总耗时 {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
