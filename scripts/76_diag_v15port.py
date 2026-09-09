"""v14(v15port) Oct 掉分隔离诊断：-0.05 vs v6 同集，定位掉分在哪个机制/分量。

v14 Oct sanity composite 0.51586 << v6 同集 0.56866（scripts/75）。-0.05 远超"机制不迁移"
应有的量级（组长自己的 v15_calibrated 在回放上也就 -0.006 级），需先定位是：
  (a) activity 条件过滤把训练集砍到 47.7% 造成的信息损失；
  (b) prior 切分双专家在 Oct 上本身变弱；
  (c) LR 专家（近2月×spw×eq 在切分后子池）退化；
  (d) 跨专家全局百分位秩融合的对齐方式有 bug。

方法：固定 Oct-2010 测试（wide_v2_2010-10，53,034 行），训练 06..09；对照 4 个配方
分别给每个族（lgb/xgb/lr）打 Oct 分，逐一与 v6 同族基线（oof_eqg_preds Oct 行）比
单分量 GAUC/composite；再做 3×4=12 个融合网格（不是选权，是看融合掉分是否来自某分量）。

记录：outputs/candidate/replay/v15port_diag_report.txt
用法：venv\\Scripts\\python.exe scripts\\76_diag_v15port.py
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

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
from src.candidate_v2 import V2_FEATS

set_seed(SEED)

REPLAY_DIR = PROJECT_ROOT / "outputs" / "candidate" / "replay"
LEADER_CLEAN = PROJECT_ROOT / "data" / "processed" / "leader_clean.csv"
FEATS = list(cand.FEATURES_CAND) + list(V2_FEATS)
MONTHS = [f"2010-{m:02d}-01" for m in range(6, 11)]
OCT = "2010-10-01"
LR_LAST_N = 2
KINDS = ["lgb_macro_eq", "xgb_macro_eq", "lr_eq_l2_sp"]
TAG = {"lgb_macro_eq": "lgb", "xgb_macro_eq": "xgb", "lr_eq_l2_sp": "lr"}


def load_wide(months) -> dict[str, pd.DataFrame]:
    return {m: pd.read_csv(REPLAY_DIR / f"wide_v2_{m[:7]}.csv", dtype={"item_id": str})
            for m in months}


def activity_sets() -> dict[str, set[str]]:
    cl = pd.read_csv(LEADER_CLEAN, usecols=["CustomerID", "InvoiceDate"])
    cl["CustomerID"] = cl["CustomerID"].astype(str)
    cl["ym"] = pd.to_datetime(cl["InvoiceDate"]).dt.strftime("%Y-%m")
    return {m: set(cl.loc[cl["ym"] == m[:7], "CustomerID"]) for m in MONTHS}


def eq_group_sw(tr, with_sp):
    n_g = tr.groupby(["user_id", "month"], sort=True).size()
    keys = pd.MultiIndex.from_arrays([tr["user_id"], tr["month"].astype(str)])
    w = 1.0 / n_g.reindex(keys).to_numpy(dtype="float64")
    if not with_sp:
        return w.astype("float64")
    spw = float((tr["label"] == 0).sum() / max(1, int(tr["label"].sum())))
    return (w * np.where(tr["label"] == 1, spw, 1.0)).astype("float64")


def fit_one(kind, tr, with_sp):
    if kind == "lgb_macro_eq":
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                             min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=1.0, random_state=SEED, verbosity=-1,
                             deterministic=True, n_jobs=1)
    elif kind == "xgb_macro_eq":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8,
                            tree_method="hist", n_jobs=1, random_state=SEED,
                            eval_metric="auc", verbosity=0)
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        clf = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, C=1.0, random_state=SEED))
    sw = eq_group_sw(tr, with_sp=with_sp)
    if kind == "lr_eq_l2_sp":
        clf.fit(tr[FEATS], tr["label"], logisticregression__sample_weight=sw)
    else:
        clf.fit(tr[FEATS], tr["label"], sample_weight=sw)
    return clf


def pred_one(clf, te):
    return clf.predict_proba(te[FEATS])[:, 1].astype("float64")


# teacher 复合（scripts/59 同口径）
def comp_on(df: pd.DataFrame, s: np.ndarray):
    df = df.copy()
    df["ym"] = pd.to_datetime(df["month"]).dt.strftime("%Y-%m")
    g, _ = pd.factorize(df["user_id"].astype(str) + "_" + df["ym"].astype(str))
    lbl = df["label"].to_numpy(dtype="int64")
    n_g = np.bincount(g); p_g = np.bincount(g, weights=lbl.astype("float64"))
    sub = pd.DataFrame({"g": g, "s": np.asarray(s, float), "lbl": lbl})
    sub["ra"] = sub.groupby("g")["s"].rank(method="average")
    sp = np.bincount(g, weights=(sub["ra"].to_numpy() * lbl.astype("float64")))
    nn = n_g - p_g
    with np.errstate(divide="ignore", invalid="ignore"):
        auc_g = np.where((p_g > 0) & (nn > 0),
                         (sp - p_g * (p_g + 1.0) / 2.0) / (p_g * nn), np.nan)
    sub["rd"] = sub.groupby("g")["s"].rank(method="first", ascending=False)
    top = sub["rd"].to_numpy() <= 10.0
    hit = top & (lbl == 1)
    hits_g = np.bincount(g, weights=hit.astype("float64"))
    dcg_g = np.bincount(g, weights=np.where(
        hit, 1.0 / np.log2(sub["rd"].to_numpy() + 1.0), 0.0))
    denom = np.log2(np.arange(1, 11, dtype="float64") + 1.0)
    idcg = np.array([denom[:int(min(v, 10))].sum() for v in p_g])
    with np.errstate(divide="ignore", invalid="ignore"):
        ndcg_g = np.where(p_g > 0, dcg_g / np.where(idcg > 0, idcg, np.nan), np.nan)
        rec_g = np.where(p_g > 0, hits_g / p_g, np.nan)
    def macro(m):
        m = m[~np.isnan(m)]
        return float(m.mean()) if len(m) else float("nan")
    a, d, r = macro(auc_g), macro(ndcg_g), macro(rec_g)
    return 0.4 * a + 0.4 * d + 0.2 * r, a, d, r


def main() -> None:
    t0 = time.time()
    wide = load_wide(MONTHS)
    act = activity_sets()
    for m, df in wide.items():
        df["active"] = np.fromiter((str(u) in act[m] for u in df["user_id"]), bool, len(df))
    train_months = [m for m in MONTHS if m < OCT]
    pool = pd.concat([wide[m] for m in train_months], ignore_index=True)
    te = wide[OCT].copy()
    n_te = len(te)
    print(f"[diag] test={OCT[:7]} 行={n_te}  train={[m[:7] for m in train_months]}", flush=True)

    # v6 基线分量（oof_eqg Oct 行，训练 06..09 同窗）
    eq = pd.read_csv(REPLAY_DIR / "oof_eqg_preds.csv", dtype={"item_id": str})
    eqo = eq[eq["month"] == OCT].copy()
    key = te["user_id"].astype(str) + "_" + te["item_id"].astype(str)
    assert (key.values == (eqo["user_id"].astype(str) + "_" + eqo["item_id"].astype(str)).values).all()
    v6 = {k: eqo[k].to_numpy(float) for k in KINDS}
    v6f = sum(W for W in [0.1 * rankdata(v6["lgb_macro_eq"]) / n_te,
                          0.3 * rankdata(v6["xgb_macro_eq"]) / n_te,
                          0.6 * rankdata(v6["lr_eq_l2_sp"]) / n_te])

    def report(name, scores, score6=None):
        lines = []
        c = comp_on(te, scores)
        lines.append(f"[{name}] composite={c[0]:.5f} (GAUC {c[1]:.4f} NDCG {c[2]:.4f} Rec {c[3]:.4f})")
        if score6 is not None:
            c6 = comp_on(te, score6)
            lines.append(f"[{name}]   v6融合={c6[0]:.5f}  Δ={c[0]-c6[0]:+.5f}")
        return lines

    L = []
    # ---- 配方 A: v14 原样（active 过滤 + prior 切分双专家），逐分量 + 融合 ----
    cond = pool.loc[pool["active"]].copy()
    resA = {}
    for kind in KINDS:
        use = cond if "lr_" not in kind else cond[cond["month"].isin(train_months[-LR_LAST_N:])]
        s_all = np.zeros(n_te)
        for pr in (0, 1):
            sub = use.loc[use["ui_owned"].round().astype(int) == pr]
            clf = fit_one(kind, sub, with_sp=("lr_" in kind))
            mask = te["ui_owned"].round().astype(int).to_numpy() == pr
            s_all[mask] = pred_one(clf, te.loc[mask])
        resA[kind] = s_all
    L += report("A v14完整(active+split) 融合", 0.1*rankdata(resA["lgb_macro_eq"])/n_te
                + 0.3*rankdata(resA["xgb_macro_eq"])/n_te + 0.6*rankdata(resA["lr_eq_l2_sp"])/n_te,
                v6f)
    for k in KINDS:
        L += [f"   A 分量 {k:14s}: comp={comp_on(te, resA[k])[0]:.5f} | "
              f"v6同分量={comp_on(te, v6[k])[0]:.5f}  Δ={comp_on(te, resA[k])[0]-comp_on(te, v6[k])[0]:+.5f}"]
    L.append("")

    # ---- 配方 B: 只 active 过滤，不切分（单模型）----
    resB = {}
    for kind in KINDS:
        use = cond if "lr_" not in kind else cond[cond["month"].isin(train_months[-LR_LAST_N:])]
        clf = fit_one(kind, use, with_sp=("lr_" in kind))
        resB[kind] = pred_one(clf, te)
    L += report("B 仅active过滤(不split)", 0.1*rankdata(resB["lgb_macro_eq"])/n_te
                + 0.3*rankdata(resB["xgb_macro_eq"])/n_te + 0.6*rankdata(resB["lr_eq_l2_sp"])/n_te,
                v6f)
    L.append("")

    # ---- 配方 C: 只 prior 切分（不 active 过滤）----
    resC = {}
    for kind in KINDS:
        use = pool if "lr_" not in kind else pool[pool["month"].isin(train_months[-LR_LAST_N:])]
        s_all = np.zeros(n_te)
        for pr in (0, 1):
            sub = use.loc[use["ui_owned"].round().astype(int) == pr]
            clf = fit_one(kind, sub, with_sp=("lr_" in kind))
            mask = te["ui_owned"].round().astype(int).to_numpy() == pr
            s_all[mask] = pred_one(clf, te.loc[mask])
        resC[kind] = s_all
    L += report("C 仅prior切分(不active)", 0.1*rankdata(resC["lgb_macro_eq"])/n_te
                + 0.3*rankdata(resC["xgb_macro_eq"])/n_te + 0.6*rankdata(resC["lr_eq_l2_sp"])/n_te,
                v6f)
    for k in KINDS:
        L += [f"   C 分量 {k:14s}: comp={comp_on(te, resC[k])[0]:.5f} | "
              f"v6同分量={comp_on(te, v6[k])[0]:.5f}  Δ={comp_on(te, resC[k])[0]-comp_on(te, v6[k])[0]:+.5f}"]
    L.append("")

    # ---- 配方 D: v6 原样（不 active 不 split）对照组 ----
    resD = {}
    for kind in KINDS:
        use = pool if "lr_" not in kind else pool[pool["month"].isin(train_months[-LR_LAST_N:])]
        clf = fit_one(kind, use, with_sp=("lr_" in kind))
        resD[kind] = pred_one(clf, te)
    L += report("D v6同窗复现(不active不split)", 0.1*rankdata(resD["lgb_macro_eq"])/n_te
                + 0.3*rankdata(resD["xgb_macro_eq"])/n_te + 0.6*rankdata(resD["lr_eq_l2_sp"])/n_te,
                v6f)
    L.append("")

    # ---- 配方 E: 交叉验证 v6 LR 是不是真的在 last-2 (08,09)（v6 配方近2月 vs 全月）----
    txt = "\n".join(L) + "\n"
    (REPLAY_DIR / "v15port_diag_report.txt").write_text(txt, encoding="utf-8")
    print(txt, flush=True)
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
