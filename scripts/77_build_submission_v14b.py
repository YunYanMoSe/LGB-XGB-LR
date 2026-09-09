"""候选集打分 v14b（v15port 修正）：双专家只用于树，LR 保持 v6 单模型。

scripts/75 的 v14 完整移植 Oct sanity 掉分 −0.053（vs v6 0.56866）。scripts/76 隔离诊断
把根因钉死：**掉分全在「把 LR 也按 prior 切双专家」**——A 分量 LR comp 0.46417 Δ−0.0979，
而 A 分量 LGB +0.0036（略好）、XGB −0.0056；activity 过滤单独 −0.0013（中性，保留）。
根因机理：v6 里 LR(.6 主分量) 的威力来自把 ui_owned 当特征在**全池**线性学跨组序；一旦按
prior 切池，每个专家池内 ui_owned 变常数（repurchase 池全=1），LR 失去杠杆且小池过拟合。
组长 v15 架构里"首购/复购双专家"本来就是 LGBM 树，LR 只做末端 stacker/融合器 —— 我误把
LR 当可切分的基模型。

v14b 修正 = 移植的只是树侧：LGB/XGB 各按 ui_owned 切首购/复购双专家（仅在当月有真实交易
的用户行 active==1 上训 = 条件负样本），LR 保持 v6 原配方（近2月、eq×spw、不切分、全池）。
融合仍是 v6 三族权 {.1,.3,.6} + 全局百分位秩。OOF 复合已证与 Nov 反向（scripts/74），
只做 Oct 单月 sanity 记录，Nov 由板上裁决。

产物：Oct sanity 报告 + 提交 csv + v14b×v6 秩融合保险
用法：venv\\Scripts\\python.exe scripts\\77_build_submission_v14b.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SEED, set_seed
from src import candidate as cand
from src.candidate_v2 import V2_FEATS

set_seed(SEED)

REPLAY_DIR = PROJECT_ROOT / "outputs" / "candidate" / "replay"
LEADER_CLEAN = PROJECT_ROOT / "data" / "processed" / "leader_clean.csv"
CAND_INPUT = PROJECT_ROOT / "train_data" / "sample_candidates.csv"
SC_CUT = pd.Timestamp("2010-11-01")
SC_VEC = PROJECT_ROOT / "outputs" / "candidate" / "v9_vecs_20101101_skip.npz"
FEATS = list(cand.FEATURES_CAND) + list(V2_FEATS)
NAME = "cand-align-v14b-v15port"
W = {"lgb_macro_eq": 0.1, "xgb_macro_eq": 0.3, "lr_eq_l2_sp": 0.6}
LR_LAST_N = 2
MONTHS = [f"2010-{m:02d}-01" for m in range(6, 11)]
OCT = "2010-10-01"
SPLIT_KINDS = ["lgb_macro_eq", "xgb_macro_eq"]   # 树切双专家
LR_KIND = "lr_eq_l2_sp"                          # LR 单模型（v6 原配方）


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
    sw = eq_group_sw(tr, with_sp=with_sp)
    if kind == "lgb_macro_eq":
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                             min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=1.0, random_state=SEED, verbosity=-1,
                             deterministic=True, n_jobs=1)
        clf.fit(tr[FEATS], tr["label"], sample_weight=sw)
    elif kind == "xgb_macro_eq":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8,
                            tree_method="hist", n_jobs=1, random_state=SEED,
                            eval_metric="auc", verbosity=0)
        clf.fit(tr[FEATS], tr["label"], sample_weight=sw)
    else:
        clf = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, C=1.0, random_state=SEED))
        clf.fit(tr[FEATS], tr["label"], logisticregression__sample_weight=sw)
    return clf


def pred_one(clf, frame):
    return clf.predict_proba(frame[FEATS])[:, 1].astype("float64")


def score_split_experts(cond_pool, kind, te, with_sp):
    """按 ui_owned 切 0/1 两专家（active 行训练），对 te 逐行路由打分。"""
    s = np.zeros(len(te), dtype="float64")
    prior = te["ui_owned"].round().astype(int).to_numpy()
    for pr in (0, 1):
        sub = cond_pool.loc[cond_pool["ui_owned"].round().astype(int) == pr]
        clf = fit_one(kind, sub, with_sp=with_sp)
        mask = prior == pr
        if mask.any():
            s[mask] = pred_one(clf, te.loc[mask])
    return s


def teacher_comp(df: pd.DataFrame, s: np.ndarray):
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
    hit = (sub["rd"].to_numpy() <= 10.0) & (lbl == 1)
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


def fused_from(score_map):
    n = next(iter(score_map.values())).size
    return sum(W[k] * rankdata(score_map[k]) / n for k in W)


def main() -> None:
    t0 = time.time()
    wide = load_wide(MONTHS)
    act = activity_sets()
    for m, df in wide.items():
        df["active"] = np.fromiter((str(u) in act[m] for u in df["user_id"]), bool, len(df))
    train_months = [m for m in MONTHS if m < OCT]
    pool = pd.concat([wide[m] for m in train_months], ignore_index=True)
    cond = pool.loc[pool["active"]].copy()
    te = wide[OCT].copy()
    n_te = len(te)

    # ---- Oct sanity ----
    res = {}
    for kind in SPLIT_KINDS:
        res[kind] = score_split_experts(cond, kind, te, with_sp=False)
    lr_pool = pool[pool["month"].isin(train_months[-LR_LAST_N:])]
    lr_clf = fit_one(LR_KIND, lr_pool, with_sp=True)
    res[LR_KIND] = pred_one(lr_clf, te)
    port = fused_from(res)

    eq = pd.read_csv(REPLAY_DIR / "oof_eqg_preds.csv", dtype={"item_id": str})
    eqo = eq[eq["month"] == OCT].copy()
    key = te["user_id"].astype(str) + "_" + te["item_id"].astype(str)
    assert (key.values == (eqo["user_id"].astype(str) + "_" + eqo["item_id"].astype(str)).values).all()
    v6f = (0.1 * rankdata(eqo["lgb_macro_eq"].to_numpy()) / n_te
           + 0.3 * rankdata(eqo["xgb_macro_eq"].to_numpy()) / n_te
           + 0.6 * rankdata(eqo["lr_eq_l2_sp"].to_numpy()) / n_te)
    c_p = teacher_comp(te, port); c_v = teacher_comp(te, v6f)
    lines = ["## v14b(v15port修正) Oct-2010 sanity（teacher 复合；非 Nov 门禁）",
             f"test={OCT[:7]} 行={n_te}",
             f"| 模型 | composite | GAUC | NDCG@10 | Recall@10 |",
             f"| --- | --- | --- | --- | --- |",
             f"| v14bport | {c_p[0]:.5f} | {c_p[1]:.4f} | {c_p[2]:.4f} | {c_p[3]:.4f} |",
             f"| v6_fused(同集) | {c_v[0]:.5f} | {c_v[1]:.4f} | {c_v[2]:.4f} | {c_v[3]:.4f} |",
             f"| Δ | {c_p[0]-c_v[0]:+.5f} | | | |",
             "", "v14b=树切双专家(active行) + LR v6原配方(不切)。诊断参考 scripts/76（v14 全切 LR 掉 .098）。", ""]
    txt = "\n".join(lines)
    (REPLAY_DIR / "v15port_v14b_oct_sanity_report.txt").write_text(txt, encoding="utf-8")
    print(txt, flush=True)

    # ---- serve（final：全 06..10 池）----
    pool_all = pd.concat([wide[m] for m in MONTHS], ignore_index=True)
    cond_all = pool_all.loc[pool_all["active"]].copy()
    lr_pool_all = pool_all[pool_all["month"].isin(MONTHS[-LR_LAST_N:])]
    clean = cand.load_clean_train(LEADER_CLEAN)
    raw = pd.read_csv(CAND_INPUT, dtype={c: str for c in ["user_id", "item_id"]})
    df = raw[["user_id", "item_id"]].copy()
    df["user_id"] = pd.to_numeric(df["user_id"]).astype("int64")
    df["item_id"] = df["item_id"].astype(str)
    bundle = cand.build_bundle(clean, cut=SC_CUT, vec_cache=SC_VEC)
    Xb = cand.features_for_pairs(bundle, df["user_id"].to_numpy(), df["item_id"].to_numpy())
    Xv = __import__("src.candidate_v2", fromlist=["v2_features"]).v2_features(clean, SC_CUT, df)
    Xs = pd.concat([Xb.reset_index(drop=True), Xv.reset_index(drop=True)], axis=1)
    assert list(Xs.columns) == FEATS and Xs.isna().sum().sum() == 0
    prior_serve = Xs["ui_owned"].round().astype(int).to_numpy()
    print(f"[serve] final 特征 {len(Xs):,} 行；tree experts 训 {len(cond_all):,} 行；"
          f"LR 近2月 {len(lr_pool_all):,} 行", flush=True)

    smap = {}
    for kind in SPLIT_KINDS:
        s = np.zeros(len(Xs), dtype="float64")
        for pr in (0, 1):
            sub = cond_all.loc[cond_all["ui_owned"].round().astype(int) == pr]
            clf = fit_one(kind, sub, with_sp=False)
            mask = prior_serve == pr
            if mask.any():
                s[mask] = pred_one(clf, Xs.loc[mask])
        smap[kind] = s
    lr_final = fit_one(LR_KIND, lr_pool_all, with_sp=True)
    smap[LR_KIND] = pred_one(lr_final, Xs)
    port_score = fused_from(smap)
    assert np.isfinite(port_score).all() and len(port_score) == len(raw)

    v6 = pd.read_csv(PROJECT_ROOT / "outputs" / "candidate" / "sample_submission_cand-align-v6-fuse.csv")
    assert (v6["user_id"].astype(str).reset_index(drop=True) == raw["user_id"].reset_index(drop=True)).all()
    r6 = rankdata(v6["score"].to_numpy(float)) / len(raw)
    rp = rankdata(port_score) / len(raw)

    out = pd.DataFrame({"user_id": raw["user_id"], "item_id": raw["item_id"], "score": port_score})
    out_path = PROJECT_ROOT / "outputs" / "candidate" / f"sample_submission_{NAME}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    meta = {"name": NAME,
            "desc": "v15port修正: 树(lgb/xgb)按ui_owned切首购/复购双专家且只在active行训, "
                    "LR保持v6原配方(近2月 eq×spw 不切); 融合权{.1,.3,.6}全局百分位秩",
            "why_lr_unsplit": "scripts/76: LR切双专家 Oct Δ−0.098(ui_owned池内常数,失去全池杠杆); "
                              "树切LGB +0.0036",
            "oct_sanity": c_p, "oct_sanity_v6_same": c_v,
            "weights": W, "train_months": MONTHS, "cut": str(SC_CUT.date())}
    (PROJECT_ROOT / "models" / f"candidate_meta_{NAME}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {NAME} → {out_path.name} ({time.time()-t0:.0f}s)", flush=True)

    for wb in (0.50, 0.75):
        score = wb * rp + (1.0 - wb) * r6
        nm = f"{NAME}_m6_{int(wb*100):02d}"
        p = PROJECT_ROOT / "outputs" / "candidate" / f"sample_submission_{nm}.csv"
        pd.DataFrame({"user_id": raw["user_id"], "item_id": raw["item_id"], "score": score}
                     ).to_csv(p, index=False, encoding="utf-8")
        (PROJECT_ROOT / "models" / f"candidate_meta_{nm}.json").write_text(
            json.dumps({"name": nm, "desc": f"{wb}×v14b 秩 + {1-wb:.2f}×v6 秩",
                        "w_v14b": wb}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[out] {nm} → {p.name}", flush=True)
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
