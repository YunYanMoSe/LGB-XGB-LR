"""候选集打分 v14（v15port）：组长 v15 三个真机制移植进我方 v6 44 特征回放管线。

组长 v15（traincsv-v15-activity-moe）榜分 0.7845 >> 我方 v6 0.7782。V15_CODE_BUNDLE
源码研读 + scripts/74 实证后判定：**activity 门控树不移植**（p_activity 组内常数、秩不变，
单独 teacher 复合 .252≈随机，已证伪）；真正可迁移的三个机制为：
  (1) prior(=ui_owned) 首购/复购双专家 —— first_purchase(ui_owned==0) 与
      repurchase(ui_owned==1) 各自独立训一个模型（我们 v6 只在单模型内把 ui_owned 当特征，
      不分离函数类）；
  (2) activity-conditional 训练 —— 只训「该月有真实交易的用户」的行（剔 dormant 用户
      全负组 = 干净负样本），与组长 add_activity_labels 同口径（当月任一交易即 active）；
  (3) 跨专家尺度对账 —— 组长用前向 LR stacker over [logit(pa),logit(pc),prior,lc*prior]；
      pa 组内常数、只抬组内截距 ⇒ 秩不变，真正起作用的是把两个专家概率归一到一个可比的
      用户内序。我方融合本身就是「每分量→全局百分位秩」再做加权，天然完成跨专家对账，
      故不额外引入其 LR stacker（避免组内常数项，且省一层上月拟合的复杂度）。

移植载体 = 我方 v6 已证最强引擎（脚本 47/48/49）：LGB(eq)×.1 + XGB(eq)×.3 + LR(eq×spw,
近2月)×.6，全局百分位秩融合，44 特征，cut=11-01。区别仅在于：每个族模型先按 ui_owned
切成首购/复购两个专家、各在 active==1 行上以 eq 权重独立训，预测时按该行 ui_owned 路由到
对应专家。=「组长机制 × 我方平台」的干净 A/B（其他都锁死与 v6 同配方）。

落地（2026-09-09 复盘裁决）：OOF teacher 复合已证与 Nov 板反向（scripts/74），
本模型不做 Nov 门禁、只做 Oct 单月 sanity 记录 + serve 上板由真实 Nov 裁决。

产物：
  * outputs/candidate/replay/v15port_oct_sanity_report.txt —— Oct 单月 sanity（非门禁）
  * outputs/candidate/sample_submission_cand-align-v14-v15port.csv（纯移植）
  * outputs/candidate/sample_submission_cand-align-v14-v15port_m6_{50,75}.csv（v14×v6 秩融合保险）
用法：venv\\Scripts\\python.exe scripts\\75_build_submission_v14_v15port.py
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
OOF_EQG = REPLAY_DIR / "oof_eqg_preds.csv"

FEATS = list(cand.FEATURES_CAND) + list(V2_FEATS)
NAME = "cand-align-v14-v15port"
W = {"lgb_macro_eq": 0.1, "xgb_macro_eq": 0.3, "lr_eq_l2_sp": 0.6}  # v6 融合权（锁死对照）
LR_LAST_N = 2
# v6 用 2010-06..10 五个月池（scripts/47/49 同窗）
MONTHS = [f"2010-{m:02d}-01" for m in range(6, 11)]


def load_wide(months) -> dict[str, pd.DataFrame]:
    return {m: pd.read_csv(REPLAY_DIR / f"wide_v2_{m[:7]}.csv", dtype={"item_id": str})
            for m in months}


def activity_sets() -> dict[str, set[str]]:
    """每月有任一真实交易的用户集（组长 add_activity_labels 同口径）。"""
    cl = pd.read_csv(LEADER_CLEAN, usecols=["CustomerID", "InvoiceDate"])
    cl["CustomerID"] = cl["CustomerID"].astype(str)
    cl["ym"] = pd.to_datetime(cl["InvoiceDate"]).dt.strftime("%Y-%m")
    out = {}
    for m in MONTHS:
        out[m] = set(cl.loc[cl["ym"] == m[:7], "CustomerID"])
    return out


def eq_group_sw(tr: pd.DataFrame, with_sp: bool) -> np.ndarray:
    """每 (user,month) 组内行等权 w=1/n_group；×spw 放大正例（线性用）。"""
    n_g = tr.groupby(["user_id", "month"], sort=True).size()
    keys = pd.MultiIndex.from_arrays([tr["user_id"], tr["month"].astype(str)])
    w = 1.0 / n_g.reindex(keys).to_numpy(dtype="float64")
    if not with_sp:
        return w.astype("float64")
    spw = float((tr["label"] == 0).sum() / max(1, int(tr["label"].sum())))
    return (w * np.where(tr["label"] == 1, spw, 1.0)).astype("float64")


def flag_col(tr: pd.DataFrame, on: str = "month"):
    pass  # (占位不用)


def train_learners(pool: pd.DataFrame, pool_months: list[str],
                   kinds: list[str]) -> dict[str, dict[int, object]]:
    """给一族可训集训首购/复购双专家。

    机制(2)：只在 active==1 行上训（dormant 用户全负组整月剔除）。
    机制(1)：按 ui_owned 切 0/1 两专家，各自 eq 权重独立 fit。
    kinds: 'lgb_macro_eq'/'xgb_macro_eq'/'lr_eq_l2_sp'，'lr_*' 取近 LR_LAST_N 月。
    返回 {kind: {prior: model}}。
    """
    out = {}
    for kind in kinds:
        use = pool if "lr_" not in kind else pool[pool["month"].isin(pool_months[-LR_LAST_N:])]
        models = {}
        for prior in (0, 1):
            sub = use.loc[use["ui_owned"].round().astype(int) == prior]
            if sub["label"].nunique() != 2:
                raise ValueError(f"{kind} prior={prior} 训练集缺类: n={len(sub)}")
            sw = eq_group_sw(sub, with_sp=("lr_" in kind))
            if kind == "lgb_macro_eq":
                from lightgbm import LGBMClassifier
                clf = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                                     min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                                     scale_pos_weight=1.0, random_state=SEED, verbosity=-1,
                                     deterministic=True, n_jobs=1)
                clf.fit(sub[FEATS], sub["label"], sample_weight=sw)
            elif kind == "xgb_macro_eq":
                from xgboost import XGBClassifier
                clf = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6,
                                    subsample=0.8, colsample_bytree=0.8,
                                    tree_method="hist", n_jobs=1, random_state=SEED,
                                    eval_metric="auc", verbosity=0)
                clf.fit(sub[FEATS], sub["label"], sample_weight=sw)
            else:  # lr_eq_l2_sp
                clf = make_pipeline(StandardScaler(), LogisticRegression(
                    max_iter=2000, C=1.0, random_state=SEED))
                clf.fit(sub[FEATS], sub["label"], logisticregression__sample_weight=sw)
            models[prior] = clf
        out[kind] = models
    return out


def predict_models(models: dict[str, dict[int, object]], frame: pd.DataFrame) -> dict[str, np.ndarray]:
    prior = frame["ui_owned"].round().astype(int).to_numpy()
    scores = {}
    for kind, m in models.items():
        s = np.zeros(len(frame), dtype="float64")
        for flag, clf in m.items():
            mask = prior == flag
            if mask.any():
                s[mask] = clf.predict_proba(frame.loc[mask, FEATS])[:, 1].astype("float64")
        scores[kind] = s
    return scores


def fused_port(score_map: dict[str, np.ndarray]) -> np.ndarray:
    n = next(iter(score_map.values())).size
    return sum(W[k] * rankdata(score_map[k]) / n for k in W)


# ---- scripts/59/74 同口径 teacher 复合（仅记录用，非门禁）----
def group_meta(df):
    g, _ = pd.factorize(df["user_id"].astype(str) + "_" + df["ym"].astype(str))
    lbl = df["label"].to_numpy(dtype="int64")
    n_g = np.bincount(g)
    p_g = np.bincount(g, weights=lbl.astype("float64"))
    return g, n_g, p_g, lbl


def comp_on(df: pd.DataFrame, s: np.ndarray):
    df = df.copy()
    df["ym"] = pd.to_datetime(df["month"]).dt.strftime("%Y-%m")
    g, n_g, p_g, lbl = group_meta(df)
    sub = pd.DataFrame({"g": g, "s": np.asarray(s, float), "lbl": lbl})
    sub["ra"] = sub.groupby("g")["s"].rank(method="average")
    sp = np.bincount(g, weights=(sub["ra"].to_numpy() * lbl.astype("float64")))
    nn = n_g - p_g
    with np.errstate(divide="ignore", invalid="ignore"):
        auc_g = np.where((p_g > 0) & (nn > 0),
                         (sp - p_g * (p_g + 1.0) / 2.0) / (p_g * nn), np.nan)
    sub["rd"] = sub.groupby("g")["s"].rank(method="first", ascending=False)
    top = (sub["rd"].to_numpy() <= 10.0)
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
    GAUC, NDCG, REC = macro(auc_g), macro(ndcg_g), macro(rec_g)
    return 0.4 * GAUC + 0.4 * NDCG + 0.2 * REC, GAUC, NDCG, REC


def main() -> None:
    t0 = time.time()
    wide = load_wide(MONTHS)
    act = activity_sets()
    # 给每行标 active（该月用户有任一交易）
    for m, df in wide.items():
        df["active"] = np.fromiter(
            (str(u) in act[m] for u in df["user_id"]), bool, len(df))

    # ---------- Oct 单月 sanity（记录用，非 Nov 门禁）----------
    octm = "2010-10-01"
    train_months = [m for m in MONTHS if m < octm]          # 06..09
    pool = pd.concat([wide[m] for m in train_months], ignore_index=True)
    cond = pool.loc[pool["active"]].copy()
    print(f"[sanity] train pool {len(pool):,} 行 → active 过滤 {len(cond):,} "
          f"({len(cond)/len(pool):.1%})；LR 近2月 {train_months[-LR_LAST_N:]}", flush=True)
    for pr in (0, 1):
        sub = cond.loc[cond["ui_owned"].round().astype(int) == pr]
        print(f"    prior={pr}（{'first_purchase' if pr==0 else 'repurchase'}）: "
              f"n={len(sub):,} pos={int(sub['label'].sum()):,} "
              f"rate={sub['label'].mean():.4f}", flush=True)
    models_s = train_learners(cond, train_months,
                              ["lgb_macro_eq", "xgb_macro_eq", "lr_eq_l2_sp"])
    te = wide[octm].copy()
    sc = predict_models(models_s, te)
    port = fused_port(sc)
    # v6 同 Oct 基准（读 oof_eqg，逐行对齐 Oct）
    eq = pd.read_csv(OOF_EQG, dtype={"item_id": str})
    eq_oct = eq[eq["month"] == octm].copy()
    te_key = te["user_id"].astype(str) + "_" + te["item_id"].astype(str)
    eq_key = eq_oct["user_id"].astype(str) + "_" + eq_oct["item_id"].astype(str)
    assert te_key.isin(set(eq_key)).all() and eq_key.isin(set(te_key)).all(), "Oct 行未对齐"
    m6 = eq_oct.set_index(["user_id", "item_id"])
    te_m = te.set_index(["user_id", "item_id"])
    sub = te_m.join(m6[["lgb_macro_eq", "xgb_macro_eq", "lr_eq_l2_sp"]], how="inner")
    n_ = len(sub)
    v6f = (W["lgb_macro_eq"] * rankdata(sub.lgb_macro_eq.to_numpy()) / n_
           + W["xgb_macro_eq"] * rankdata(sub.xgb_macro_eq.to_numpy()) / n_
           + W["lr_eq_l2_sp"] * rankdata(sub.lr_eq_l2_sp.to_numpy()) / n_)
    c_port = comp_on(sub.reset_index(), port[sub.index.to_numpy().argsort()] if False else port)
    # 直接用对齐后的行序重算 port
    order = te_m.index.get_indexer(sub.index)   # sub 行在原 te 中的位置
    c_port = comp_on(sub.reset_index(), port[order])
    c_v6 = comp_on(sub.reset_index(), v6f)
    lines = [
        "## v14(v15port) Oct-2010 单月 sanity（teacher 复合口径，非 Nov 门禁——scripts/74 已证回放复合与 Nov 反向）",
        f"train months={[m[:7] for m in train_months]}  active过滤后 {len(cond):,} 行  test={octm[:7]} 行 {len(sub):,}",
        f"| 模型 | composite | GAUC | NDCG@10 | Recall@10 |",
        f"| --- | --- | --- | --- | --- |",
        f"| v14port | {c_port[0]:.5f} | {c_port[1]:.4f} | {c_port[2]:.4f} | {c_port[3]:.4f} |",
        f"| v6_fused(同集) | {c_v6[0]:.5f} | {c_v6[1]:.4f} | {c_v6[2]:.4f} | {c_v6[3]:.4f} |",
        "",
        "说明：sanity 仅验证移植管线不崩、行对齐、复合与 v6 同量级。不回放选权/不据此定 Nov。",
        "",
    ]
    txt = "\n".join(lines)
    (REPLAY_DIR / "v15port_oct_sanity_report.txt").write_text(txt, encoding="utf-8")
    print(txt, flush=True)

    # ---------- serve：cut=11-01，全 06..10 池 final 训练 ----------
    pool_all = pd.concat([wide[m] for m in MONTHS], ignore_index=True)
    cond_all = pool_all.loc[pool_all["active"]].copy()
    print(f"[serve] final pool {len(pool_all):,} → active 过滤 {len(cond_all):,}；"
          f"LR 近2月 {MONTHS[-LR_LAST_N:]}", flush=True)
    models = train_learners(cond_all, MONTHS,
                            ["lgb_macro_eq", "xgb_macro_eq", "lr_eq_l2_sp"])

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
    print(f"[serve] 特征就绪 {len(Xs):,} 行；prior0={int((prior_serve==0).sum())} "
          f"prior1={int((prior_serve==1).sum())}", flush=True)

    comp_scores = {}
    for kind, m in models.items():
        s = np.zeros(len(df), dtype="float64")
        for flag, clf in m.items():
            mask = prior_serve == flag
            if mask.any():
                s[mask] = clf.predict_proba(Xs.loc[mask, FEATS])[:, 1].astype("float64")
        comp_scores[kind] = s
    port_score = fused_port(comp_scores)
    assert np.isfinite(port_score).all() and len(port_score) == len(raw)

    # v6 提交（同集秩）
    v6 = pd.read_csv(PROJECT_ROOT / "outputs" / "candidate"
                     / "sample_submission_cand-align-v6-fuse.csv")
    assert (v6["user_id"].astype(str).reset_index(drop=True)
            == raw["user_id"].reset_index(drop=True)).all()
    r6 = rankdata(v6["score"].to_numpy(float)) / len(raw)
    rp = rankdata(port_score) / len(raw)

    out = pd.DataFrame({"user_id": raw["user_id"], "item_id": raw["item_id"],
                        "score": port_score})
    out_path = PROJECT_ROOT / "outputs" / "candidate" / f"sample_submission_{NAME}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    meta = {"name": NAME,
            "desc": "v15port: prior(ui_owned)首购/复购双专家 + activity条件过滤训练 "
                    "(active==1行 eq权重) 移植进 v6 三族融合(LGB.1/XGB.3/LR近2月.6)；"
                    "跨专家对账=每分量全局百分位秩(吸收组长LR stacker且避组内常数)。"
                    "activity门控树不移植(p_activity秩不变已证伪)",
            "port_sources": ["leader V15_CODE_BUNDLE: fit_base/experts/conditional",
                             "our v6 scripts/47/48/49 recipe"],
            "mechanisms": {"prior_split_experts": "ui_owned==0 first_purchase / ==1 repurchase",
                           "activity_conditional": "train only rows whose user had >=1 tx that month",
                           "gate_tree": "NOT ported (rank-inert, p_activity comp ~.252)"},
            "weights": W, "train_months": MONTHS, "cut": str(SC_CUT.date()),
            "n_cond_train": int(len(cond_all)), "sanity_oct": c_port,
            "sanity_v6_oct_same_set": c_v6,
            "note": "OOF复合非Nov门禁(scripts/74反向)；Nov由板上裁决"}
    (PROJECT_ROOT / "models" / f"candidate_meta_{NAME}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {NAME} → {out_path.name} ({time.time()-t0:.0f}s)", flush=True)

    # ---------- v14 × v6 秩融合保险 ----------
    for w14 in (0.50, 0.75):
        score = w14 * rp + (1.0 - w14) * r6
        nm = f"{NAME}_m6_{int(w14*100):02d}"
        p = PROJECT_ROOT / "outputs" / "candidate" / f"sample_submission_{nm}.csv"
        pd.DataFrame({"user_id": raw["user_id"], "item_id": raw["item_id"], "score": score}
                     ).to_csv(p, index=False, encoding="utf-8")
        (PROJECT_ROOT / "models" / f"candidate_meta_{nm}.json").write_text(
            json.dumps({"name": nm, "desc": f"{w14}×v14(v15port) 秩 + {1-w14:.2f}×v6 秩 全局百分位融合",
                        "w_v14": w14, "base_board": {"v6": 0.7782, "leader_v15": 0.7845}},
                        ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[out] {nm} → {p.name}", flush=True)

    print(f"[done] 总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
