"""统一评分器：与 scripts/74_leader_oof_composite.py / scripts/59 逐函数一致。

口径：组 = (user_id, ym)；GAUC/NDCG@10/Recall@10 组内计算，macro 等权；单类组 NaN 跳过。
综合分 = 0.40*GAUC + 0.40*NDCG@10 + 0.20*Recall@10。

本模块的函数体与 scripts/74 保持一致（新实现不得与老师口径漂移）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def group_meta(df: pd.DataFrame):
    """每组 (user,ym) 的 g 编码、n、p、label；返回 (g, n_g, p_g, lbl)。"""
    g, _ = pd.factorize(df["user_id"].astype(str) + "_" + df["ym"].astype(str))
    lbl = df["label"].to_numpy(dtype="int64")
    n_g = np.bincount(g)
    p_g = np.bincount(g, weights=lbl.astype("float64"))
    return g, n_g, p_g, lbl


def group_metrics(s: np.ndarray, g: np.ndarray, n_g: np.ndarray, p_g: np.ndarray,
                  lbl: np.ndarray):
    """逐组 AUC / NDCG@10 / Recall@10（rank average 算 AUC，rank first desc 算 top-10）。"""
    sub = pd.DataFrame({"g": g, "s": np.asarray(s, dtype="float64"), "lbl": lbl})
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
    return auc_g, ndcg_g, rec_g


def macro_mean(m: np.ndarray) -> float:
    m = m[~np.isnan(m)]
    return float(m.mean()) if len(m) else float("nan")


def comp(s, g, n_g, p_g, lbl):
    """全量综合分 → (composite, GAUC, NDCG@10, Recall@10)。"""
    a, d, r = group_metrics(s, g, n_g, p_g, lbl)
    GAUC, NDCG, REC = macro_mean(a), macro_mean(d), macro_mean(r)
    return 0.4 * GAUC + 0.4 * NDCG + 0.2 * REC, GAUC, NDCG, REC


def composite_score(df: pd.DataFrame, score_col: str):
    """df 需含 user_id / ym / label / score_col → 全量综合分。"""
    g, n_g, p_g, lbl = group_meta(df)
    return comp(df[score_col].to_numpy(dtype="float64"), g, n_g, p_g, lbl)


def monthly_composite(df: pd.DataFrame, score_col: str):
    """逐 ym 综合分 → DataFrame[ym, composite, GAUC, NDCG@10, Recall@10, rows, users, pos]。

    逐月指标 = 该月内按 (user,ym) 组（等价按 user）macro。组与整体口径完全一致。
    """
    out = []
    for ym, sub in df.groupby("ym", sort=True):
        g, n_g, p_g, lbl = group_meta(sub)
        cval, ga, nd, rc = comp(sub[score_col].to_numpy(dtype="float64"), g, n_g, p_g, lbl)
        out.append({"ym": ym, "composite": cval, "GAUC": ga, "NDCG@10": nd,
                    "Recall@10": rc, "rows": len(sub), "users": int(pd.Series(sub["user_id"].unique()).shape[0]),
                    "pos": int(lbl.sum())})
    return pd.DataFrame(out)
