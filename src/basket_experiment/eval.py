"""组员路线评估工具：residual 尺度统一（组内 percentile）、alpha 扫描、救回/误踢、分桶。

final_score = anchor_score + alpha * resid_pct，其中 resid_pct = 残差分数在 (user,ym) 组内的
百分位（等秩均值），把不同用户/月份的量纲统一，避免 scale 差异（任务书第六节融合规则）。
alpha=0 ⇒ 恒等于 anchor（不归一/不截断/不改并列序）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import composite_score, monthly_composite

ANCHOR_COL = "blend_v15"


def group_percentile(df: pd.DataFrame, col: str) -> pd.Series:
    """组内 (user,ym) 等秩百分位 ∈ [0,1]；组大小 1 → 0（组内常数无排序意义）。"""
    g = df.groupby(["user_id", "ym"], sort=False)
    r = g[col].rank(method="average")
    n = df.groupby(["user_id", "ym"], sort=False)[col].transform("size").astype("float64")
    return ((r - 1.0) / (n - 1.0)).fillna(0.0)


def add_resid(df: pd.DataFrame, col: str, anchor: str = ANCHOR_COL) -> pd.DataFrame:
    """追加 resid_pct（组内百分位）与 anchor 等列；df 需含 user_id/ym/col/anchor。"""
    out = df.copy()
    out["resid_pct"] = group_percentile(out, col)
    return out


def final_col(df: pd.DataFrame, alpha: float, anchor: str = ANCHOR_COL) -> np.ndarray:
    return df[anchor].to_numpy(float) + float(alpha) * df["resid_pct"].to_numpy(float)


def sweep(df: pd.DataFrame, alphas, anchor: str = ANCHOR_COL, months=None) -> pd.DataFrame:
    """逐 alpha 综合分。months=None 用 df 全部行（组内口径自然覆盖全部 ym）。

    返回 DataFrame[alpha, composite, GAUC, NDCG@10, Recall@10]。
    """
    if months is not None:
        df = df[df["ym"].isin(set(months))]
    rows = []
    for a in alphas:
        cval, ga, nd, rc = composite_score(df.assign(_fin=final_col(df, a, anchor)), "_fin")
        rows.append({"alpha": a, "composite": cval, "GAUC": ga,
                     "NDCG@10": nd, "Recall@10": rc})
    return pd.DataFrame(rows)


def best_alpha_on(sweep_df: pd.DataFrame, month_col: bool = False) -> float:
    """扫描表在给定月份/整体上 composite argmax 的 alpha（跳过 alpha=0 取最优正 alpha 由调用者处理）。"""
    s = sweep_df
    return float(s.loc[s["composite"].idxmax(), "alpha"])


def bucket_topk_metrics(df: pd.DataFrame, alpha: float, topk: int = 10,
                        anchor: str = ANCHOR_COL) -> pd.DataFrame:
    """每 (user,ym) 组：anchor 与 final(alpha) 各自的 top-k 命中正例、救回/误踢。

    输出按行粒度附加标志列，返回统计 DataFrame：
      bucket(prior_bought 0/1), pos_total, anchor_hits, final_hits,
      saved(anchor 11..topk 正例进入 final topk), kicked(anchor topk 正例跌出), net.
    """
    d = df.copy()
    d["final"] = final_col(d, alpha, anchor)
    d["s_anchor"] = d.groupby(["user_id", "ym"])[anchor].rank(method="first", ascending=False)
    d["s_final"] = d.groupby(["user_id", "ym"])["final"].rank(method="first", ascending=False)
    d["is_pos"] = d["label"].astype(int) == 1
    d["a_top"] = d["s_anchor"] <= topk
    d["a_top20"] = (d["s_anchor"] <= topk * 3) & (d["s_anchor"] > topk)
    d["f_top"] = d["s_final"] <= topk
    d["saved"] = d["is_pos"] & d["a_top20"] & d["f_top"] & (~d["a_top"])
    d["kicked"] = d["is_pos"] & d["a_top"] & (~d["f_top"])
    stat = (d[d["is_pos"]].groupby("prior_bought").agg(
        pos_total=("label", "size"),
        anchor_top10_hits=("a_top", "sum"),
        final_top10_hits=("f_top", "sum"),
        saved_from_11_30=("saved", "sum"),
        kicked_from_top10=("kicked", "sum"),
    ).reset_index())
    stat["net_saved"] = stat["saved_from_11_30"] - stat["kicked_from_top10"]
    return stat


def repeat_explore_pos_hist(df: pd.DataFrame) -> pd.DataFrame:
    """Repeat(prior_bought>=1) / Explore(=0) 的行数与正例数分布。"""
    return (df.groupby("prior_bought")
            .agg(rows=("label", "size"), pos=("label", "sum"))
            .reset_index()
            .rename(columns={"prior_bought": "bucket"}))


def monthly_gain(df: pd.DataFrame, alpha: float, anchor: str = ANCHOR_COL,
                 months=None) -> pd.DataFrame:
    """逐月 anchor vs final(alpha) 的 composite/分量 + 行数/正例。"""
    out = []
    mm = months or sorted(df["ym"].unique())
    for ym in mm:
        sub = df[df["ym"] == ym]
        ca = composite_score(sub, anchor)
        cf = composite_score(sub.assign(_fin=final_col(sub, alpha, anchor)), "_fin")
        out.append({"ym": ym, "anchor_comp": ca[0], "final_comp": cf[0],
                    "d_comp": cf[0] - ca[0], "anchor_GAUC": ca[1], "final_GAUC": cf[1],
                    "anchor_NDCG": ca[2], "final_NDCG": cf[2],
                    "anchor_Recall": ca[3], "final_Recall": cf[3],
                    "rows": len(sub), "pos": int(sub["label"].sum())})
    return pd.DataFrame(out)
