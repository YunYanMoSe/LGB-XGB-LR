"""候选集打分 v4 特征块（加法式，独立于 v1 33 列 / v2 11 列）。

目标 = 逐用户列表类指标（探针锁定 + gAUC_m 榜序实证）：能真正区分"同一用户候选列表里
买哪几件"的信号必须随商品或 pair 变化（用户级特征组内恒定无效）。组长 PDF 交互层/候选
上下文层已覆盖的 pair 历史、周期、ItemCF/Item2Vec 我们大多有；仍缺的是「pair 在月度尺度的
复购节奏」与「商品可消耗性」的显式刻画。本模块补 4 个（全部只 < cut 历史）：

    V4_FEATS = [
        "p_ui_act_mon_frac",     # pair 活跃月数 / 该用户活跃月数（fp 月序复购节奏，0~1）
        "p_ui_recent_ord30_log", # log1p(pair 在截点前 30 天内的订单数)—— 近期刚补货/在购
        "i_rep_incid_frac",      # 商品 fp 重复购买 incidence 占比（可消耗/常回购品，0~1）
        "i_r7_buy_log",          # log1p(商品 fp 最近 7 天购买用户数)—— 比 r30 更尖的近热
    ]

全部特征只使用 < cut 的 fp；对训练(回放行)与打分(cut 全量候选)同一套定义。列序/行数 =
cand_df。冷用户/冷商品一律 0（与 owned=0 语义一致）。

用法：from src.candidate_v4 import v4_features, V4_FEATS
    extra = v4_features(clean, cut, cand_df)
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

from config.settings import COLUMNS

_USER = COLUMNS["user"]
_ITEM = COLUMNS["item"]
_TIME = COLUMNS["time"]
_QTY = COLUMNS["qty"]
_INVOICE = COLUMNS["invoice"]

V4_FEATS = [
    "p_ui_act_mon_frac", "p_ui_recent_ord30_log",
    "i_rep_incid_frac", "i_r7_buy_log",
]
V4_DTYPES = {
    "p_ui_act_mon_frac": np.float64, "p_ui_recent_ord30_log": np.float64,
    "i_rep_incid_frac": np.float64, "i_r7_buy_log": np.float64,
}


def _log1p(x: pd.Series | np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(x, dtype="float64")).astype("float64")


def v4_features(clean: pd.DataFrame, cut: pd.Timestamp | str,
                cand_df: pd.DataFrame, smooth: float = 0.5) -> pd.DataFrame:
    """在 cut 处为 cand_df 每行 (user_id,item_id) 计算 V4_FEATS。"""
    cut = pd.Timestamp(cut)
    fp = clean[clean[_TIME] < cut].copy()
    fp[_USER] = pd.to_numeric(fp[_USER], errors="coerce").astype("int64")
    fp[_ITEM] = fp[_ITEM].astype(str)
    fp[_TIME] = pd.to_datetime(fp[_TIME])
    fp[_INVOICE] = fp[_INVOICE].astype(str)

    N = len(cand_df)
    u_arr = cand_df["user_id"].to_numpy(dtype="int64")
    c_arr = cand_df["item_id"].astype(str).to_numpy(dtype=object)
    idx = pd.MultiIndex.from_arrays([u_arr, c_arr], names=["u", "c"])
    days = (cut - fp[_TIME]).dt.days.to_numpy(dtype="float64")
    MON = "_mon"
    fp = fp.assign(_d=days, _mon=fp[_TIME].dt.strftime("%Y-%m"))

    out = pd.DataFrame(index=pd.RangeIndex(N))

    # ---- 1. pair 月度活跃节奏：pair 活跃月数 / 用户活跃月数 ----
    u_mon = fp.groupby(_USER)[MON].nunique()  # 每用户活跃月数
    p_mon = fp.groupby([_USER, _ITEM])[MON].nunique()   # 每 pair 活跃月数
    u_mon_r = u_mon.reindex(u_arr).fillna(0.0).to_numpy(dtype="float64")
    p_mon_r = p_mon.reindex(idx).fillna(0.0).to_numpy(dtype="float64")
    denom = np.maximum(u_mon_r, 1.0)
    out["p_ui_act_mon_frac"] = (p_mon_r / denom).astype("float64")

    # ---- 2. pair 近 30 天订单数 ----
    w30 = fp[fp["_d"] <= 30.0]
    p30 = w30.groupby([_USER, _ITEM])[_INVOICE].nunique()
    out["p_ui_recent_ord30_log"] = _log1p(p30.reindex(idx).fillna(0.0))

    # ---- 3. 商品 fp 重复购买 incidence 占比 ----
    # 每 (user,item) 去重发票后的事件：n_inv = 该用户买该品的发票数；repeat 次数 = max(n-1,0)。
    ev = fp.groupby([_USER, _ITEM])[_INVOICE].nunique().rename("n_inv").reset_index()
    ev["repeat"] = np.maximum(ev["n_inv"] - 1, 0).astype("float64")
    rep_sum = ev.groupby(_ITEM)["repeat"].sum()
    n_trip = ev.groupby(_ITEM)["n_inv"].sum()
    frac = rep_sum / n_trip.replace(0.0, np.nan)
    out["i_rep_incid_frac"] = frac.reindex(c_arr).fillna(0.0).to_numpy(dtype="float64")

    # ---- 4. 商品最近 7 天购买用户数 ----
    w7 = fp[fp["_d"] <= 7.0]
    r7 = w7.groupby(_ITEM)[_USER].nunique()
    out["i_r7_buy_log"] = _log1p(r7.reindex(c_arr).fillna(0.0))

    assert list(out.columns) == V4_FEATS and out.isna().sum().sum() == 0
    return out.astype(V4_DTYPES)
