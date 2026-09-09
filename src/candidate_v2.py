"""候选集打分 v2 特征块（加法式，独立于 v1 的 33 列路径）。

老师评分口径已被探针锁死 = 纯逐用户内部序 macro（跨用户比较无权重），
⇒ 用户级特征对榜分无效（组内恒定），真正能拉开候选行的是
「pair 级 / 商品近期 / 候选上下文-item 级」三类信号。本模块补齐：

    V2_FEATS = [
        # A. pair 历史（fp 内该 (u,i) 的强度与形态）—— 复购区分主力
        "p_ui_n_ord_log",        # log1p(该 pair 在 fp 的订单数)
        "p_ui_qty_log",          # log1p(该 pair 在 fp 的总件数)
        "p_ui_first_age_log",    # log1p(首次购买距今天数)
        # B. 商品近期窗口（截点前 30/60 日活跃）—— 帮助"新组合"里选品
        "i_r30_buy_log", "i_r30_ord_log", "i_r60_buy_log",
        "i_growth30_log",        # log1p(近30日买家 / 更早买家)，近期增长
        # C. 候选上下文（在当月候选宇宙 C 上统计）—— 商品候选频度 + 组内命中
        "cc_u_cand_size_log",    # log1p(该用户当月候选行数)
        "cc_i_cand_freq_log",    # log1p(该商品当月作为候选出现的行数)
        "cc_u_owned_ratio",      # 用户当月候选中 fp 已购占比（平滑）
        "cc_i_hit_ratio",        # 商品当月候选用户里 fp 已购该品的占比（平滑）
    ]

全部特征只使用 < cut 的历史（fp），候选上下文只在「当月候选宇宙 C」（训练=当月
回放行本身；打分=cut 时全量候选）上统计 —— 函数对训练/打分完全同一套定义。

用法：from src.candidate_v2 import v2_features, V2_FEATS
    extra = v2_features(clean, cut, cand_df)   # cand_df: user_id(int64)+item_id(str)，顺序与基础特征行一致
    extra.columns == V2_FEATS，len == len(cand_df)
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config.settings import COLUMNS

_USER = COLUMNS["user"]       # CustomerID
_ITEM = COLUMNS["item"]       # StockCode
_TIME = COLUMNS["time"]       # InvoiceDate
_QTY = COLUMNS["qty"]
_INVOICE = COLUMNS["invoice"]

V2_FEATS = [
    "p_ui_n_ord_log", "p_ui_qty_log", "p_ui_first_age_log",
    "i_r30_buy_log", "i_r30_ord_log", "i_r60_buy_log", "i_growth30_log",
    "cc_u_cand_size_log", "cc_i_cand_freq_log", "cc_u_owned_ratio", "cc_i_hit_ratio",
]
V2_DTYPES = {
    "p_ui_n_ord_log": np.float64, "p_ui_qty_log": np.float64, "p_ui_first_age_log": np.float64,
    "i_r30_buy_log": np.float64, "i_r30_ord_log": np.float64, "i_r60_buy_log": np.float64,
    "i_growth30_log": np.float64,
    "cc_u_cand_size_log": np.float64, "cc_i_cand_freq_log": np.float64,
    "cc_u_owned_ratio": np.float64, "cc_i_hit_ratio": np.float64,
}


def _log1p(x: pd.Series | np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(x, dtype="float64")).astype("float64")


def v2_features(clean: pd.DataFrame, cut: pd.Timestamp | str,
                cand_df: pd.DataFrame, smooth: float = 0.5) -> pd.DataFrame:
    """在 cut 处为 cand_df 每一行(user_id,item_id)计算 V2_FEATS。

    clean 需含 CustomerID/StockCode/InvoiceDate/Quantity/InvoiceNo（leader_clean 同款）。
    cand_df 列：user_id int64、item_id str；顺序/行数决定返回行序（用于按列追加）。
    smooth 用于命中占比的加性平滑（小样本方差）。
    """
    cut = pd.Timestamp(cut)
    fp = clean[clean[_TIME] < cut].copy()
    fp[_USER] = pd.to_numeric(fp[_USER], errors="coerce").astype("int64")
    fp[_ITEM] = fp[_ITEM].astype(str)
    fp[_TIME] = pd.to_datetime(fp[_TIME])

    N = len(cand_df)
    u_arr = cand_df["user_id"].to_numpy(dtype="int64")
    c_arr = cand_df["item_id"].astype(str).to_numpy(dtype=object)
    idx = pd.MultiIndex.from_arrays([u_arr, c_arr], names=["u", "c"])

    out = pd.DataFrame(index=pd.RangeIndex(N))
    days = (cut - fp[_TIME]).dt.days.to_numpy(dtype="float64")

    # ---- A. pair 历史（fp 内逐 (u,i)）----
    pair = fp.assign(_d=days).groupby([_USER, _ITEM], sort=True)
    p_ninv = pair[_INVOICE].nunique()
    p_qty = pair[_QTY].sum()
    p_first = pair["_d"].max()  # 距今最大 = 最早一次购买
    # 用 MultiIndex→reindex，保序、确定性；未购过的对 → NaN → fillna(0)（与 owned=0 语义一致）
    out["p_ui_n_ord_log"] = _log1p(p_ninv.reindex(idx).fillna(0.0).to_numpy())
    out["p_ui_qty_log"] = _log1p(p_qty.reindex(idx).fillna(0.0).to_numpy())
    out["p_ui_first_age_log"] = _log1p(p_first.reindex(idx).fillna(0.0).to_numpy())

    # ---- B. 商品近期窗口（截点前 30/60 日）----
    w30 = fp[days <= 30.0]
    w60 = fp[days <= 60.0]
    r30_buy = w30.groupby(_ITEM)[_USER].nunique()
    r30_ord = w30.groupby(_ITEM)[_INVOICE].nunique()
    r60_buy = w60.groupby(_ITEM)[_USER].nunique()
    life_buy = fp.groupby(_ITEM)[_USER].nunique()
    r30_buy_r = r30_buy.reindex(c_arr).fillna(0.0).to_numpy()
    r30_ord_r = r30_ord.reindex(c_arr).fillna(0.0).to_numpy()
    r60_buy_r = r60_buy.reindex(c_arr).fillna(0.0).to_numpy()
    life_r = life_buy.reindex(c_arr).fillna(0.0).to_numpy()
    older = np.maximum(life_r - r30_buy_r, 0.0)
    growth = r30_buy_r / np.maximum(older, 1.0)
    out["i_r30_buy_log"] = _log1p(r30_buy_r)
    out["i_r30_ord_log"] = _log1p(r30_ord_r)
    out["i_r60_buy_log"] = _log1p(r60_buy_r)
    out["i_growth30_log"] = _log1p(growth)

    # ---- C. 候选上下文（宇宙 = cand_df 自身行）----
    u_size = cand_df.groupby("user_id").size()
    i_freq = cand_df.groupby("item_id").size()
    out["cc_u_cand_size_log"] = _log1p(u_size.reindex(u_arr).to_numpy())
    out["cc_i_cand_freq_log"] = _log1p(i_freq.reindex(c_arr).to_numpy())

    # 用户组内 fp 已购集合
    owned_fp: dict[int, set[str]] = {}
    for u, gg in fp.groupby(_USER):
        owned_fp[int(u)] = set(gg[_ITEM].astype(str))
    owned_here = np.zeros(N, dtype="float64")
    u_own_cnt = np.zeros(N, dtype="float64")   # 组内已购候选行数（分母=u_size）
    u_size_arr = u_size.reindex(u_arr).fillna(0.0).to_numpy()
    for j, (u, c) in enumerate(zip(u_arr, c_arr)):
        s = owned_fp.get(int(u))
        if s is not None and c in s:
            owned_here[j] = 1.0
    # 每用户已购候选行数 = 对该用户候选行的 owned_here 求和
    df_local = pd.DataFrame({"u": u_arr, "o": owned_here})
    u_own = df_local.groupby("u")["o"].transform("sum").to_numpy()
    denom = np.maximum(u_size_arr, 1.0)
    out["cc_u_owned_ratio"] = ((u_own + smooth) / (denom + 2 * smooth)).astype("float64")

    # 商品组内：候选用户里 fp 已购过该品的比例
    buyers_of: dict[str, set[int]] = {}
    for i, gg in fp.groupby(_ITEM):
        buyers_of[str(i)] = set(int(x) for x in gg[_USER].unique())
    i_owned_hits = np.zeros(N, dtype="float64")
    for j, (u, c) in enumerate(zip(u_arr, c_arr)):
        b = buyers_of.get(c)
        if b is not None and int(u) in b:
            i_owned_hits[j] = 1.0
    i_freq_arr = i_freq.reindex(c_arr).fillna(0.0).to_numpy()
    df_local2 = pd.DataFrame({"c": c_arr, "o": i_owned_hits})
    i_hit = df_local2.groupby("c")["o"].transform("sum").to_numpy()
    denom_i = np.maximum(i_freq_arr, 1.0)
    out["cc_i_hit_ratio"] = ((i_hit + smooth) / (denom_i + 2 * smooth)).astype("float64")

    assert list(out.columns) == V2_FEATS and out.isna().sum().sum() == 0
    return out.astype(V2_DTYPES)
