"""候选集打分 v3 特征块（加法式，独立于 44 列路径；只依赖 clean + (cut, cand 行)）。

老师口径 = 0.4 GAUC + 0.4 NDCG@10 + 0.2 Recall@10，逐用户列表 ⇒ 新特征只做
pair 级 / item 级 / user×item 上下文（user 恒定列对组内排序无效，不做）。本块补 5 个
现有 44 列**没有**的机制：

    V3_FEATS = [
        # A. pair 复购周期（owned 且 ≥2 个购买日的对才有；补 p_ui_first_age/recency 缺的"周期"）
        "p_cycle_median_log",   # log1p(该 pair 相邻购买日间隔的中位数)——复购节奏
        "p_due_capped",         # min(距上次天数 / 周期中位数, 6)——"该复购了没"（>1=已过周期）
        "p_n_dates_log",        # log1p(该 pair 独立购买日数)——真复购次数（发票数口径的补强）
        # B. user × category 近窗动量（45d 内用户买过的、与候选同收拢类目的去重商品）
        "cc_cat_recent_log",    # log1p(同收拢类目近购去重商品数)
        "cc_cat_recent_share",  # (该数+.5)/(用户近45d去重商品数+1) 平滑占比
        # C. item 极短脉冲
        "i_r7_buy_log",         # log1p(近 7 天买过该商品的用户数)（补 i_r30/i_r60 缺的短脉冲）
    ]

全部只用 < cut 历史；对训练月回放行与打分 cut 完全同一套定义（同 v2 约定）。
per-user 组内排序只用这些能区分行间差异的信号。

用法：from src.candidate_v3 import v3_features, V3_FEATS
    extra = v3_features(clean, cut, cand_df)  # cand_df: user_id(int64)+item_id(str)，顺序与行一致
    extra.columns == V3_FEATS，len == len(cand_df)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import COLUMNS
from src import feature_engineering as fe

_USER = COLUMNS["user"]
_ITEM = COLUMNS["item"]
_TIME = COLUMNS["time"]

_REF = pd.Timestamp("2009-11-30")   # day-0 锚点（2010 全年日期转整数天）
_RECENT_DAYS = 45.0
_PULSE_DAYS = 7.0
_SMOOTH = 0.5

V3_FEATS = [
    "p_cycle_median_log", "p_due_capped", "p_n_dates_log",
    "cc_cat_recent_log", "cc_cat_recent_share",
    "i_r7_buy_log",
]
V3_DTYPES = {f: np.float64 for f in V3_FEATS}


def _log1p(x: pd.Series | np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(x, dtype="float64")).astype("float64")


def _collapse_of(fp: pd.DataFrame, codes) -> pd.Series:
    """给定特征期 fp，返回 code → 收拢类目 的映射（对齐 bundle 的 cat['collapse_of_item']）。"""
    category_of = fe._load_category_of()
    cat = fe.setup_categories(fp, category_of)
    need = np.unique(np.asarray(codes, dtype=object))
    raw = np.asarray([category_of.get(str(c), "") for c in need], dtype=object)
    coll = np.asarray([cat["collapse"](str(r)) for r in raw], dtype=object)
    return pd.Series(coll, index=need.astype(str))


def v3_features(clean: pd.DataFrame, cut: pd.Timestamp | str,
                cand_df: pd.DataFrame) -> pd.DataFrame:
    """在 cut 处为 cand_df 每一行 (user_id,item_id) 计算 V3_FEATS（只用 < cut 历史）。"""
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
    cut_day = int((cut - _REF).days)

    # ---- A. pair 复购周期（逐 (u,c) 独立购买日）----
    dp = fp[[_USER, _ITEM, _TIME]].drop_duplicates().copy()
    dp["day"] = (dp[_TIME] - _REF).dt.days.astype("int64")
    dp = dp.sort_values([_USER, _ITEM, "day"], kind="mergesort")
    gp = dp.groupby([_USER, _ITEM], sort=True)
    gap = gp["day"].diff().to_numpy(dtype="float64")       # 相邻购买日间隔（天）
    ps = dp.assign(_gap=gap).groupby([_USER, _ITEM], sort=True).agg(
        med=("_gap", "median"), n=("day", "size"), last=("day", "max"))
    ps_r = ps.reindex(idx)
    n_dates = ps_r["n"].fillna(0.0).to_numpy()
    med_gap = ps_r["med"].fillna(0.0).to_numpy()
    last_day = ps_r["last"].fillna(-1e9).to_numpy()
    cycle_ok = (n_dates >= 2) & (med_gap > 0)
    days_since = (cut_day - last_day).clip(min=0)
    due = np.where(cycle_ok, (days_since / med_gap).clip(0, 6), 0.0)
    out["p_cycle_median_log"] = np.where(cycle_ok, _log1p(med_gap), 0.0)
    out["p_due_capped"] = due.astype("float64")
    out["p_n_dates_log"] = _log1p(n_dates)

    # ---- C. item 短脉冲：近 7 天买家数 ----
    w7 = fp[(cut - fp[_TIME]).dt.days <= _PULSE_DAYS]
    buyers7 = w7.groupby(_ITEM)[_USER].nunique()
    out["i_r7_buy_log"] = _log1p(buyers7.reindex(c_arr).fillna(0.0).to_numpy())

    # ---- B. user × category 近窗动量（45d 内同收拢类目近购）----
    w45 = fp[(cut - fp[_TIME]).dt.days <= _RECENT_DAYS]
    rec_items = w45[[_USER, _ITEM]].drop_duplicates()
    need_codes = np.unique(np.concatenate([rec_items[_ITEM].to_numpy(dtype=object),
                                           c_arr.astype(dtype=object)]))
    coll_map = _collapse_of(fp, need_codes)   # code(str) → 收拢类目(str)
    rec_items = rec_items.assign(_b=coll_map.reindex(rec_items[_ITEM]).to_numpy(dtype=object))
    # 每用户近 45d 去重商品数
    u_tot = rec_items.groupby(_USER).size()
    # 每 (user, 收拢类目) 近购去重商品数
    cat_cnt = rec_items.groupby([_USER, "_b"]).size()
    cand_b = pd.Series(coll_map.reindex(c_arr.astype(str)).to_numpy(dtype=object))
    key = pd.MultiIndex.from_arrays([pd.Series(u_arr), cand_b], names=["u", "_b"])
    cnt = cat_cnt.reindex(key).fillna(0.0).to_numpy(dtype="float64")
    tot = u_tot.reindex(u_arr).fillna(0.0).to_numpy(dtype="float64")
    out["cc_cat_recent_log"] = _log1p(cnt)
    out["cc_cat_recent_share"] = ((cnt + _SMOOTH) / (tot + 2 * _SMOOTH)).astype("float64")

    out = out.reindex(columns=V3_FEATS)   # 计算序(B块末)≠V3_FEATS序(C块夹中间) → 归一
    assert list(out.columns) == V3_FEATS and out.isna().sum().sum() == 0
    return out.astype(V3_DTYPES)
