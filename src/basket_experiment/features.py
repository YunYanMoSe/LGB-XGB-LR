"""Basket/Hypergraph 路线的篮构建与共现相似度。

约定（节点1 互审项）：
  * 篮 = (CustomerID, InvoiceNo)，同一篮内商品去重；
  * 服务码（POST/DOT/... 见 cand.SERVICE_CODES）不入篮；
  * 只用交易时间 < cutoff 的篮（时间安全），cutoff = snapshot 月 1 日；
  * 批发大篮（中位 15、最大 250）⇒ 默认按 1/sqrt(|B|-1) 对篮子内部共现加权，防止 O(m²) 支配。

相似度主方案（H1）：对带权篮子频次矩阵 W（item×basket），C = W Wᵀ，余弦归一 → S ∈ [0,1]。
对照方案：cos_raw（0/1 余弦，无大篮归一）、jaccard、cond（条件共现 P(j|i)）、time-decay。

H2（线性超图）复用本模块的 basket 存储，仅换相似度矩阵来源。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import sparse as sp

from . import io_data as io

SERVICE_CODES = {
    "POST", "DOT", "M", "S", "B", "PADS", "C2", "D",
    "BANK CHARGES", "ADJUST", "ADJUST2", "TEST001", "TEST002",
}


def build_item_vocab(clean: pd.DataFrame, extra_items: pd.Series | None = None) -> tuple[np.ndarray, dict]:
    """商品词汇表：leader 全部商品（除服务码）∪ extra；排序保证确定。"""
    base = set(clean[io._COL_ITEM].unique()) - SERVICE_CODES
    if extra_items is not None:
        base |= set(extra_items.astype(str).unique())
    arr = np.array(sorted(base), dtype=object)
    mapping = {str(x): i for i, x in enumerate(arr)}
    return arr, mapping


@dataclass
class BasketStore:
    """篮子存储：(user,invoice) 去重、篮内去重后的篮集合。"""
    users: np.ndarray            # int64 per basket
    times: np.ndarray            # datetime64[ns] per basket
    sizes: np.ndarray            # int per basket（去重后 item 数）
    indptr: np.ndarray           # offsets into items_flat
    items_flat: np.ndarray       # item idx flat
    order: np.ndarray = field(default=None)  # basket-idx sorted by time

    def basket_ids_before(self, cutoff: pd.Timestamp) -> np.ndarray:
        if self.order is None:
            self.order = np.argsort(self.times, kind="mergesort")
        t = self.times[self.order]
        return self.order[t < np.datetime64(cutoff)]


def build_baskets(clean: pd.DataFrame, mapping: dict) -> BasketStore:
    """(user, invoice) 去重且篮内商品去重 → BasketStore（篮时间 = 最早行时间）。"""
    df = clean[[io._COL_USER, io._COL_INVOICE, io._COL_ITEM, io._COL_TIME]].copy()
    df[io._COL_ITEM] = df[io._COL_ITEM].astype(str)
    df = df[~df[io._COL_ITEM].isin(SERVICE_CODES)]
    df = df[df[io._COL_ITEM].map(mapping).notna()]
    df = df.drop_duplicates([io._COL_USER, io._COL_INVOICE, io._COL_ITEM])
    if len(df) == 0:
        return BasketStore(np.empty(0, "int64"), np.empty(0, "datetime64[ns]"),
                           np.empty(0, "int64"), np.array([0], "int64"),
                           np.empty(0, "int64"))

    users, times, sizes, items = [], [], [], []
    indptr = [0]
    # groupby 单遍迭代（total ~ O(n)），比 per-group get_group 快
    for (u, _inv), sub in df.groupby([io._COL_USER, io._COL_INVOICE], sort=False):
        sub = sub.sort_values(io._COL_TIME)
        idxs = np.fromiter((mapping[str(x)] for x in sub[io._COL_ITEM]), dtype="int64")
        users.append(int(u))
        times.append(sub[io._COL_TIME].iloc[0].to_datetime64())
        sizes.append(len(idxs))
        indptr.append(indptr[-1] + len(idxs))
        items.append(idxs)
    return BasketStore(
        users=np.asarray(users, dtype="int64"),
        times=np.asarray(times, dtype="datetime64[ns]"),
        sizes=np.asarray(sizes, dtype="int64"),
        indptr=np.asarray(indptr, dtype="int64"),
        items_flat=np.concatenate(items) if items else np.empty(0, dtype="int64"),
    )


def build_weighted_incidence(store: BasketStore, basket_ids: np.ndarray,
                             big_basket: bool | str = True,
                             decay_tau: float | None = None,
                             cutoff: pd.Timestamp | None = None,
                             n_items: int = 0) -> sp.csr_matrix:
    """W（n_items × len(basket_ids)）带权频次矩阵。

    big_basket ∈ {False/'plain', True/'sqrtBm1', 'sqrtB'}:
      * plain / False      → 每商品权重 1
      * 'sqrtBm1'（H1 主） → 1/sqrt(|B|-1)（篮内两两配对贡献均摊）
      * 'sqrtB'（H2 线性超图）→ 1/sqrt(|B|)（篮度归一的消息传递）
    decay_tau 给定时按 exp(-Δt/τ) 时间衰减（Δt 天）。"""
    if len(basket_ids) == 0:
        return sp.csr_matrix((n_items, 0))
    mode = "sqrtBm1" if big_basket is True else (big_basket or "plain")
    # 单 basket 权重是标量 ⇒ 整行向量化即可
    w = np.ones(len(basket_ids), dtype="float64")
    bs = store.sizes[basket_ids].astype("float64")
    if mode == "sqrtBm1":
        m = bs > 1
        w[m] = 1.0 / np.sqrt(bs[m] - 1.0)
    elif mode == "sqrtB":
        w[bs > 0] = 1.0 / np.sqrt(bs[bs > 0])
    if decay_tau is not None and cutoff is not None:
        dt = (np.datetime64(cutoff) - store.times[basket_ids]) / np.timedelta64(1, "D")
        w *= np.exp(-dt.astype("float64") / decay_tau)
    nnz = int(store.sizes[basket_ids].sum())
    rows = np.empty(nnz, dtype="int64")
    cols = np.empty(nnz, dtype="int64")
    vals = np.empty(nnz, dtype="float64")
    ptr = 0
    for col, bid in enumerate(basket_ids):
        a, b = store.indptr[bid], store.indptr[bid + 1]
        k = b - a
        rows[ptr:ptr + k] = store.items_flat[a:b]
        cols[ptr:ptr + k] = col
        vals[ptr:ptr + k] = w[col]
        ptr += k
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_items, len(basket_ids)))


def cos_from_C(C: np.ndarray) -> np.ndarray:
    n = C.shape[0]
    d = np.sqrt(np.maximum(C.diagonal(), 0.0))
    denom = np.outer(d, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        S = C / denom
    mask = d > 0
    S = S * mask[None, :] * mask[:, None]
    S[~np.isfinite(S)] = 0.0
    np.fill_diagonal(S, mask.astype("float64"))
    return S


def jaccard_from_C(C: np.ndarray) -> np.ndarray:
    diag = C.diagonal().copy()
    nn = diag[:, None] + diag[None, :] - C
    S = np.zeros_like(C)
    nz = nn > 0
    S[nz] = C[nz] / nn[nz]
    np.fill_diagonal(S, 1.0)
    return S


def build_sim_matrix(store: BasketStore, n_items: int, cutoff: pd.Timestamp,
                     scheme: str = "cos_bigbasket") -> np.ndarray:
    """cutoff 下 item×item 相似度 S（float32 稠密）。

    scheme ∈ {cos_bigbasket(主), cos_raw, jaccard, cond, cos_bigbasket_t90, cos_bigbasket_t30}
    cond = 条件共现 P(j|i)=C_ij/C_ii（行归一、非对称），供 H2 对照。
    """
    bid_ids = store.basket_ids_before(cutoff)
    decay = None
    for tag, tau in (("_t90", 90.0), ("_t30", 30.0)):
        if tag in scheme:
            decay = tau
    if scheme == "cos_raw":
        mode: bool | str = "plain"
    elif scheme == "hg_linear":
        mode = "sqrtB"           # H2 线性超图：篮度归一消息传递
    else:
        mode = "sqrtBm1"         # cos_bigbasket / cond / *_t30/_t90 默认大篮归一
    W = build_weighted_incidence(store, bid_ids, big_basket=mode, decay_tau=decay,
                                 cutoff=cutoff, n_items=n_items)
    if scheme == "jaccard":
        W0 = build_weighted_incidence(store, bid_ids, big_basket="plain", n_items=n_items)
        S = jaccard_from_C((W0 @ W0.T).toarray().astype("float64"))
    else:
        C = (W @ W.T).toarray().astype("float64")
        if scheme == "cond":
            diag = C.diagonal().copy()
            with np.errstate(divide="ignore", invalid="ignore"):
                S = C / diag[:, None]
            S[~np.isfinite(S)] = 0.0
            S = np.clip(S, 0.0, 1.0)
            S[diag == 0, :] = 0.0
            S[diag == 0] = 0.0
        else:
            S = cos_from_C(C)
    return S.astype("float32")


def all_baskets_per_user(store: BasketStore, users_needed: np.ndarray,
                         cutoff: pd.Timestamp) -> dict[int, list[np.ndarray]]:
    """{user: 全部 < cutoff 的篮 item 数组（时间升序）}，仅含 users_needed。"""
    need = set(int(u) for u in users_needed)
    bid_ids = store.basket_ids_before(cutoff)
    out: dict[int, list[np.ndarray]] = {}
    for bid in bid_ids:
        u = int(store.users[bid])
        if u in need:
            a, b = store.indptr[bid], store.indptr[bid + 1]
            out.setdefault(u, []).append(store.items_flat[a:b])
    return out


def month_features(S: np.ndarray, oof: pd.DataFrame, mapping: dict,
                   baskets: dict[int, list[np.ndarray]],
                   K: int = 3) -> pd.DataFrame:
    """为同一个月 anchor OOF 行计算篮共现特征（向量化：按 user 逐篮运算）。

    返回列（按 oof 行序对齐，float64）：
      sim_last1      : 候选 item 与最近 1 篮的最大相似度
      sim_last3_max  : 最近 K(=3) 篮逐篮 max 的最大
      sim_last3_mean : 最近 K 篮逐篮 max 的均值
      nb_item_baskets: 截止前用户所有篮中含该 item 的篮数（≈ v15 已有频次，重复信号，仅对照）
      has_basket     : 用户截止前是否有任意篮
    """
    users = oof["user_id"].to_numpy(dtype="int64")
    items = oof["item_id"].astype(str).to_numpy()
    n = len(oof)
    s1 = np.zeros(n, dtype="float64")
    s3max = np.zeros(n, dtype="float64")
    s3mean = np.zeros(n, dtype="float64")
    sn1 = np.zeros(n, dtype="float64")      # self-exclusive（共购邻居）
    sn3max = np.zeros(n, dtype="float64")
    sn3mean = np.zeros(n, dtype="float64")
    nb = np.zeros(n, dtype="float64")
    hasb = np.zeros(n, dtype="float64")

    # 每 user：候选商品去重 → 向量化逐篮特征 → 展开回行
    for u in np.unique(users):
        rows = np.flatnonzero(users == u)
        bl = baskets.get(int(u))
        if not bl:
            continue
        hasb[rows] = 1.0
        uniq_item, inv = np.unique(items[rows], return_inverse=True)
        gi = np.array([mapping.get(str(x), -1) for x in uniq_item], dtype="int64")
        ok = gi >= 0
        m_uniq = len(uniq_item)
        n_bl = len(bl)
        simmax_all = np.zeros((m_uniq, n_bl), dtype="float32")
        simnb_all = np.zeros((m_uniq, n_bl), dtype="float32")
        memb_all = np.zeros((m_uniq, n_bl), dtype="bool")
        if ok.any():
            gi_ok = gi[ok]
            S_row = S[gi_ok]                 # (m_ok, n_items)
            pos = np.flatnonzero(ok)
            for bi, b in enumerate(bl):
                if len(b) == 0:
                    continue
                S_b = S_row[:, b]            # (m_ok, |b|)
                simmax_all[pos, bi] = S_b.max(axis=1)
                memb = np.isin(gi_ok, b)     # (m_ok,)
                memb_all[pos, bi] = memb
                excl = S_b.max(axis=1).copy()
                # 对每个在篮中的候选商品：去掉自身列再取 max（自相似=1 会掩盖共购邻居）
                for p in np.flatnonzero(memb):
                    g = gi_ok[p]
                    col = np.flatnonzero(b == g)[0]
                    rv = S_b[p].copy()
                    rv[col] = -1.0
                    excl[p] = rv.max()
                simnb_all[pos, bi] = excl
        # 最近篮聚合（bl 升序 ⇒ 末尾最新）
        s1_u = simmax_all[:, -1]
        sn1_u = simnb_all[:, -1]
        lastK = simmax_all[:, -K:] if n_bl >= K else simmax_all
        lastKnb = simnb_all[:, -K:] if n_bl >= K else simnb_all
        s3max_u, s3mean_u = lastK.max(axis=1), lastK.mean(axis=1)
        sn3max_u, sn3mean_u = lastKnb.max(axis=1), lastKnb.mean(axis=1)
        nb_u = memb_all.sum(axis=1).astype("float64")
        s1[rows] = s1_u[inv]; s3max[rows] = s3max_u[inv]; s3mean[rows] = s3mean_u[inv]
        sn1[rows] = sn1_u[inv]; sn3max[rows] = sn3max_u[inv]; sn3mean[rows] = sn3mean_u[inv]
        nb[rows] = nb_u[inv]
    return pd.DataFrame({
        "sim_last1": s1, "sim_last3_max": s3max, "sim_last3_mean": s3mean,
        "simnb_last1": sn1, "simnb_last3_max": sn3max, "simnb_last3_mean": sn3mean,
        "nb_item_baskets": nb, "has_basket": hasb,
    })
