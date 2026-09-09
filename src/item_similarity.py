"""相似商品计算模块（基于物品的协同过滤基础层）。

给定一个商品 StockCode，返回与该商品「购买画像」最相似的 Top-K 商品及余弦相似度。

建模方式：与 [[user_similarity]] 使用同一张「用户 × 商品」稀疏购买矩阵，
只是把物品当作「行向量」（每个商品的分量 = 各用户的购买数量）。
因此物品相似度 = 两个商品被「同一批用户」以相近数量购买的程度。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import COLUMNS
from src.user_similarity import load_clean_transactions

_USER = COLUMNS["user"]
_ITEM = COLUMNS["item"]
_QTY = COLUMNS["qty"]
_PRICE = COLUMNS["price"]
_INVOICE = COLUMNS["invoice"]
_DESC = COLUMNS["desc"]


def build_item_catalog(df: pd.DataFrame) -> dict[str, dict]:
    """汇总每个商品的展示信息：首个描述 / 中位单价 / 购买用户数 / 总销量 / 涉及订单数。

    Args:
        df: 交易数据（含 Quantity/UnitPrice/Description，需已清洗）。

    Returns:
        {StockCode: {desc, price, buyers, units, orders}}，按键即商品码。
    """
    d = df.copy()
    d[_DESC] = d[_DESC].fillna("").astype(str)

    def first_nonempty(s: pd.Series) -> str:
        for v in s:
            if v.strip():
                return v.strip()
        return ""

    g = d.groupby(_ITEM, sort=False)
    rows = {}
    for code, sub in g:
        rows[str(code)] = {
            "desc": first_nonempty(sub[_DESC]) or "(无描述)",
            "price": round(float(sub[_PRICE].median()), 2),
            "buyers": int(sub[_USER].nunique()),
            "units": int(sub[_QTY].sum()),
            "orders": int(sub[_INVOICE].nunique()),
        }
    return rows


def top_k_similar_rows(
    row_matrix: csr_matrix,
    labels: np.ndarray,
    top_k: int = 10,
    batch: int = 256,
) -> dict[str, list[tuple[str, float]]]:
    """对稀疏矩阵的每一行，批量求它与所有行的余弦相似度，保留 Top-K 邻居。

    通用实现：用户相似、商品相似共用。每行视为一个实体，行标签与行一一对应。

    Args:
        row_matrix: CSR 稀疏矩阵（行 = 待比较实体）。
        labels: 与行序一一对应的实体 ID（int 用户号或 str 商品码）。
        top_k: 每行保留的最多相似邻居数。
        batch: 一次同时处理的行数（控制中间稠密矩阵的内存）。

    Returns:
        {row_id: [(neighbor_id, similarity), ...]}，降序、排除自身、仅保留相似度 > 0。
    """
    n = row_matrix.shape[0]
    out: dict[str, list[tuple[str, float]]] = {}
    for start in range(0, n, batch):
        block = row_matrix[start : start + batch]
        sims = cosine_similarity(block, row_matrix)  # 稠密块: batch × n
        for i_local in range(sims.shape[0]):
            gi = start + i_local
            row_sims = sims[i_local]
            row_sims[gi] = -1.0  # 排除自身
            order = np.argsort(-row_sims)
            hits: list[tuple[str, float]] = []
            for pos in order:
                score = float(row_sims[pos])
                if score <= 0.0:
                    break
                hits.append((str(labels[pos]), round(score, 4)))
                if len(hits) >= top_k:
                    break
            out[str(labels[gi])] = hits
    return out


def load_item_matrix(
    df: pd.DataFrame | None = None,
) -> tuple[csr_matrix, np.ndarray]:
    """以「商品 × 用户」视角取矩阵（用户×商品转置），行 = 商品，列 = 用户。

    Returns:
        (item_user_matrix, item_ids): CSR 矩阵 + 与行序对应的商品码数组。
    """
    from src.user_similarity import load_purchase_matrix

    if df is None:
        df = load_clean_transactions()
    user_item, _, item_ids = load_purchase_matrix(df)  # 用户 × 商品
    return user_item.T.tocsr(), item_ids  # 商品 × 用户
