"""用户相似度计算模块（基于用户的协同过滤基础层）。

给定一个用户 ID，返回与该用户购买行为最相似的 Top-K 用户及余弦相似度。
相似度定义：以「用户 × 商品」稀疏矩阵的行向量做余弦相似度。
    - 默认按购买数量加权（Quantity 为用户向量的分量）；
    - binary=True 时退化为「是否购买过」（0/1），只关心品类重叠。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import COLUMNS, ENCODINGS, PATHS

_USER = COLUMNS["user"]
_ITEM = COLUMNS["item"]
_QTY = COLUMNS["qty"]


def load_clean_transactions() -> pd.DataFrame:
    """读取清洗后的交易数据（不存在则回退到原始数据并提示）。"""
    clean = PATHS["clean_transactions"]
    if clean.exists():
        return pd.read_csv(clean, encoding="utf-8")
    print("[提示] data/processed/clean_transactions.csv 不存在，回退到原始数据。")
    return pd.read_csv(PATHS["raw_transactions"], encoding=ENCODINGS["transactions"])


def load_purchase_matrix(
    df: pd.DataFrame | None = None, *, binary: bool = False
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """
    构建「用户 × 商品」稀疏购买矩阵。

    Args:
        df: 交易数据；None 时自动加载（优先清洗数据）。
        binary: True 时矩阵元素为 0/1（是否购买），否则为购买数量。

    Returns:
        (matrix, user_ids, item_ids): 稀疏矩阵、按行序排列的用户ID、按列序排列的商品ID。
    """
    if df is None:
        df = load_clean_transactions()

    d = df.copy()
    d = d[d[_USER].notna()]  # 防御：无归属用户的交互无法使用
    d = d[d[_QTY] > 0]       # 防御：只保留有效正购买

    if binary:
        # 是否购买模式：同一 (用户,商品) 跨订单只计一次
        d = d.drop_duplicates(subset=[_USER, _ITEM])
        values = np.ones(len(d), dtype=np.float64)
    else:
        # 数量加权模式：跨订单重复购买需要求和（coo 转 csr 时自动累加重复坐标）
        values = d[_QTY].astype(np.float64).to_numpy()

    d[_USER] = d[_USER].astype("int64")
    d[_ITEM] = d[_ITEM].astype(str)

    user_codes, user_ids = pd.factorize(d[_USER])
    item_codes, item_ids = pd.factorize(d[_ITEM])

    matrix = coo_matrix(
        (values, (user_codes, item_codes)),
        shape=(len(user_ids), len(item_ids)),
    ).tocsr()
    return matrix, user_ids.astype(np.int64), item_ids


def row_index_of(matrix: csr_matrix, user_ids: np.ndarray, user_id: int) -> int:
    """返回 user_id 在矩阵中的行号；不存在返回 -1。"""
    hits = np.where(user_ids == int(user_id))[0]
    return int(hits[0]) if len(hits) else -1


def find_similar_users(
    matrix: csr_matrix, user_ids: np.ndarray, user_id: int, top_k: int = 10
) -> list[dict]:
    """
    返回目标用户最相似的 Top-K 用户（相似度 > 0，降序）。

    Args:
        matrix: load_purchase_matrix 的稀疏矩阵。
        user_ids: 与矩阵行一一对应的用户ID。
        user_id: 目标用户。
        top_k: 返回个数上限。

    Returns:
        [{user_id, similarity}, ...]；目标用户不存在返回 []。
    """
    idx = row_index_of(matrix, user_ids, user_id)
    if idx < 0:
        return []

    sims = cosine_similarity(matrix[idx : idx + 1], matrix).ravel()
    sims[idx] = -1.0  # 排除自身

    results = []
    for pos in np.argsort(-sims):
        score = float(sims[pos])
        if score <= 0:
            break  # 已按降序，后面全部为 0
        results.append({"user_id": int(user_ids[pos]), "similarity": score})
        if len(results) >= top_k:
            break
    return results


def n_items_of(matrix: csr_matrix, user_ids: np.ndarray, user_id: int) -> int:
    """目标用户购买过的商品种类数；不存在返回 0。"""
    idx = row_index_of(matrix, user_ids, user_id)
    if idx < 0:
        return 0
    return int(matrix.getrow(idx).nnz)
