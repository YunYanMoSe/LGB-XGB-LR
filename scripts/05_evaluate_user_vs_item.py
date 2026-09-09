"""离线对比评测：基于用户的协同过滤 vs 基于物品的协同过滤。

协议：把每个用户的「最后一笔订单(按时间)」作为测试购物篮，其余历史作为训练。
分别用 UserCF 与 ItemCF 从训练历史中为测试篮补推荐 Top-N，衡量命中情况。
「全局热门」作为第三参照基线。

指标：recall@N / precision@N / hit@10。
运行：venv\\Scripts\\python.exe scripts\\05_evaluate_user_vs_item.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import COLUMNS
from src.user_similarity import load_clean_transactions

_USER, _ITEM = COLUMNS["user"], COLUMNS["item"]
_TIME, _QTY = COLUMNS["time"], COLUMNS["qty"]


def to_csr(sub: pd.DataFrame) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    s = sub.drop_duplicates([_USER, _ITEM])
    u_codes, u_ids = pd.factorize(s[_USER])
    i_codes, i_ids = pd.factorize(s[_ITEM])
    m = coo_matrix(
        (np.ones(len(s), dtype=np.float64), (u_codes, i_codes)),
        shape=(len(u_ids), len(i_ids)),
    ).tocsr()
    return m, u_ids.astype("int64"), i_ids


def main() -> None:
    df = load_clean_transactions()
    df = df.copy()
    df[_USER] = df[_USER].astype("int64")
    df[_ITEM] = df[_ITEM].astype(str)
    df[_TIME] = pd.to_datetime(df[_TIME])

    # 每个用户最后一笔订单 = 测试篮
    last_t = df.groupby(_USER)[_TIME].max().rename("last_t")
    df = df.merge(last_t, left_on=_USER, right_index=True)
    test = df[df[_TIME] >= df["last_t"]]
    train = df[df[_TIME] < df["last_t"]]

    B, train_users, train_items = to_csr(train)  # 用户×商品 0/1
    n_items = B.shape[1]
    code_of = {c: str(i) for c, i in enumerate(train_items)}
    col_of = {str(i): c for c, i in enumerate(train_items)}

    users = sorted(set(int(x) for x in train_users) & set(int(x) for x in test[_USER]))
    print(f"训练用户数 {B.shape[0]}  商品数 {n_items}  可分训练/测试的用户 {len(users)}")

    user_row = {int(u): r for r, u in enumerate(train_users)}
    own_codes: dict[int, set] = {int(u): set() for u in users}
    for u in users:
        own_codes[u] = {code_of[c] for c in B[user_row[u]].indices.tolist()}

    # 测试篮：{用户: 商品码集合}; known 为该篮中训练期见过的商品码
    test_map: dict[int, set] = {}
    known_map: dict[int, list[int]] = {}
    for uid, g in test.groupby(_USER):
        uid = int(uid)
        basket = {str(s) for s in set(g[_ITEM])}
        test_map[uid] = basket
        known_map[uid] = [col_of[x] for x in basket if x in col_of]

    # 全局热门（训练期被多少用户买，作基线）
    pop_cols = np.argsort(-np.asarray(B.sum(axis=0)).ravel()).tolist()
    pop_codes = [code_of[c] for c in pop_cols]

    # ---------- UserCF: 邻居 训练矩阵上的用户余弦 ----------
    rows = [user_row[u] for u in users]
    sim = cosine_similarity(B[rows], B)  # n_users × n_train_users
    nbr_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k, u in enumerate(users):
        s = sim[k].copy()
        s[rows[k]] = -1.0
        nbr = np.argsort(-s)[:50]
        nbr_cache[u] = (nbr, np.clip(s[nbr], 0, None))

    def rec_user(u: int) -> list[str]:
        nbr, w = nbr_cache[u]
        nbr, w = nbr[:20], w[:20]  # K=20 邻居投票
        cand: dict[str, float] = {}
        for nn, ww in zip(nbr, w):
            if ww <= 0:
                continue
            for c in B[nn].indices.tolist():
                code = code_of[c]
                if code not in own_codes[u]:
                    cand[code] = cand.get(code, 0.0) + ww
        return [c for c, _ in sorted(cand.items(), key=lambda kv: -kv[1])]

    # ---------- ItemCF: 相似商品投票（缓存 每测试商品 的 top200 相似列）----------
    ITEM = B.T.tocsr()
    sim_cache: dict[int, np.ndarray] = {}

    def item_sim_vec(col: int) -> np.ndarray:
        v = sim_cache.get(col)
        if v is None:
            row = ITEM[col]
            v = cosine_similarity(row, ITEM).ravel() if row.nnz else np.zeros(n_items)
            v[col] = 0.0
            sim_cache[col] = v
        return v

    def rec_item(u: int) -> list[str]:
        cand: dict[str, float] = {}
        for col in known_map[u]:
            v = item_sim_vec(col)
            for j in np.argsort(-v)[:200]:
                sc = float(v[j])
                if sc <= 0:
                    break
                code = code_of[j]
                if code not in own_codes[u]:
                    cand[code] = cand.get(code, 0.0) + sc
        return [c for c, _ in sorted(cand.items(), key=lambda kv: -kv[1])]

    # ---------- 评测 ----------
    N_list = [5, 10]

    def metrics(rec_fn) -> dict[str, float]:
        r5, p5, r10, p10, h10 = [], [], [], [], []
        for u in users:
            basket = test_map[u]
            topn = rec_fn(u)
            for N, (rl, pl) in ((5, (r5, p5)), (10, (r10, p10))):
                hits = len(set(topn[:N]) & basket)
                rl.append(hits / len(basket))
                pl.append(hits / N)
            h10.append(1.0 if len(set(topn[:10]) & basket) else 0.0)
        return {"recall@5": float(np.mean(r5)), "precision@5": float(np.mean(p5)),
                "recall@10": float(np.mean(r10)), "precision@10": float(np.mean(p10)),
                "hit@10": float(np.mean(h10))}

    def rec_pop(u: int) -> list[str]:
        return [c for c in pop_codes if c not in own_codes[u]]

    print("=" * 72)
    for name, fn in [("UserCF (数量余弦, 邻居K=20)", rec_user),
                     ("ItemCF (相似商品聚合投票)", rec_item),
                     ("全局热门 基线", rec_pop)]:
        res = metrics(fn)
        line = f"  {name:<28}" + "  ".join(f"{k}={v:.4f}" for k, v in res.items())
        print(line)
    print("=" * 72)


if __name__ == "__main__":
    main()
