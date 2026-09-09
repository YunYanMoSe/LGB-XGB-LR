"""阶段六链路共用：召回 / 特征 / 精排 / 在线推荐的公共实现。

把 05/08 中散在评测脚本里的逻辑收敛为可复用的 API，供新脚本
(scripts/10~13) 与 Flask web/app.py 共用；既有 01~09 不改动。

防泄漏约定：离线评估一律用「每用户最后一笔订单=测试篮」协议（与 05/08 相同），
所有特征（热度/向量/价格/ItemCF）只由训练期(train, 去掉最后一单)数据计算；
网页/演示 API 用全量数据 ctx（商品语义相似属静态知识，无用户评估泄漏）。

本模块刻意做「上下文在调用方手里、函数收 ctx 参数」的纯函数设计，
避免把大矩阵当全局变量；脚本负责把 train 或 full 数据建成 RecCtx。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.sparse import coo_matrix, csr_matrix

from config.settings import COLUMNS, HYPERPARAMS, PATHS

_USER, _ITEM, _TIME = COLUMNS["user"], COLUMNS["item"], COLUMNS["time"]
_QTY, _PRICE = COLUMNS["qty"], COLUMNS["price"]

# 精排特征（顺序即模型特征列顺序，改动必须同步 rank_meta.json）
FEATURES = [
    "pop_buyers_log",    # 训练期购买该商品的用户数(买家数) log1p
    "pop_rank_frac",     # 热门排名位置分数 0=最热门 .. 1=最冷门
    "price_log",         # 训练期单价中位数 log1p
    "emb_sim_uservec",   # 用户历史商品向量的均值 与该商品向量 余弦
    "cf_agg_sim",        # 与用户已购商品集合的 ItemCF 聚合余弦(同 05 rec_item)
    "n_routes",          # 该候选被几路召回命中(0~3)
    "fl_pop",            # 是否热门路召回
    "fl_sim",            # 是否相似扩展路召回
    "fl_vec",            # 是否用户向量路召回
    "user_n_owned_log",  # 用户训练期购买去重商品数 log1p
]

# 召回配额（与 config.settings.HYPERPARAMS["recall"] 保持一致）
_FINAL_K = HYPERPARAMS["recall"]["final_k"]
_QUOTAS = dict(HYPERPARAMS["recall"]["quotas"])


def load_clean(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """读取清洗后交易并统一 dtype / 时间列（CustomerID 避免 17850.0 问题）。"""
    if df is None:
        df = pd.read_csv(PATHS["clean_transactions"], encoding="utf-8")
    df = df.copy()
    df[_USER] = pd.to_numeric(df[_USER], errors="coerce").astype("int64")
    df[_ITEM] = df[_ITEM].astype(str)
    df[_TIME] = pd.to_datetime(df[_TIME])
    return df


def load_products_zh() -> dict[str, dict]:
    """products.csv 的中文描述/类目映射 {code: {desc_zh, category}}。"""
    p = PATHS["raw_products"]
    if not p.exists():
        return {}
    raw = pd.read_csv(p, encoding="utf-8-sig", dtype=str)
    raw["StockCode"] = raw["StockCode"].astype(str)
    out: dict[str, dict] = {}
    zh = "Chinese_Description" if "Chinese_Description" in raw.columns else None
    cat = "Product_Category" if "Product_Category" in raw.columns else None
    for _, r in raw.iterrows():
        out[r["StockCode"]] = {
            "desc_zh": (r[zh] if zh and pd.notna(r[zh]) else ""),
            "category": (r[cat] if cat and pd.notna(r[cat]) else ""),
        }
    return out


def split_last_invoice(df: pd.DataFrame):
    """05/08 同款留出：每用户最后一次时间=测试，更早=训练。返回 (train, test)。"""
    last_t = df.groupby(_USER)[_TIME].max()
    df = df.merge(last_t.rename("_last_t"), left_on=_USER, right_index=True)
    test = df[df[_TIME] >= df["_last_t"]].drop(columns="_last_t")
    train = df[df[_TIME] < df["_last_t"]].drop(columns="_last_t")
    return train.copy(), test.copy()


def binary_csr(df: pd.DataFrame) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """(用户,商品) 去重为 0/1 的 user×item CSR；返回 (B, users, item_codes)。"""
    s = df.drop_duplicates([_USER, _ITEM])
    u_codes, u_ids = pd.factorize(s[_USER].astype("int64"))
    i_codes, i_ids = pd.factorize(s[_ITEM].astype(str))
    m = coo_matrix((np.ones(len(s), dtype=np.float32), (u_codes, i_codes)),
                   shape=(len(u_ids), len(i_ids))).tocsr()
    return m, u_codes.astype(np.int64), i_ids


@dataclass
class RecCtx:
    """一次评测/一次服务共用的上下文（由调用方从 train 或 full 数据构建）。"""
    B: csr_matrix                    # user × item (0/1)
    users: np.ndarray                # int64 用户号（行序对齐 B 行）
    item_codes: np.ndarray           # str 商品号（列序对齐 B 列）
    col_of: dict = field(default_factory=dict)
    code_of: dict = field(default_factory=dict)
    buyers: np.ndarray | None = None        # 每列买家数 log1p
    pop_rank: np.ndarray | None = None      # 热门位置 0=最热 .. 1=最冷
    price_log: np.ndarray | None = None     # 单价中位数 log1p
    item_unit: np.ndarray | None = None     # item×user 行归一化稠密 (n_items, n_users)
    # —— 商品向量（离线=train-only 重训；在线=全量模型）——
    embed_keys: np.ndarray | None = None
    embed_row: dict = field(default_factory=dict)
    embed_unit: np.ndarray | None = None    # (vocab, dim) 行归一化
    recent_items: dict = field(default_factory=dict)  # user -> 训练期最后一单商品(去重)
    user_row: dict = field(default_factory=dict)

    # 惰性缓存
    _owned_cols: dict = field(default_factory=dict)
    _owned_prof: dict = field(default_factory=dict)
    _emb_prof: dict = field(default_factory=dict)


def build_ctx(train_df: pd.DataFrame,
              embed_keys, embed_unit,
              *,
              build_item_unit: bool = True) -> RecCtx:
    """由训练期 df 与商品向量建 ctx。价格中位数只用该 df（防未来泄漏）。

    用户行序 = 排序后的唯一用户（升序），与 binary_csr 行号一致。
    """
    s = train_df.drop_duplicates([_USER, _ITEM])[[_USER, _ITEM]]
    s = s.sort_values([_USER, _ITEM]).reset_index(drop=True)
    u_codes, u_ids = pd.factorize(s[_USER].astype("int64"))
    i_codes, i_ids = pd.factorize(s[_ITEM].astype(str))
    B = coo_matrix((np.ones(len(s), dtype=np.float32), (u_codes, i_codes)),
                   shape=(len(u_ids), len(i_ids))).tocsr()

    col_of = {c: i for i, c in enumerate(i_ids)}
    code_of = {i: c for i, c in enumerate(i_ids)}
    buyers_cnt = np.asarray(B.sum(axis=0)).ravel()
    log_buyers = np.log1p(buyers_cnt)
    pos = np.argsort(np.argsort(-buyers_cnt))          # 0=最热门
    pop_rank = pos / max(1, len(pos) - 1)
    price = train_df.groupby(_ITEM)[_PRICE].median().reindex(i_ids).fillna(0.0)
    price_log = np.log1p(np.clip(price.to_numpy(dtype=np.float64), 0, None))

    item_unit = None
    if build_item_unit:
        MI = B.T.tocsr()
        dense = MI.toarray().astype(np.float32)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        item_unit = (dense / norms).astype(np.float32)

    keys = np.asarray([str(k) for k in embed_keys])
    unit = np.asarray(embed_unit, dtype=np.float32)
    unit = unit / np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
    embed_row = {k: i for i, k in enumerate(keys)}

    # 每用户「训练期最后一单」商品集（相似扩展路的输入）
    recent: dict[int, set[str]] = {}
    if len(train_df):
        last = train_df.sort_values(_TIME).groupby(_USER)[_ITEM].tail(1)
        # 更严谨：取用户时间戳最大的那一整张单
        tmax = train_df.groupby(_USER)[_TIME].transform("max")
        rows_last = train_df[train_df[_TIME] == tmax]
        for u, g in rows_last.groupby(_USER):
            recent[int(u)] = set(g[_ITEM].astype(str))

    return RecCtx(
        B=B, users=u_ids.astype("int64"), item_codes=i_ids,
        col_of=col_of, code_of=code_of,
        buyers=log_buyers.astype(np.float32), pop_rank=pop_rank.astype(np.float32),
        price_log=price_log.astype(np.float32),
        item_unit=item_unit,
        embed_keys=keys, embed_row=embed_row, embed_unit=unit,
        recent_items=recent,
        user_row={int(u): r for r, u in enumerate(u_ids)},
    )


def owned_cols(ctx: RecCtx, u: int) -> np.ndarray:
    if u not in ctx._owned_cols:
        r = ctx.user_row.get(u)
        ctx._owned_cols[u] = (ctx.B.indices[ctx.B.indptr[r]:ctx.B.indptr[r + 1]]
                              if r is not None else np.array([], dtype=int))
    return ctx._owned_cols[u]


def owned_prof(ctx: RecCtx, u: int) -> np.ndarray:
    """用户已购商品的 ItemCF 聚合画像（item×user 行单位求和）。"""
    if u not in ctx._owned_prof:
        cols = owned_cols(ctx, u)
        if ctx.item_unit is None or len(cols) == 0:
            ctx._owned_prof[u] = np.zeros(ctx.item_unit.shape[1], np.float32) if ctx.item_unit is not None else None
        else:
            ctx._owned_prof[u] = ctx.item_unit[cols].sum(axis=0).astype(np.float32)
    return ctx._owned_prof[u]


def emb_prof(ctx: RecCtx, u: int) -> np.ndarray:
    """用户商品向量均值（vocab 维）。"""
    if u not in ctx._emb_prof:
        rows = []
        for j in owned_cols(ctx, u):          # j 是 item 列号
            code = ctx.item_codes[j]
            ri = ctx.embed_row.get(code)      # embed 词表行
            if ri is not None:
                rows.append(ri)
        if ctx.embed_unit is None or not rows:
            ctx._emb_prof[u] = np.zeros(ctx.embed_unit.shape[1], np.float32) if ctx.embed_unit is not None else None
        else:
            ctx._emb_prof[u] = ctx.embed_unit[rows].mean(axis=0).astype(np.float32)
    return ctx._emb_prof[u]


# ---------------------------------------------------------------- 三路召回
def _pop_order(ctx: RecCtx, exclude: set[str]) -> list[str]:
    b = np.asarray(ctx.B.sum(axis=0)).ravel()
    order = np.argsort(-b)
    return [ctx.code_of[i] for i in order if ctx.code_of[i] not in exclude]


def recall_user_vec(ctx: RecCtx, u: int, top: int) -> list[tuple[str, float]]:
    """用户向量路：已购商品向量均值 × 全词表余弦。"""
    prof = emb_prof(ctx, u)
    if prof is None or not np.any(prof):
        return []
    own = set(ctx.item_codes[owned_cols(ctx, u)])
    sims = ctx.embed_unit @ prof
    idx = np.argsort(-sims)
    out = []
    for i in idx:
        code = ctx.embed_keys[i]
        if code in own:
            continue
        out.append((code, float(sims[i])))
        if len(out) >= top:
            break
    return out


def recall_similar(ctx: RecCtx, u: int, top: int) -> list[tuple[str, float]]:
    """相似扩展路：从用户训练期「最后一单」的每个商品出发，取同订单配套邻居。"""
    recent = ctx.recent_items.get(int(u))
    if not recent or ctx.embed_unit is None:
        return []
    own = set(ctx.item_codes[owned_cols(ctx, u)])
    scores: dict[str, float] = {}
    for code in recent:
        ri = ctx.embed_row.get(code)
        if ri is None:
            continue
        sims = ctx.embed_unit @ ctx.embed_unit[ri]
        idx = np.argsort(-sims)[: 13]           # 每商品取 top-12
        for i in idx:
            nb = ctx.embed_keys[i]
            if nb == code or nb in own:
                continue
            scores[nb] = max(scores.get(nb, 0.0), float(sims[i]))
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[: top]
    return ranked


def recall_popular_only(ctx: RecCtx, u: int, k: int) -> list[tuple[str, float]]:
    """热门基线：全训练期买家数最多的商品，去掉已购。"""
    own = set(ctx.item_codes[owned_cols(ctx, u)])
    order = _pop_order(ctx, own)
    return [(order[i], float(-i)) for i in range(min(k, len(order)))]


def recall_itemcf_only(ctx: RecCtx, u: int, k: int) -> list[tuple[str, float]]:
    """ItemCF 基线：候选商品与用户已购集合的聚合余弦（同 05 rec_item），去已购。"""
    prof = owned_prof(ctx, u)
    if prof is None or not np.any(prof):
        return []
    own_cols = set(int(j) for j in owned_cols(ctx, u))
    scores = ctx.item_unit @ prof                    # (n_items,) 每候选对已购的聚合余弦
    scores[list(own_cols)] = -np.inf
    order = np.argsort(-scores)
    out = [(ctx.code_of[i], float(scores[i])) for i in order[:k]]
    return [x for x in out if x[1] != -np.inf]


def recall_fusion(ctx: RecCtx, u: int,
                  final_k: int = _FINAL_K, quotas: dict | None = None) -> list[dict]:
    """三路召回按配额融合去重，返回按「归一化融合分」降序的候选：
    [{'code', 'routes': [..], 'score'}]，不足 final_k 用热门补足。

    各路由分数量纲不同（热门=名次、嵌入=余弦），不能直接比大小；
    故把每路由内部的名次归一化到 0~1（第 0 名=1.0，第 q-1 名→接近 0），
    取同一商品命中的各路由分数的最大值作为 fusion 分，再跨路由排序，
    避免“热门路被分数量纲整体压到队尾”的假象。
    """
    quotas = quotas or _QUOTAS
    own = set(ctx.item_codes[owned_cols(ctx, u)])
    pop_cands = [(c, float(-i)) for i, c in enumerate(_pop_order(ctx, own))]

    def take(route: str, seq, q: int):
        """取每路前 q 名；返回 (code, route, 路由内排名归一化分 1 - i/q)。"""
        pick = []
        for i, (c, _sc) in enumerate(seq):
            if len(pick) >= q:
                break
            pick.append((c, route, max(0.0, 1.0 - i / max(1, q))))
        return pick

    merged: dict[str, dict] = {}
    order_seq = [
        *take("popular", pop_cands, quotas.get("popular", 0)),
        *take("similar", recall_similar(ctx, u, quotas.get("similar", 0) * 4),
              quotas.get("similar", 0)),
        *take("user_vec", recall_user_vec(ctx, u, quotas.get("user_vec", 0) * 4),
              quotas.get("user_vec", 0)),
    ]
    for c, route, norm in order_seq:
        if c in own or c not in ctx.col_of:
            continue
        m = merged.get(c)
        if m is None:
            merged[c] = {"code": c, "routes": [route], "score": norm}
        elif route not in m["routes"]:
            m["routes"].append(route)
            m["score"] = max(m["score"], norm)

    # 补足：热门溢出回填（排在所有配额候选之后）
    fill_i = 0
    for c, _sc in pop_cands:
        if len(merged) >= final_k:
            break
        if c not in merged:
            merged[c] = {"code": c, "routes": ["popular"],
                         "score": max(0.02, 0.2 - 0.01 * fill_i)}
            fill_i += 1

    ranked = sorted(merged.values(), key=lambda m: (-m["score"], m["code"]))
    return ranked[: final_k]


# ---------------------------------------------------------------- 特征
def features_for_rows(rows, ctx: RecCtx) -> np.ndarray:
    """rows: iterable of (user:int, code:str, routes:[str]) → (n, len(FEATURES))。"""
    rows = list(rows)
    X = np.zeros((len(rows), len(FEATURES)), dtype=np.float64)
    from collections import defaultdict
    by_user: dict[int, list[int]] = defaultdict(list)
    for idx, (u, code, routes) in enumerate(rows):
        by_user[u].append(idx)
    for u, idxs in by_user.items():
        prof_i = owned_prof(ctx, u)
        prof_e = emb_prof(ctx, u)
        own_cols = owned_cols(ctx, u)
        own = set(ctx.item_codes[own_cols])
        for idx in idxs:
            code, routes = rows[idx][1], rows[idx][2]
            col = ctx.col_of.get(code, -1)
            if col >= 0:
                X[idx, 0] = float(ctx.buyers[col])
                X[idx, 1] = float(ctx.pop_rank[col])
                X[idx, 2] = float(ctx.price_log[col])
            if ctx.embed_row and code in ctx.embed_row and prof_e is not None:
                X[idx, 3] = float(np.dot(ctx.embed_unit[ctx.embed_row[code]], prof_e))
            if prof_i is not None and col >= 0 and np.any(prof_i):
                X[idx, 4] = float(np.dot(ctx.item_unit[col], prof_i))
            rt = [r for r in routes if r in ("popular", "similar", "user_vec")]
            X[idx, 5] = len(rt)
            X[idx, 6] = 1.0 if "popular" in rt else 0.0
            X[idx, 7] = 1.0 if "similar" in rt else 0.0
            X[idx, 8] = 1.0 if "user_vec" in rt else 0.0
            X[idx, 9] = math.log1p(max(0, len(own)))
    return X


def no_leak_skip_vectors(df: pd.DataFrame, cache_path=None):
    """训练期-only 的 Skip-gram 商品向量（防泄漏铁律，同 scripts/08）。

    用「每用户最后一单=测试篮、更早=训练」的切分，只在训练期订单上重训 sg=1，
    返回 (keys, raw_vectors)；cache_path 存在时直接读取，避免 11/12 各自重训 15s。
    """
    from src.embedding import build_basket_sentences, load_vectors, save_vectors, train_basket_embeddings
    from config.settings import SEED

    if cache_path is not None and Path(cache_path).exists():
        keys, vecs = load_vectors(str(cache_path))
        return np.asarray(keys), vecs
    train, _test = split_last_invoice(df)
    sent = build_basket_sentences(train, seed=SEED, min_len=2)
    print(f"  重训语料：训练期 {len(sent)} 个订单篮（min_len>=2）")
    model, _loss = train_basket_embeddings(sent, sg=1, hyper={"seed": SEED})
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_vectors(model, str(cache_path))
        print(f"  已缓存无泄漏向量 → {cache_path}")
    return np.asarray(model.wv.index_to_key, dtype=str), model.wv.vectors


def score_rank(ctx: RecCtx, u: int, clf, scaler=None,
               final_k: int = _FINAL_K) -> list[tuple[str, float]]:
    """在线精排：取融合召回候选 → 特征 → 模型打分 → 降序返回 [(code, proba), ...]。

    clf 需要 predict_proba（LR / LGB 都是）；scaler 传 None 表示模型已内置缩放
    （我们导出的是 Pipeline，直接给 pipeline 即可）。
    """
    cands = recall_fusion(ctx, u, final_k=final_k)
    if not cands:
        return []
    X = features_for_rows([(u, c["code"], c["routes"]) for c in cands], ctx)
    if scaler is not None:
        X = scaler.transform(X)
    p = clf.predict_proba(X)[:, 1]
    order = np.argsort(-p)
    return [(cands[i]["code"], float(p[i])) for i in order]


def test_basket_map(test_df: pd.DataFrame) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for u, g in test_df.groupby(_USER):
        out[int(u)] = set(g[_ITEM].astype(str))
    return out


def precision_atk(ranked: list[str], truth: set[str], k: int) -> float:
    hit = len(set(ranked[:k]) & truth)
    return hit / k


def hit_atk(ranked: list[str], truth: set[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & truth else 0.0
