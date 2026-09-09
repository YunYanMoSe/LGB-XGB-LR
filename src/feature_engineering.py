"""阶段七 · 多维度特征挖掘 → 全数值宽表（推荐项目新增独立阶段）。

口径（与既有「最后一单=测试篮」协议刻意不同，自成一套，防泄漏铁律见下）：
    FEAT_CUT = 2011-11-09 为切窗锚点；
    特征期 feature period = 全部 InvoiceDate < FEAT_CUT 的历史；
    标签窗 label window  = 尾随 30 天 InvoiceDate >= FEAT_CUT（数据止于 2011-12-09）。
    正样本 = 用户在标签窗内「首次购买」的商品（不在其标签窗前的购买史中）；
    负样本 = 每正样本从「用户全数据集从未购买 ∩ 特征期有销量的商品目录」随机配 3 个（1:3）。

防泄漏铁律：
    所有模型特征只用特征期（FEAT_CUT 前）数据计算；
    复用的商品向量一律为「特征期订单篮重训」的 Skip-gram 向量（outputs/feature_stage/
    vectors_feat_skip.npz 缓存），严禁引用 models/item2vec_*.npz、outputs/chain/
    vectors_train_skip.npz（它们都含标签窗 / 测试篮信息）。
    类目 / 国家等静态知识表可直接使用（不是随时间变化的行为）。

产出物（由 scripts/15~17 落盘）：
    outputs/feature_stage/feature_wide.csv   宽表：user_id, item_id, 31 特征, label
    outputs/feature_stage/id_map.csv         商品 StockCode ↔ 数值 item_id
    outputs/feature_stage/category_map.csv   类目收拢 raw→collapsed
    outputs/feature_stage/features_meta.json 样本数 / 切分 / 特征均值 / 可复现信息
    models/lgb_feature.pkl + 重要性图        见 scripts/16_train_feature_lgb.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.sparse import coo_matrix, csr_matrix

from config.settings import COLUMNS, SEED
from src import recommender as rec

_USER, _ITEM = COLUMNS["user"], COLUMNS["item"]
_TIME, _QTY = COLUMNS["time"], COLUMNS["qty"]
_PRICE = COLUMNS["price"]
_INVOICE, _COUNTRY = COLUMNS["invoice"], COLUMNS["country"]

# ---------------------------------------------------------------- 口径常量
FEAT_CUT = pd.Timestamp("2011-11-09")     # 特征只用更早历史，>= 即标签窗
MIN_FP_ORDERS = 2                          # 候选用户：特征期订单数下限
NEG_RATIO = 3                              # 负样本配对比例 1:3
TOP_CAT_N = 9                              # 类目收拢保留头部类数（其余折叠进「其他」）
NN_K = 20                                  # ui_nn_hit_rate 的近邻数
EMB_DIM_OR = 128

# 特征列序（即宽表列序；改动必须与 features_meta.json 同步）。
# 前两列 user_id / item_id、末列 label 由脚本拼装，这里只列 31 个数值特征。
FEATURES = [
    # A. 用户画像（10）
    "u_n_orders_log", "u_n_items_log", "u_qty_log", "u_spend_log",
    "u_avg_unit_price_log", "u_avg_basket_size_log", "u_active_days_log",
    "u_last_active_gap_log", "u_weekend_ratio", "u_cat_entropy",
    # B. 商品 / 统计 / 时间（10）
    "i_pop_buyers_log", "i_pop_orders_log", "i_qty_log", "i_price_log",
    "i_pop_rank_frac", "i_first_gap_log", "i_last_gap_log",
    "i_cat_sales_log", "i_cat_sales_share", "i_top_weekday_code",
    # C. 用户 × 商品 交互（5）
    "ui_cat_lift_log", "ui_price_gap_log", "ui_emb_sim", "ui_cf_sim",
    "ui_nn_hit_rate",
    # D. 文本 → 数值编码（6）
    "country_uk", "country_de", "country_fr", "country_eire", "country_other",
    "cat_code_ordinal",
]

FEAT_ZH = {
    "u_n_orders_log": "用户历史订单数(log1p)",
    "u_n_items_log": "用户历史去重购买商品数(log1p)",
    "u_qty_log": "用户历史总购买件数(log1p)",
    "u_spend_log": "用户历史总消费金额(log1p)",
    "u_avg_unit_price_log": "用户历史自购商品均价(log1p)",
    "u_avg_basket_size_log": "用户平均每单商品种数(log1p)",
    "u_active_days_log": "用户历史活跃购买天数(log1p)",
    "u_last_active_gap_log": "距切窗最近一次购买间隔天数(log1p，0=非常近)",
    "u_weekend_ratio": "用户周末下单占比(0~1)",
    "u_cat_entropy": "用户购买类目集中度熵(归一化)",
    "i_pop_buyers_log": "商品热度：特征期购买用户数(log1p)",
    "i_pop_orders_log": "商品特征期被购订单数(log1p)",
    "i_qty_log": "商品特征期总销量(log1p)",
    "i_price_log": "商品特征期单价中位数(log1p)",
    "i_pop_rank_frac": "商品热门排名位置(0=最热→1=最冷)",
    "i_first_gap_log": "商品上架时间：首销距今间隔(log1p，值大=老品)",
    "i_last_gap_log": "商品滞销程度：最近销售距今间隔(log1p，值大=久未售)",
    "i_cat_sales_log": "所属类目特征期销售额(log1p)",
    "i_cat_sales_share": "所属类目特征期销售额占比(0~1)",
    "i_top_weekday_code": "商品销量峰值星期(0=周一~6=周日)",
    "ui_cat_lift_log": "用户对该类目的偏好提升比 log1p(用户类目份额/全局份额)",
    "ui_price_gap_log": "商品价位 − 用户自购均价（对数差，带符号）",
    "ui_emb_sim": "与用户历史购买向量的余弦（特征期重训向量）",
    "ui_cf_sim": "与用户已购集合的 ItemCF 聚合余弦（共同购买者画像）",
    "ui_nn_hit_rate": "商品向量近邻中被用户买过的占比(0~1)",
    "country_uk": "国家=英国 one-hot",
    "country_de": "国家=德国 one-hot",
    "country_fr": "国家=法国 one-hot",
    "country_eire": "国家=爱尔兰(EIRE) one-hot",
    "country_other": "国家=其它 one-hot",
    "cat_code_ordinal": "收拢后类目编码(按销售额降序 0=头部~9=长尾/其它)",
}

# 特征属组（md / 报告用）
GROUP = {}
for _f in FEATURES:
    if _f.startswith("u_"):
        GROUP[_f] = "A 用户画像"
    elif _f.startswith("i_"):
        GROUP[_f] = "B 商品·统计·时间"
    elif _f.startswith("ui_"):
        GROUP[_f] = "C 用户×商品交互"
    else:
        GROUP[_f] = "D 文本→数值编码"

_COUNTRY_LIST = ["United Kingdom", "Germany", "France", "EIRE"]
_COUNTRY_COLS = ["country_uk", "country_de", "country_fr", "country_eire", "country_other"]


def _load_category_of() -> dict[str, str]:
    """{StockCode: Product_Category}；products.csv 缺该商品的类目记空串。"""
    products = rec.load_products_zh()
    out: dict[str, str] = {}
    for code, info in products.items():
        cat = (info.get("category") or "").strip()
        out[str(code)] = cat
    return out


# ---------------------------------------------------------------- 切窗
def split_windows(df: pd.DataFrame, cut=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """特征期（< cut）与标签窗（>= cut）。cut 缺省用模块 FEAT_CUT（stage-7 口径）。"""
    c = FEAT_CUT if cut is None else pd.Timestamp(cut)
    fp = df[df[_TIME] < c].copy()
    lp = df[df[_TIME] >= c].copy()
    return fp, lp


# ---------------------------------------------------------------- 类目收拢
def setup_categories(fp: pd.DataFrame, category_of: dict[str, str]) -> dict:
    """按特征期销售额做类目收拢：top-N 保留、其余含未标注折叠进「其他」。

    Returns dict: raw_to_collapsed / ordered(按销售额降序) / ordinal_of /
        bucket_spend / bucket_share / collapse（闭包）。
    """
    s = fp[[_ITEM, _QTY, _PRICE]].copy()
    s["_spend"] = s[_QTY].astype(float) * s[_PRICE].astype(float)
    item_spend = s.groupby(_ITEM)["_spend"].sum()

    def raw_of(code: str) -> str:
        cat = category_of.get(str(code), "")
        return cat if cat else "未标注"

    raw = pd.Series([raw_of(c) for c in item_spend.index], index=item_spend.index)
    tbl = pd.DataFrame({"spend": item_spend, "raw": raw.values})
    eligible = tbl[~tbl["raw"].isin({"其他", "未标注"})]
    top_raw = (eligible.groupby("raw")["spend"].sum()
               .sort_values(ascending=False).head(TOP_CAT_N).index.tolist())
    top_set = set(top_raw)
    tbl["collapsed"] = np.where(tbl["raw"].isin(top_set), tbl["raw"], "其他")
    bucket_spend = tbl.groupby("collapsed")["spend"].sum().sort_values(ascending=False)
    ordered = list(bucket_spend.index)
    ordinal_of = {b: i for i, b in enumerate(ordered)}
    share_of = (bucket_spend / bucket_spend.sum()).to_dict()
    collapse = {r: (r if r in top_set else "其他") for r in tbl["raw"].unique()}
    collapse.setdefault("未标注", "其他")
    collapse.setdefault("其他", "其他")

    def collapse_fn(raw_label: str) -> str:
        raw_label = raw_label.strip()
        if not raw_label:
            return "其他"
        return collapse.get(raw_label, "其他")

    return {
        "raw_to_collapsed": {k: collapse_fn(k) for k in collapse},
        "top_raw": top_raw, "ordered": ordered, "ordinal_of": ordinal_of,
        "bucket_spend": bucket_spend, "bucket_share": share_of,
        "collapse": collapse_fn,
    }


# ---------------------------------------------------------------- 用户画像
def user_features(fp: pd.DataFrame, cat: dict, *, cut=None) -> pd.DataFrame:
    """特征期用户画像 → DataFrame(index=用户 int，升序)。含全部 10 个 u_* 特征、
    均价/类目份额等内部量、国家代码。cut 缺省用模块 FEAT_CUT（最近活跃间隔的锚点）。"""
    c = FEAT_CUT if cut is None else pd.Timestamp(cut)
    users = np.asarray(sorted(fp[_USER].unique()), dtype="int64")
    tmp = fp[[_USER, _ITEM, _INVOICE, _TIME, _QTY, _PRICE]].copy()
    tmp["_spend"] = tmp[_QTY].astype(float) * tmp[_PRICE].astype(float)
    g = tmp.groupby(_USER)

    n_orders = g[_INVOICE].nunique()
    n_items = g[_ITEM].nunique()
    qty_sum = g[_QTY].sum().clip(lower=0)
    spend = g["_spend"].sum().clip(lower=0)
    avg_px = spend / qty_sum.where(qty_sum > 0, np.nan)

    inv = tmp.groupby([_USER, _INVOICE])
    inv_items = inv[_ITEM].nunique()
    basket_mean = inv_items.groupby(level=0).mean()
    inv_wd = inv[_TIME].first().dt.weekday
    weekend_n = inv_wd[inv_wd >= 5].groupby(level=0).size()

    tmp["_day"] = tmp[_TIME].dt.date
    active_days = tmp.groupby(_USER)["_day"].nunique()
    last_ts = g[_TIME].max()
    gap_days = (c - last_ts).dt.days.clip(lower=0)

    # 类目熵（按去重购买商品的收拢类目）
    item_bucket = tmp[_ITEM].astype(str).map(cat["collapse_of_item"])
    tmp["_bucket"] = item_bucket.fillna("其他")
    cnt = tmp.groupby([_USER, "_bucket"])[_ITEM].nunique()
    c = cnt.unstack(fill_value=0).reindex(columns=cat["ordered"], fill_value=0)
    cidx = c.index
    srow = c.sum(axis=1).to_numpy().astype(np.float64)
    p_arr = c.to_numpy(dtype=np.float64)
    srow[srow == 0] = np.nan
    p_arr = p_arr / srow[:, None]
    p_arr[~np.isfinite(p_arr)] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.log(p_arr)
    logp[~np.isfinite(logp)] = 0.0
    ent = -(p_arr * logp).sum(axis=1)
    nb = (c > 0).sum(axis=1).to_numpy()
    ent_norm = pd.Series(ent / np.log(np.maximum(nb, 2)), index=cidx).where(nb >= 2, 0.0)

    # 用户-类目消费份额矩阵（ui_cat_lift 用）
    u_bucket_spend = tmp.assign(_b=tmp["_bucket"]).groupby([_USER, "_b"])["_spend"].sum()
    ubs = u_bucket_spend.unstack(fill_value=0.0).reindex(columns=cat["ordered"], fill_value=0.0)
    ubs_norm = ubs.div(ubs.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # 国家 one-hot 代码
    country = fp.groupby(_USER)[_COUNTRY].agg(lambda s: s.mode().iloc[0])

    out = pd.DataFrame(index=pd.Index(users, name=_USER))
    out["u_n_orders_log"] = np.log1p(n_orders.reindex(users).fillna(0))
    out["u_n_items_log"] = np.log1p(n_items.reindex(users).fillna(0))
    out["u_qty_log"] = np.log1p(qty_sum.reindex(users).fillna(0))
    out["u_spend_log"] = np.log1p(spend.reindex(users).fillna(0))
    out["u_avg_unit_price_log"] = np.log1p(avg_px.reindex(users).fillna(0).clip(lower=0))
    out["u_avg_basket_size_log"] = np.log1p(basket_mean.reindex(users).fillna(0))
    out["u_active_days_log"] = np.log1p(active_days.reindex(users).fillna(0))
    out["u_last_active_gap_log"] = np.log1p(gap_days.reindex(users).fillna(0))
    out["u_weekend_ratio"] = (weekend_n.reindex(users).fillna(0)
                              / n_orders.reindex(users).replace(0, np.nan)).fillna(0)
    out["u_cat_entropy"] = ent_norm.reindex(users).fillna(0)

    # 内部量：均价(原始)、n_orders、类目份额矩阵
    out["_avg_px_raw"] = avg_px.reindex(users).fillna(0.0)
    out["_n_orders"] = n_orders.reindex(users).fillna(0).astype("int64")
    out["_ubs_norm"] = [ubs_norm.loc[u].to_numpy() if u in ubs_norm.index
                        else np.zeros(len(cat["ordered"])) for u in users]
    country_code = pd.Series(
        [(_COUNTRY_LIST.index(x) if x in _COUNTRY_LIST else len(_COUNTRY_LIST))
         for x in country.reindex(users).fillna("")], index=pd.Index(users))
    out["_country"] = country_code.to_numpy()
    return out


# ---------------------------------------------------------------- 商品画像
def item_features(fp: pd.DataFrame) -> pd.DataFrame:
    """特征期商品画像 → DataFrame(index=商品码 升序)。含 8 个 per-item 统计字段
    （buyers/orders/qty/price_med/first/last/weekday_peak）+ buyers 计数（排名用）。"""
    catalog = np.asarray(sorted(fp[_ITEM].unique()), dtype=object)
    tmp = fp[[_ITEM, _USER, _INVOICE, _QTY, _PRICE, _TIME]].copy()
    g = tmp.groupby(_ITEM)

    buyers = g[_USER].nunique()
    orders = g[_INVOICE].nunique()
    qty = g[_QTY].sum()
    price_med = g[_PRICE].median()
    first_t = g[_TIME].min()
    last_t = g[_TIME].max()

    tmp["_wd"] = tmp[_TIME].dt.weekday
    wd_qty = tmp.groupby([_ITEM, "_wd"])[_QTY].sum()
    peak_wd = wd_qty.groupby(level=0).idxmax().map(lambda t: t[1])

    df_item = pd.DataFrame({
        "buyers": buyers, "orders": orders, "qty": qty,
        "price_med": price_med, "first": first_t, "last": last_t,
        "peak_wd": peak_wd,
    })
    df_item = df_item.reindex(catalog)
    return df_item


# ---------------------------------------------------------------- 特征期向量
def feature_skip_vectors(fp: pd.DataFrame, cache_path: Path):
    """特征期订单篮重训 Skip-gram 商品向量（防泄漏）。返回 (keys, raw_vecs)。"""
    from src.embedding import (build_basket_sentences, load_vectors,
                               save_vectors, train_basket_embeddings)
    cache_path = Path(cache_path)
    if cache_path.exists():
        keys, vecs = load_vectors(str(cache_path))
        return np.asarray(keys, dtype=object), vecs
    sent = build_basket_sentences(fp, seed=SEED, min_len=2)
    print(f"  [item2vec] 特征期订单篮 {len(sent)} 个，重训 sg=1（防泄漏）…")
    model, _ = train_basket_embeddings(sent, sg=1, hyper={"seed": SEED})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_vectors(model, str(cache_path))
    print(f"  [item2vec] 已缓存 → {cache_path}")
    keys, vecs = load_vectors(str(cache_path))
    return np.asarray(keys, dtype=object), vecs


def _unit_matrix(vecs: np.ndarray) -> np.ndarray:
    unit = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(unit, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return unit / norms


def _topk_neighbors(unitE: np.ndarray, k: int) -> np.ndarray:
    """每个词最近的 k 个（排除自身）在词表内的行号，(V, k)。"""
    sim = unitE @ unitE.T
    np.fill_diagonal(sim, -np.inf)
    idx = np.argsort(-sim, axis=1)[:, :k].astype(np.int64)
    return idx


# ---------------------------------------------------------------- 样本构造
def sample_rows(df: pd.DataFrame, fp: pd.DataFrame,
                n_orders_by_user: pd.Series) -> dict:
    """正样本（标签窗首购）+ 1:3 配对负采样 → 行序数组。

    Returns dict: pos_n / neg_n / sample_users(int array) / users 数组 /
        codes 数组 / labels 数组 / user_owned_codes（每样本用户特征期去重购买，列表）。
    """
    # 每用户「特征期已购」与「全数据集已购」
    owned_fp: dict[int, set[str]] = {}
    for u, gg in fp.groupby(_USER):
        owned_fp[int(u)] = set(gg[_ITEM].astype(str))
    owned_all: dict[int, set[str]] = {}
    for u, gg in df.groupby(_USER):
        owned_all[int(u)] = set(gg[_ITEM].astype(str))

    lp_items: dict[int, set[str]] = {}
    for u, gg in df[df[_TIME] >= FEAT_CUT].groupby(_USER):
        lp_items[int(u)] = set(gg[_ITEM].astype(str))

    # 候选用户 = 特征期订单数 >= MIN_FP_ORDERS
    cand_users = sorted(int(u) for u, n in n_orders_by_user.items()
                        if int(n) >= MIN_FP_ORDERS)
    catalog = set(fp[_ITEM].astype(str).unique())

    # 正样本：候选用户在标签窗首次买（不在其特征期购买史）
    positives: list[tuple[int, str]] = []
    for u in cand_users:
        new_items = (lp_items.get(u, set()) - owned_fp.get(u, set()))
        for code in sorted(new_items):
            positives.append((int(u), code))

    # 负样本：配对 1:3（每正样本抽 3 个“全数据集从未购买 ∩ 目录”的商品）
    rng = np.random.default_rng(SEED)
    pos_by_user: dict[int, int] = {}
    for u, code in positives:
        pos_by_user[u] = pos_by_user.get(u, 0) + 1
    negatives: list[tuple[int, str]] = []
    short: list[tuple[int, int, int]] = []
    for u in cand_users:
        n_p = pos_by_user.get(u, 0)
        if n_p == 0:
            continue
        pool_arr = np.asarray(sorted(catalog - owned_all.get(u, set())), dtype=object)
        need = NEG_RATIO * n_p
        if need <= len(pool_arr):
            picks = pool_arr[rng.choice(len(pool_arr), size=need, replace=False)]
        else:  # 理论上目录足够大不会触发，留一道保险
            picks = pool_arr
            short.append((u, n_p, len(pool_arr)))
        for code in picks:
            negatives.append((int(u), str(code)))
    if short:
        print(f"  [样本] 注意：{len(short)} 个用户负池不足 3 倍正样本，已退化为全池负采样")

    # 行序：按用户分组，组内先正后负（代码升序，可复现）
    pos_map: dict[int, list[str]] = {}
    for u, code in positives:
        pos_map.setdefault(u, []).append(code)
    neg_map: dict[int, list[str]] = {}
    for u, code in negatives:
        neg_map.setdefault(u, []).append(code)

    sample_users, user_codes = [], []
    labels: list[int] = []
    for u in cand_users:
        if u not in pos_map:
            continue
        for code in pos_map[u]:
            sample_users.append(int(u)); user_codes.append(code); labels.append(1)
        for code in neg_map.get(u, []):
            sample_users.append(int(u)); user_codes.append(code); labels.append(0)

    return {
        "pos_n": len(positives), "neg_n": len(negatives),
        "pos_short": short,
        "users": np.asarray(sample_users, dtype="int64"),
        "codes": np.asarray(user_codes, dtype=object),
        "labels": np.asarray(labels, dtype="int64"),
        "owned_fp": owned_fp, "catalog": catalog,
    }


# ---------------------------------------------------------------- 组装宽表
def build_feature_wide(df: pd.DataFrame, vector_cache: Path) -> pd.DataFrame:
    """主入口：切窗 → 类目/国家/用户/商品画像 → 特征期向量 → 1:3 负采样 →
    逐用户批量拼接 31 特征 → 返回宽表 DataFrame（user_id, item_id, 31 特征, label）。

    宽表里 item_id 为数值：StockCode 经 id_map 排序映射 0..n-1（脚本负责落盘 id_map）。
    """
    # ---- 切窗 ----
    fp, lp = split_windows(df)
    n_lp_rows = len(lp)
    print(f"[切窗] 特征期 {len(fp):,} 行；标签窗 {n_lp_rows:,} 行（{fp[_TIME].min().date()} ~ "
          f"{FEAT_CUT.date()} | 标签窗 ≥ {FEAT_CUT.date()}）")

    category_of = _load_category_of()
    cat = setup_categories(fp, category_of)
    # 给商品码 -> 收拢类目 的 Series（含未标注商品码也映射到「其他」）
    all_codes = np.unique(np.concatenate([
        fp[_ITEM].astype(str).to_numpy(),
        df[_ITEM].astype(str).to_numpy()]))
    cat["collapse_of_item"] = pd.Series(
        {str(c): cat["collapse"](category_of.get(str(c), "")) for c in all_codes})

    U = user_features(fp, cat)
    users_all = U.index.to_numpy(dtype="int64")
    user_pos = {int(u): i for i, u in enumerate(users_all)}
    I = item_features(fp)
    catalog = I.index.to_numpy(dtype=object)
    cat_pos = {str(c): i for i, c in enumerate(catalog)}
    print(f"[画像] 特征期用户 {len(users_all):,}；目录商品 {len(catalog):,}")

    # ---- 特征期向量（缓存）----
    keys, vecs = feature_skip_vectors(fp, vector_cache)
    unitE = _unit_matrix(vecs)
    V = len(keys)
    emb_pos = {str(k): i for i, k in enumerate(keys)}
    Nb = _topk_neighbors(unitE, NN_K)
    print(f"[向量] 词表 {V}（min_count=5，特征期重训）")

    # ---- 样本 ----
    n_orders_by_user = fp.groupby(_USER)[_INVOICE].nunique()
    S = sample_rows(df, fp, n_orders_by_user)
    users_s = S["users"]; codes_s = S["codes"]; labels_s = S["labels"]
    n_s = len(users_s)
    upos = np.asarray([user_pos[int(u)] for u in users_s], dtype="int64")
    icat = np.asarray([cat_pos.get(str(c), -1) for c in codes_s], dtype="int64")
    ivoc = np.asarray([emb_pos.get(str(c), -1) for c in codes_s], dtype="int64")
    bucket_idx_s = np.zeros(n_s, dtype="int64")
    for i in range(n_s):
        c = str(codes_s[i])
        b = cat["collapse_of_item"].get(c, "其他")
        bucket_idx_s[i] = cat["ordinal_of"][b]
    print(f"[样本] 候选用户 {sum(int(n) >= MIN_FP_ORDERS for n in n_orders_by_user):,}；"
          f"正样本 {S['pos_n']:,} / 负样本 {S['neg_n']:,}（1:{S['neg_n']/max(1,S['pos_n']):.1f}）"
          f"；行 {n_s:,}")

    # ---- 逐列取数 ----
    X = np.zeros((n_s, len(FEATURES)), dtype=np.float64)
    feat_i = {f: i for i, f in enumerate(FEATURES)}

    u10 = U[FEATURES[:10]].to_numpy(dtype=np.float64)      # 顺序与 FEATURES[:10] 一致
    X[:, 0:10] = u10[upos]

    # 商品纯统计（列 11,12,13,14,15,16,17,20 = FEATURES 中 per-item 8 列）
    buyers = I["buyers"].to_numpy(dtype=np.float64)
    rank = np.argsort(np.argsort(-buyers))
    rank_frac = rank / max(1, len(catalog) - 1)
    buyers_log = np.log1p(buyers)
    orders_log = np.log1p(I["orders"].to_numpy(dtype=np.float64))
    qty_log = np.log1p(I["qty"].to_numpy(dtype=np.float64).clip(min=0))
    price_log = np.log1p(I["price_med"].to_numpy(dtype=np.float64).clip(min=0))
    first_gap = (FEAT_CUT - I["first"]).dt.days.clip(lower=0).to_numpy(dtype=np.float64)
    last_gap = (FEAT_CUT - I["last"]).dt.days.clip(lower=0).to_numpy(dtype=np.float64)
    peak_wd = I["peak_wd"].fillna(0).to_numpy(dtype=np.float64)
    base_cols = {
        "i_pop_buyers_log": buyers_log, "i_pop_orders_log": orders_log,
        "i_qty_log": qty_log, "i_price_log": price_log,
        "i_pop_rank_frac": rank_frac, "i_first_gap_log": np.log1p(first_gap),
        "i_last_gap_log": np.log1p(last_gap), "i_top_weekday_code": peak_wd,
    }
    for f, arr in base_cols.items():
        vals = np.zeros(n_s)
        valid = icat >= 0
        vals[valid] = arr[icat[valid]]
        X[:, feat_i[f]] = vals

    # 类目口径两列（按收拢类目，冷商品也能命中其类目）
    bucket_spend = cat["bucket_spend"].to_numpy(dtype=np.float64)
    bshare = bucket_spend / bucket_spend.sum()
    X[:, feat_i["i_cat_sales_log"]] = np.log1p(bucket_spend[bucket_idx_s])
    X[:, feat_i["i_cat_sales_share"]] = bshare[bucket_idx_s]

    # 文本编码：国家 one-hot + 类目 ordinal
    cc = U["_country"].to_numpy(dtype="int64")
    country_code_s = cc[upos]
    for k, col in enumerate(_COUNTRY_COLS):
        X[:, feat_i[col]] = (country_code_s == k).astype(np.float64)
    X[:, feat_i["cat_code_ordinal"]] = bucket_idx_s.astype(np.float64)

    # 交互三件套：用户已购类目偏好 / 价格差 / 向量 / ItemCF / 近邻命中
    ubs_all = U["_ubs_norm"].to_numpy(dtype=object)
    user_avg_px = U["_avg_px_raw"].to_numpy(dtype=np.float64)

    # (i) ui_cat_lift：用户在该类目消费份额 / 全局份额
    global_share = bshare
    cat_lift = np.zeros(n_s)
    for i in range(n_s):
        sh = ubs_all[upos[i]]
        cat_lift[i] = sh[bucket_idx_s[i]]
    cat_lift = np.log1p(cat_lift / np.maximum(global_share[bucket_idx_s], 1e-9))
    X[:, feat_i["ui_cat_lift_log"]] = cat_lift

    # (ii) ui_price_gap：log1p(商品价) − log1p(用户均价)，未知价(冷)置 0
    med_log = price_log
    uavg_log = np.log1p(user_avg_px.clip(min=0))
    gap = np.zeros(n_s)
    valid = icat >= 0
    gap[valid] = med_log[icat[valid]] - uavg_log[upos[valid]]
    X[:, feat_i["ui_price_gap_log"]] = gap

    # (iii) ui_emb_sim / ui_nn_hit_rate —— 逐用户批量
    emb_sim = np.zeros(n_s)
    nn_hit = np.zeros(n_s)
    owned_in_vocab: dict[int, np.ndarray] = {}
    for pos, owned in S["owned_fp"].items():
        idxs = [emb_pos[str(c)] for c in owned if str(c) in emb_pos]
        owned_in_vocab[int(pos)] = np.asarray(idxs, dtype="int64")

    for pos_u in np.unique(upos):
        u_id = int(users_all[pos_u])
        rows = np.flatnonzero(upos == pos_u)
        own_v = owned_in_vocab.get(u_id, np.asarray([], dtype="int64"))
        if len(own_v) == 0:
            continue
        prof = unitE[own_v].mean(axis=0)
        owned_mask = np.zeros(V, dtype=bool)
        owned_mask[own_v] = True
        for i in rows:
            e = int(ivoc[i])
            if e < 0:
                continue
            emb_sim[i] = float(np.dot(unitE[e], prof))
            nn_hit[i] = float(owned_mask[Nb[e]].mean())
    X[:, feat_i["ui_emb_sim"]] = emb_sim
    X[:, feat_i["ui_nn_hit_rate"]] = nn_hit

    # (iv) ui_cf_sim —— ItemCF 聚合余弦
    # B: (fp用户 × 目录商品) 0/1；item_unit 每商品行归一化；用户画像 = 已购行求和
    s = fp[[_USER, _ITEM]].drop_duplicates()
    uidx = np.asarray([user_pos[int(u)] for u in s[_USER]], dtype="int64")
    iidx = np.asarray([cat_pos[str(c)] for c in s[_ITEM]], dtype="int64")
    B = coo_matrix((np.ones(len(uidx), dtype=np.float32), (uidx, iidx)),
                   shape=(len(users_all), len(catalog))).tocsr()
    MI = B.T.tocsr()
    dense = MI.toarray().astype(np.float32)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    item_unit = (dense / norms).astype(np.float32)          # (目录, 用户)
    cf_prof_all = (B @ item_unit).astype(np.float32)        # (用户, 用户) Σ 已购行
    cf_sim = np.zeros(n_s)
    for pos_u in np.unique(upos):
        rows = np.flatnonzero(upos == pos_u)
        prof = cf_prof_all[pos_u]
        for i in rows:
            k = int(icat[i])
            if k < 0:
                continue
            cf_sim[i] = float(np.dot(item_unit[k], prof))
    X[:, feat_i["ui_cf_sim"]] = cf_sim

    # ---- 拼装宽表 ----
    # item_id 数值映射：只映射出现在样本里的商品码
    sample_codes = np.unique(codes_s)
    id_of = {str(c): i for i, c in enumerate(sorted(sample_codes, key=str))}
    item_id_s = np.asarray([id_of[str(c)] for c in codes_s], dtype="int64")

    # 保持列名 user_id / item_id / 特征 / label
    out = pd.DataFrame(index=np.arange(n_s))
    out["user_id"] = users_s
    out["item_id"] = item_id_s
    feat_df = pd.DataFrame(X, columns=FEATURES, dtype="float64")
    out = pd.concat([out, feat_df], axis=1)
    out["label"] = labels_s

    ordered = cat["ordered"]
    bucket_spend_full = cat["bucket_spend"]
    bshare_full = bucket_spend_full / bucket_spend_full.sum()
    out.attrs["id_map"] = {str(c): id_of[str(c)] for c in sorted(sample_codes, key=str)}
    out.attrs["buckets"] = [
        {"code": i, "name": b,
         "revenue": round(float(bucket_spend_full[b]), 2),
         "share": round(float(bshare_full[b]), 4)}
        for i, b in enumerate(ordered)
    ]
    raw_labels = sorted(set(str(x) for x in category_of.values()) | {"", "未标注", "其他"})
    out.attrs["collapse_raw"] = {r: cat["collapse"](r) for r in raw_labels}
    return out


# ---------------------------------------------------------------- 校验
def verify_wide(wide_csv: Path, id_map_csv: Path, meta: dict) -> None:
    """自检宽表：列序 / 全数值 / NaN=0 / 行数 / item_id 合法。"""
    wide = pd.read_csv(wide_csv)
    id_map = pd.read_csv(id_map_csv, dtype={"item_id": "int64", "stock_code": str})
    assert list(wide.columns[:2]) == ["user_id", "item_id"], "前两列应为 user_id/item_id"
    assert wide.columns[-1] == "label", "末列应为 label"
    feat_names = list(wide.columns[2:-1])
    assert feat_names == FEATURES, "特征列序与 FEATURES 不一致"
    bad = [c for c in wide.columns if wide[c].dtype == object or str(wide[c].dtype).startswith("category")]
    assert not bad, f"存在非数值列：{bad}"
    assert wide.isna().sum().sum() == 0, f"存在 NaN：{int(wide.isna().sum().sum())} 个"
    assert len(wide) == meta["n_pos"] + meta["n_neg"], "行数 ≠ n_pos+n_neg"
    assert set(wide["item_id"]).issubset(set(id_map["item_id"])), "item_id 超出 id_map"
    assert wide["label"].isin([0, 1]).all(), "label 非 0/1"
    print("  [校验] 通过：", len(wide), "行 ×", len(wide.columns), "列；dtypes 全数值、NaN=0、列序正确")
