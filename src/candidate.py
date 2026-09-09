"""候选集打分模块（train.csv 独立任务）。

以 data/processed/clean_train.csv（scripts/18 产物）为原料，对"任意 (user_id, StockCode)
候选对"产出打分特征并训练 LR / LightGBM，供 scripts/19/20 使用。

复用 src/feature_engineering 的画像/向量/类目/ItemCF 逻辑，但标签口径不同：
    - 时间切分锚点 = cut（本任务默认 train.csv 尾 31 天做标签窗，见 CUT_DEFAULT）；
    - 标签 = 用户在标签窗内"是否购买过该商品"（**含复购**，候选集里有历史已购对）；
    - 负样本刻意允许抽"特征期已购但标签窗未复购"的商品，模拟真实候选集的已购形态；
    - 特征 = fe.FEATURES 31 列 + 追加 ui_owned / ui_owned_recency_log（33 列，全为 cut 前历史）。

训练特征与打分特征共用 features_for_pairs（同一行、同一装配代码），杜绝列序漂移。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from config.settings import COLUMNS, PROJECT_ROOT, SEED
from src import feature_engineering as fe

_USER = COLUMNS["user"]
_ITEM = COLUMNS["item"]
_QTY = COLUMNS["qty"]
_PRICE = COLUMNS["price"]
_INVOICE = COLUMNS["invoice"]
_TIME = COLUMNS["time"]
_COUNTRY = COLUMNS["country"]

# ---------------------------------------------------------------- 常量
CUT_DEFAULT = pd.Timestamp("2010-10-01")   # 历史 < cut；标签窗 = 其后到数据尾（31 天）
MIN_FP_ORDERS = 2                          # 候选用户：特征期订单数下限
NEG_RATIO = 3                              # 负样本配对比例 1:NEG_RATIO
OWNED_NEG_FRAC = 0.125                     # 负样本中"特征期已购未复购"的占比（贴近候选集形态）
EVAL_NEG = 200                             # valid 用户每人在其真实负样本外补抽的负对数
MODEL_VERSION = "cand-lgb-v1"              # 提交时填写的模型版本号（每次迭代递增）

# 追加特征（fp 历史即可算，无泄漏）；顺序即列序尾段
EXTRA_FEATS = ["ui_owned", "ui_owned_recency_log"]
FEATURES_CAND = list(fe.FEATURES) + EXTRA_FEATS

# 服务/非商品码（精确匹配黑名单，避免误伤如 GOTHIC CARRIAGE LANTERN 这类真商品）
SERVICE_CODES = {
    "POST", "DOT", "M", "S", "B", "PADS", "C2", "D",
    "BANK CHARGES", "ADJUST", "ADJUST2", "TEST001", "TEST002",
}

CLEAN_TRAIN = PROJECT_ROOT / "data" / "processed" / "clean_train.csv"
VEC_CACHE = PROJECT_ROOT / "outputs" / "candidate" / "vectors_train_skip.npz"
CAND_META = PROJECT_ROOT / "models" / "candidate_meta.json"
LGB_PKL = PROJECT_ROOT / "models" / "candidate_lgb.pkl"
LR_PKL = PROJECT_ROOT / "models" / "candidate_lr.pkl"


# ---------------------------------------------------------------- 读入
def load_clean_train(path: str | Path | None = None) -> pd.DataFrame:
    """读 clean_train.csv → CustomerID int64 / StockCode str / InvoiceDate datetime。"""
    p = Path(path) if path is not None else CLEAN_TRAIN
    df = pd.read_csv(p)
    df[_USER] = pd.to_numeric(df[_USER], errors="coerce").astype("int64")
    df[_ITEM] = df[_ITEM].astype(str)
    df[_TIME] = pd.to_datetime(df[_TIME])
    return df


def drop_service_rows(df: pd.DataFrame) -> pd.DataFrame:
    """删除 StockCode ∈ 服务码黑名单 的行（精确匹配）。"""
    return df[~df[_ITEM].astype(str).isin(SERVICE_CODES)].copy()


# ---------------------------------------------------------------- 画像 bundle
def build_bundle(
    clean: pd.DataFrame,
    *,
    cut: pd.Timestamp | str = CUT_DEFAULT,
    vec_cache: str | Path = VEC_CACHE,
) -> dict:
    """一次构建训练/打分共用的全部画像：切窗、类目、用户/商品画像、向量、ItemCF。

    vec_cache 必须为该 (cut, 数据切片) 专属路径，勿复用 outputs/feature_stage 的缓存。
    """
    cut = pd.Timestamp(cut)
    fp = clean[clean[_TIME] < cut].copy()

    category_of = fe._load_category_of()
    cat = fe.setup_categories(fp, category_of)
    all_codes = np.unique(np.concatenate([
        fp[_ITEM].astype(str).to_numpy(), clean[_ITEM].astype(str).to_numpy()]))
    cat["collapse_of_item"] = pd.Series(
        {str(c): cat["collapse"](category_of.get(str(c), "")) for c in all_codes})

    U = fe.user_features(fp, cat, cut=cut)          # index = fp 用户 int（升序）
    users_all = U.index.to_numpy(dtype="int64")
    user_pos = {int(u): i for i, u in enumerate(users_all)}

    I = fe.item_features(fp)                        # index = fp 目录商品码（升序）
    catalog = I.index.to_numpy(dtype=object)
    cat_pos = {str(c): i for i, c in enumerate(catalog)}

    keys, vecs = fe.feature_skip_vectors(fp, Path(vec_cache))
    unitE = fe._unit_matrix(vecs)
    emb_pos = {str(k): i for i, k in enumerate(keys)}
    V = len(keys)
    Nb = fe._topk_neighbors(unitE, fe.NN_K)

    # fp 已购：全集 + 每 (u, item) 的最后购买时刻（ui_owned / recency 用）
    owned_fp: dict[int, set[str]] = {}
    owned_last: dict[tuple[int, str], pd.Timestamp] = {}
    for u, gg in fp.groupby(_USER):
        uid = int(u)
        owned_fp[uid] = set(gg[_ITEM].astype(str))
        for code, ts in gg.groupby(_ITEM)[_TIME].max().items():
            owned_last[(uid, str(code))] = ts

    # ItemCF：B = (fp 用户 × 目录商品) 0/1；item_unit 每商品行归一；用户画像 = 已购行求和
    s = fp[[_USER, _ITEM]].drop_duplicates()
    uidx = np.asarray([user_pos[int(u)] for u in s[_USER]], dtype="int64")
    iidx = np.asarray([cat_pos[str(c)] for c in s[_ITEM]], dtype="int64")
    B = coo_matrix((np.ones(len(uidx), dtype=np.float32), (uidx, iidx)),
                   shape=(len(users_all), len(catalog))).tocsr()
    dense = B.T.toarray().astype(np.float32)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    item_unit = (dense / norms).astype(np.float32)          # (目录, fp用户)
    cf_prof_all = (B @ item_unit).astype(np.float32)        # (fp用户, fp用户)

    return {
        "cut": cut, "fp": fp, "category_of": category_of, "cat": cat,
        "U": U, "users_all": users_all, "user_pos": user_pos,
        "I": I, "catalog": catalog, "cat_pos": cat_pos,
        "keys": keys, "unitE": unitE, "emb_pos": emb_pos, "V": V, "Nb": Nb,
        "owned_fp": owned_fp, "owned_last": owned_last,
        "item_unit": item_unit, "cf_prof_all": cf_prof_all,
    }


# ---------------------------------------------------------------- 任意候选对特征
def features_for_pairs(
    bundle: dict,
    users,
    codes,
) -> pd.DataFrame:
    """对任意候选对 (users[], codes[]) 装配 33 列打分特征，列序 = FEATURES_CAND。

    冷用户（fp 无购买）：用户特征 0、country 按 uk、交互特征 0；
    冷商品（fp 无售）：商品统计 0、类目列仍按收拢类目取值、交互特征 0。
    """
    n = len(users)
    cat = bundle["cat"]; U = bundle["U"]; I = bundle["I"]
    user_pos = bundle["user_pos"]; cat_pos = bundle["cat_pos"]
    unitE = bundle["unitE"]; emb_pos = bundle["emb_pos"]; Nb = bundle["Nb"]
    V = bundle["V"]; owned_fp = bundle["owned_fp"]; owned_last = bundle["owned_last"]
    item_unit = bundle["item_unit"]; cf_prof_all = bundle["cf_prof_all"]
    cut = bundle["cut"]
    users_all = bundle["users_all"]
    catalog = bundle["catalog"]

    user_list = [int(u) for u in np.asarray(users)]
    code_list = [str(c) for c in np.asarray(codes)]
    upos = np.asarray([user_pos.get(u, -1) for u in user_list], dtype="int64")
    icat = np.asarray([cat_pos.get(c, -1) for c in code_list], dtype="int64")
    ivoc = np.asarray([emb_pos.get(c, -1) for c in code_list], dtype="int64")

    ordered = cat["ordered"]
    last_ord = max(0, len(ordered) - 1)
    bucket_idx = np.asarray([
        cat["ordinal_of"].get(
            cat["collapse_of_item"].get(c, "其他"), last_ord) for c in code_list
    ], dtype="int64")

    X = np.zeros((n, len(FEATURES_CAND)), dtype=np.float64)
    fi = {f: i for i, f in enumerate(FEATURES_CAND)}
    um = upos >= 0

    # A 用户 10 列（冷用户全 0）
    u10 = U[fe.FEATURES[:10]].to_numpy(dtype=np.float64)
    X[um, :10] = u10[upos[um]]

    # B 商品 8 列纯统计（冷商品全 0）
    buyers = I["buyers"].to_numpy(dtype=np.float64)
    rank = np.argsort(np.argsort(-buyers))
    rank_frac = rank / max(1, len(catalog) - 1)
    price_med_log = np.log1p(I["price_med"].to_numpy(dtype=np.float64).clip(min=0))
    base = {
        "i_pop_buyers_log": np.log1p(buyers),
        "i_pop_orders_log": np.log1p(I["orders"].to_numpy(dtype=np.float64)),
        "i_qty_log": np.log1p(I["qty"].to_numpy(dtype=np.float64).clip(min=0)),
        "i_price_log": price_med_log,
        "i_pop_rank_frac": rank_frac,
        "i_first_gap_log": np.log1p(
            (cut - I["first"]).dt.days.clip(lower=0).to_numpy(dtype=np.float64)),
        "i_last_gap_log": np.log1p(
            (cut - I["last"]).dt.days.clip(lower=0).to_numpy(dtype=np.float64)),
        "i_top_weekday_code": I["peak_wd"].fillna(0).to_numpy(dtype=np.float64),
    }
    valid_i = icat >= 0
    for f, arr in base.items():
        vals = np.zeros(n)
        vals[valid_i] = arr[icat[valid_i]]
        X[:, fi[f]] = vals

    # 类目两列（冷商品也命中其收拢类目）
    bucket_spend = cat["bucket_spend"].to_numpy(dtype=np.float64)
    bshare = bucket_spend / bucket_spend.sum()
    X[:, fi["i_cat_sales_log"]] = np.log1p(bucket_spend[bucket_idx])
    X[:, fi["i_cat_sales_share"]] = bshare[bucket_idx]

    # D 文本编码：国家 one-hot（冷用户缺省走 country_other，与 fe 未知国家口径一致）+ 类目 ordinal
    cc = U["_country"].to_numpy(dtype="int64")
    country_code_s = np.full(n, len(fe._COUNTRY_LIST), dtype="int64")
    country_code_s[um] = cc[upos[um]]
    _country_cols = ["country_uk", "country_de", "country_fr", "country_eire", "country_other"]
    for k, col in enumerate(_country_cols):
        X[:, fi[col]] = (country_code_s == k).astype(np.float64)
    X[:, fi["cat_code_ordinal"]] = bucket_idx.astype(np.float64)

    # C 交互：类目偏好 / 价格差 / 向量 / ItemCF / 近邻命中
    ubs_all = U["_ubs_norm"].to_numpy(dtype=object)
    UBS = np.vstack([ubs_all[p] for p in range(len(ubs_all))]).astype(np.float64)  # (U, B)
    user_ubs = np.zeros((n, len(ordered)))
    user_ubs[um] = UBS[upos[um]]

    lift_num = user_ubs[np.arange(n), bucket_idx]
    X[:, fi["ui_cat_lift_log"]] = np.log1p(
        lift_num / np.maximum(bshare[bucket_idx], 1e-9))

    user_avg_px_raw = U["_avg_px_raw"].to_numpy(dtype=np.float64).clip(min=0)
    uavg_log = np.log1p(user_avg_px_raw)
    gap = np.zeros(n)
    both = um & valid_i
    gap[both] = price_med_log[icat[both]] - uavg_log[upos[both]]
    X[:, fi["ui_price_gap_log"]] = gap

    # (iii) emb_sim / nn_hit：逐用户（仅 fp 用户，取词表内已购）
    emb_sim = np.zeros(n)
    nn_hit = np.zeros(n)
    for pos_u in np.unique(upos):
        if pos_u < 0:
            continue
        uid = int(users_all[pos_u])
        own = owned_fp.get(uid, set())
        # 排序词表下标：set 迭代序随进程字符串哈希变化，会改变 mean 的浮点累加序→跨进程抖动
        own_v = np.asarray(sorted(emb_pos[c] for c in own if c in emb_pos), dtype="int64")
        if len(own_v) == 0:
            continue
        prof = unitE[own_v].mean(axis=0)
        owned_mask = np.zeros(V, dtype=bool)
        owned_mask[own_v] = True
        rows = np.flatnonzero(upos == pos_u)
        e = ivoc[rows]
        ok = e >= 0
        if ok.any():
            emb_sim[rows[ok]] = unitE[e[ok]] @ prof
            nn_hit[rows[ok]] = owned_mask[Nb[e[ok]]].mean(axis=1)
    X[:, fi["ui_emb_sim"]] = emb_sim
    X[:, fi["ui_nn_hit_rate"]] = nn_hit

    # (iv) cf_sim：ItemCF 聚合余弦
    cf_sim = np.zeros(n)
    for pos_u in np.unique(upos):
        if pos_u < 0:
            continue
        rows = np.flatnonzero(upos == pos_u)
        k = icat[rows]
        ok = k >= 0
        if ok.any():
            prof = cf_prof_all[pos_u]
            cf_sim[rows[ok]] = item_unit[k[ok]] @ prof
    X[:, fi["ui_cf_sim"]] = cf_sim

    # 追加特征：是否 fp 已购 / 已购新鲜度
    owned_bool = np.zeros(n, dtype=np.float64)
    recency = np.zeros(n, dtype=np.float64)
    for i in range(n):
        uid = user_list[i]
        code = code_list[i]
        st = owned_fp.get(uid, set())
        if code in st:
            owned_bool[i] = 1.0
            recency[i] = np.log1p(max(0.0, (cut - owned_last[(uid, code)]).days))
    X[:, fi["ui_owned"]] = owned_bool
    X[:, fi["ui_owned_recency_log"]] = recency

    return pd.DataFrame(X, columns=FEATURES_CAND, dtype="float64")


# ---------------------------------------------------------------- 训练集构建
def build_pair_dataset(
    clean: pd.DataFrame,
    bundle: dict,
    *,
    cut: pd.Timestamp | str = CUT_DEFAULT,
    min_fp_orders: int = MIN_FP_ORDERS,
    neg_ratio: int = NEG_RATIO,
    owned_neg_frac: float = OWNED_NEG_FRAC,
    seed: int = SEED,
) -> pd.DataFrame:
    """切尾窗标签：正=用户在标签窗买过（含复购），负=未买（掺 fp 已购未复购）。

    Returns wide：user_id(int64) / stock_code(str) / 33 特征 / label(0/1)。
    """
    cut = pd.Timestamp(cut)
    fp = clean[clean[_TIME] < cut]
    lp = clean[clean[_TIME] >= cut]

    n_orders_by_user = fp.groupby(_USER)[_INVOICE].nunique()
    cand_users = sorted(int(u) for u, n in n_orders_by_user.items()
                        if int(n) >= min_fp_orders)

    lp_items: dict[int, set[str]] = {}
    for u, gg in lp.groupby(_USER):
        lp_items[int(u)] = set(gg[_ITEM].astype(str))

    # 正样本：标签窗购买（不去重历史已购 → 含复购），每用户组内升序保可复现
    positives: list[tuple[int, str]] = []
    for u in cand_users:
        for code in sorted(lp_items.get(u, set())):
            positives.append((int(u), code))
    pos_by_user: dict[int, int] = {}
    for u, _c in positives:
        pos_by_user[u] = pos_by_user.get(u, 0) + 1

    catalog = set(fp[_ITEM].astype(str).unique())
    rng = np.random.default_rng(seed)
    negatives: list[tuple[int, str]] = []
    for u in cand_users:
        n_p = pos_by_user.get(u, 0)
        if n_p == 0:
            continue
        p = lp_items.get(u, set())
        owned = bundle["owned_fp"].get(u, set())
        owned_neg = np.asarray(sorted(owned - p), dtype=object)          # 已购未复购
        other = np.asarray(sorted(catalog - p - owned), dtype=object)    # 从未买
        n_neg = neg_ratio * n_p
        k_own = min(len(owned_neg), int(round(owned_neg_frac * n_neg)))
        picks: list[str] = []
        if k_own > 0:
            picks.extend(owned_neg[rng.choice(len(owned_neg), size=k_own, replace=False)])
        rem = n_neg - len(picks)
        if rem > 0 and len(other) > 0:
            m = min(rem, len(other))
            picks.extend(other[rng.choice(len(other), size=m, replace=False)])
        for code in picks:
            negatives.append((int(u), str(code)))

    rows_u = np.asarray([u for u, _c in positives] + [u for u, _c in negatives], dtype="int64")
    rows_c = np.asarray([c for _u, c in positives] + [c for _u, c in negatives], dtype=object)
    labels = np.asarray([1] * len(positives) + [0] * len(negatives), dtype="int64")

    X = features_for_pairs(bundle, rows_u, rows_c)
    out = pd.DataFrame({"user_id": rows_u, "stock_code": rows_c})
    out = pd.concat([out, X.reset_index(drop=True)], axis=1)
    out["label"] = labels
    out.attrs["stats"] = {
        "n_users_total": len(cand_users),
        "n_users_with_pos": len([u for u in cand_users if pos_by_user.get(u, 0) > 0]),
        "pos_n": len(positives), "neg_n": len(negatives),
        "cut": str(cut.date()),
    }
    return out


# ---------------------------------------------------------------- 评测
def user_split(users: np.ndarray, ratio: float = 0.3, seed: int = SEED):
    """用户级 70/30（仿 scripts/16）：整用户不跨切。返回 (fit_mask, valid_mask)。"""
    rng = np.random.default_rng(seed)
    uniq = np.sort(np.unique(users))
    perm = uniq[rng.permutation(len(uniq))]
    n_valid = int(round(len(perm) * ratio))
    valid_set = set(int(x) for x in perm[:n_valid])
    fit_mask = np.asarray([int(u) not in valid_set for u in users], dtype=bool)
    valid_mask = ~fit_mask
    return fit_mask, valid_mask


def rank_metrics(group, score_name: str = "score", ks=(5, 10)) -> dict:
    """对已带 score 与 label 的 DataFrame 算每用户 top-k macro 指标。"""
    import math
    out: dict[str, float] = {}
    for k in ks:
        hits = prec = rec = cnt = 0.0
        for _u, g in group.groupby("user_id", sort=False):
            g = g.sort_values(score_name, ascending=False)
            top = set(g["stock_code"].head(k))
            pos = set(g.loc[g["label"] == 1, "stock_code"])
            if not pos:
                continue
            cnt += 1
            hits += 1.0 if top & pos else 0.0
            prec += len(top & pos) / min(k, len(g))
            rec += len(top & pos) / len(pos)
        out[f"hit@{k}"] = hits / max(1.0, cnt)
        out[f"prec@{k}"] = prec / max(1.0, cnt)
        out[f"rec@{k}"] = rec / max(1.0, cnt)
    return out
