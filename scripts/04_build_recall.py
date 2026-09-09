"""5 路召回融合（阶段四精简指令）。

输入：
    data/processed/clean_transactions.csv  清洗后交易（用户历史 / 全局销量 / 最近会话）
    models/item2vec.model                  商品向量（Skip-gram，scripts/10 产物）

5 路召回（每路：user_id → 商品码列表，已过滤该用户已购）：
    1) popular   全局销量 Top-N
    2) similar   用户历史每件商品取 Item2Vec Top-5，合并
    3) user_vec  历史商品向量平均 → 与全词表余弦近邻
    4) itemcf    阶段三 ItemCF 相似（同一张「用户×商品」矩阵现算），取历史商品相似品合并
    5) recent    最近一张发票内商品做 Item2Vec 扩展合并

融合：按路序遍历，每路最多取满配额、跳过重复 ID、满 50 停止。
冷启动（无购买历史）：直接返回热门路并标记 is_cold_start=True。

配额写在 config/settings.py 的 HYPERPARAMS["recall5"]（种子固定 42）。

产物：
    models/user_vectors.pkl               {user_id: 单位平均向量(float32)}
    data/processed/recall_set_5route.csv  user_id, recall_list(逗号分隔), list_length
    outputs/stage04_coverage.txt          覆盖率 + 热门占比 + 冷启动统计
    outputs/stage04_quota_comparison.txt  默认配额 vs 调大热门配额对比

运行：venv\\Scripts\\python.exe scripts\\04_build_recall.py
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from config.settings import COLUMNS, HYPERPARAMS, PATHS, SEED, set_seed
from src.embedding import top_k_similar_from_vectors
from src.item_similarity import load_item_matrix, top_k_similar_rows
from src.recommender import load_clean

set_seed(SEED)

_USER = COLUMNS["user"]
_ITEM = COLUMNS["item"]
_QTY = COLUMNS["qty"]
_INVOICE = COLUMNS["invoice"]
_TIME = COLUMNS["time"]

CFG = HYPERPARAMS["recall5"]
FINAL_K = int(CFG["final_k"])
HOT_TOP_N = int(CFG["hot_top_n"])
ROUTE_ORDER = list(CFG["route_order"])
QUOTAS = dict(CFG["quotas"])
ALT_QUOTAS = dict(CFG["alt_quotas"])
NEIGHBOR_K = int(CFG["neighbor_k"])

RECALL_CSV = PROJECT_ROOT / "data" / "processed" / "recall_set_5route.csv"
USER_VEC_PKL = PROJECT_ROOT / "models" / "user_vectors.pkl"
COVERAGE_TXT = PROJECT_ROOT / "outputs" / "stage04_coverage.txt"
QUOTA_TXT = PROJECT_ROOT / "outputs" / "stage04_quota_comparison.txt"


# --------------------------------------------------------------------------- #
# 扩展合并：把一批「种子商品」的相似邻居按最优分数降序去重
# --------------------------------------------------------------------------- #
def expand_merge(
    neighbors: dict[str, list[tuple[str, float]]],
    seeds: set[str] | list[str],
    bought: set[str],
    cap: int,
) -> list[str]:
    """对每个种子取其邻居表，合并邻居出现的最优分数 → 降序、过滤已购、截断 cap。"""
    best: dict[str, float] = {}
    for seed in seeds:
        for nb, sc in neighbors.get(seed, ()):
            if nb in bought:
                continue
            if sc > best.get(nb, -1.0):
                best[nb] = sc
    order = sorted(best, key=lambda c: (-best[c], c))
    return order[:cap]


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    t0 = time.time()
    print("=" * 72)
    print("阶段四 · 5 路召回融合（默认配额 10/10/15/10/5，seed=42）")
    print("=" * 72)

    df = load_clean()
    df[_INVOICE] = df[_INVOICE].astype(str)
    users = sorted(int(u) for u in df[_USER].unique())
    library = sorted(str(x) for x in df[_ITEM].unique())
    n_users, n_lib = len(users), len(library)
    print(f"清洗数据：{len(df):,} 行 / 用户 {n_users:,} / 商品库 {n_lib:,}")

    # ---- 已购集合 + 最近一张发票的商品 -------------------------------- #
    purchased: dict[int, set[str]] = {
        int(u): set(map(str, g[_ITEM].unique()))
        for u, g in df.groupby(_USER)
    }
    df_sorted = df.sort_values([_USER, _TIME, _INVOICE], kind="mergesort")
    last_invoice = df_sorted.groupby(_USER)[_INVOICE].last()
    invoice_items = df.groupby(_INVOICE, sort=False)[_ITEM].agg(
        lambda s: set(map(str, s))
    )
    recent_items = {int(u): invoice_items[inv] for u, inv in last_invoice.items()}

    # ---- 全局销量：热门序 --------------------------------------------- #
    sales = df.groupby(_ITEM)[_QTY].sum().sort_values(ascending=False)
    pop_order = [str(x) for x in sales.index]
    pop_top_hot = set(pop_order[:HOT_TOP_N])

    # ---- Item2Vec 词表与邻居表（第 2/5 路共用）------------------------- #
    w2v = Word2Vec.load(str(PATHS["item2vec_model"]))
    keys = [str(k) for k in w2v.wv.index_to_key]
    vecs = np.asarray(w2v.wv.vectors, dtype=np.float64)
    vecset = set(keys)
    norms = np.linalg.norm(vecs, axis=1)
    norms[norms == 0] = 1.0
    unit = vecs / norms[:, None]
    i2v_nn = top_k_similar_from_vectors(keys, vecs, top_k=NEIGHBOR_K)
    print(f"Item2Vec 词表：{len(keys):,}（商品库 {n_lib:,} 中有 {len(vecset & set(library)):,} 在词表）")

    # ---- ItemCF 邻居表（第 4 路，现算，与阶段三同口径）----------------- #
    t_it = time.time()
    item_matrix, item_ids = load_item_matrix(df)  # 商品 × 用户
    itemcf_nn = top_k_similar_rows(
        item_matrix, item_ids, top_k=NEIGHBOR_K, batch=256
    )
    print(f"ItemCF 邻居表：{len(itemcf_nn):,} 件商品（{time.time()-t_it:.0f}s）")

    # ---- 每用户 5 路候选（一次性构建，供默认/对照两组配额复用）--------- #
    print("逐用户生成 5 路候选并融合 ...")
    rows_default: list[dict] = []
    rows_alt: list[dict] = []
    user_vec_out: dict[int, np.ndarray] = {}
    n_cold = 0
    n_full = 0

    for u in users:
        bought = purchased[u]
        recent = recent_items.get(u, set())

        # 冷启动：无任何购买历史 → 直接热门路
        if not bought:
            n_cold += 1
            cand = pop_order[:FINAL_K]
            for rows, quotas in ((rows_default, QUOTAS), (rows_alt, ALT_QUOTAS)):
                rows.append(_pack(u, cand, is_cold=True, quota_name="popular"))
            user_vec_out[u] = np.zeros(unit.shape[1], dtype=np.float32)
            continue

        # ① 热门路
        cand_pop = [c for c in pop_order if c not in bought][:FINAL_K]
        # ② 相似扩展（Item2Vec 邻居）
        seeds_vocab = [c for c in bought if c in vecset]
        cand_sim = expand_merge(i2v_nn, seeds_vocab, bought, FINAL_K)
        # ③ 用户向量 KNN（词表内）
        if seeds_vocab:
            uv_raw = vecs[[keys.index(c) for c in seeds_vocab]].mean(axis=0)
            uv = uv_raw / (np.linalg.norm(uv_raw) or 1.0)
        else:
            uv = None
        user_vec_out[u] = (
            uv.astype(np.float32) if uv is not None else np.zeros(unit.shape[1], dtype=np.float32)
        )
        cand_vec: list[str] = []
        if uv is not None:
            sims = unit @ uv  # 词表内余弦
            for pos in np.argsort(-sims):
                code = keys[int(pos)]
                if code not in bought:
                    cand_vec.append(code)
                    if len(cand_vec) >= FINAL_K:
                        break
        # ④ ItemCF 相似品合并
        seeds_lib = [c for c in bought if c in itemcf_nn]
        cand_cf = expand_merge(itemcf_nn, seeds_lib, bought, FINAL_K)
        # ⑤ 最近会话扩展
        recent_vocab = [c for c in recent if c in vecset]
        cand_recent = expand_merge(i2v_nn, recent_vocab, bought, FINAL_K)

        cands = {
            "popular": cand_pop,
            "similar": cand_sim,
            "user_vec": cand_vec,
            "itemcf": cand_cf,
            "recent": cand_recent,
        }

        default = _fuse(cands, QUOTAS, bought, FINAL_K)
        alt = _fuse(cands, ALT_QUOTAS, bought, FINAL_K)
        rows_default.append(_pack(u, default["list"], is_cold=False))
        rows_alt.append(_pack(u, alt["list"], is_cold=False))
        if len(default["list"]) >= FINAL_K:
            n_full += 1

    # ---- 落盘 ---------------------------------------------------------- #
    RECALL_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_default).to_csv(
        RECALL_CSV, index=False, encoding="utf-8"
    )
    with open(USER_VEC_PKL, "wb") as f:
        pickle.dump(user_vec_out, f, protocol=4)

    # ---- 覆盖/热门统计（默认配额）------------------------------------- #
    flat = [c for r in rows_default for c in r["recall_list"].split(",")]
    cov_lines = _coverage_block(
        "默认配额 10/10/15/10/5",
        rows_default, flat, set(library), pop_top_hot, n_users, n_full, n_cold,
    )
    demo_user = max(users, key=lambda u: len(purchased[u]))
    cov_lines.append("")
    cov_lines.append("示例（历史最丰富用户 %d，默认配额 Top10）：" % demo_user)
    demo = next(r for r in rows_default if r["user_id"] == demo_user)
    cov_lines.append("  " + ", ".join(demo["recall_list"].split(",")[:10]))
    COVERAGE_TXT.write_text("\n".join(cov_lines) + "\n", encoding="utf-8")

    # ---- 配额对比（默认 vs 调大热门）---------------------------------- #
    flat_alt = [c for r in rows_alt for c in r["recall_list"].split(",")]
    quo_lines = [
        "# 配额对比：默认配额 vs 调大热门配额（同一批用户、同一批候选池）",
        "# " + CFG["alt_note"],
        "",
    ]
    quo_lines += _metrics_lines("默认配额", rows_default, flat, set(library), pop_top_hot)
    quo_lines += _metrics_lines("调大热门", rows_alt, flat_alt, set(library), pop_top_hot)
    cov_d = _coverage(flat, set(library), pop_top_hot)
    cov_a = _coverage(flat_alt, set(library), pop_top_hot)
    avg_d = sum(r["list_length"] for r in rows_default) / max(1, len(rows_default))
    avg_a = sum(r["list_length"] for r in rows_alt) / max(1, len(rows_alt))
    quo_lines.append(
        "结论：调大热门配额后，热门占比 %.1f%%→%.1f%%；商品覆盖率 %.1f%%→%.1f%%；"
        "平均候选 %.1f→%.1f。"
        % (cov_d["hot_share"] * 100, cov_a["hot_share"] * 100,
           cov_d["coverage"] * 100, cov_a["coverage"] * 100,
           avg_d, avg_a)
    )
    QUOTA_TXT.write_text("\n".join(quo_lines) + "\n", encoding="utf-8")

    print("\n".join(cov_lines))
    print(f"\n产物：\n  {USER_VEC_PKL}\n  {RECALL_CSV}\n  {COVERAGE_TXT}\n  {QUOTA_TXT}")
    print(f"总耗时 {time.time()-t0:.0f}s")


# --------------------------------------------------------------------------- #
# 融合 / 统计 / 落盘辅助
# --------------------------------------------------------------------------- #
def _fuse(
    cands: dict[str, list[str]],
    quotas: dict[str, int],
    bought: set[str],
    final_k: int,
) -> dict:
    """按路序融合：每路最多取满配额，跳过已购/重复，满 final_k 停止。"""
    seen = set(bought)
    out: list[str] = []
    for name in ROUTE_ORDER:
        added = 0
        q = quotas.get(name, 0)
        for code in cands.get(name, ()):
            if code in seen:
                continue
            seen.add(code)
            out.append(code)
            added += 1
            if added >= q or len(out) >= final_k:
                break
        if len(out) >= final_k:
            break
    return {"list": out}


def _pack(user_id: int, recall: list[str], *, is_cold: bool,
          quota_name: str = "") -> dict:
    return {
        "user_id": user_id,
        "recall_list": ",".join(recall),
        "list_length": len(recall),
    }


def _coverage(flat: list[str], library: set[str], hot: set[str]) -> dict:
    unique = set(flat)
    hot_hit = sum(1 for c in flat if c in hot)
    return {
        "n_rows": len(flat),
        "n_unique": len(unique),
        "coverage": len(unique & library) / max(1, len(library)),
        "hot_share": hot_hit / max(1, len(flat)),
    }


def _metrics_lines(tag: str, rows: list[dict], flat: list[str],
                   library: set[str], hot: set[str]) -> list[str]:
    c = _coverage(flat, library, hot)
    lens = [r["list_length"] for r in rows]
    avg = sum(lens) / max(1, len(lens))
    full = sum(1 for x in lens if x >= FINAL_K)
    return [
        f"[{tag}] 用户数={len(rows):,}  召回总条目={c['n_rows']:,}  "
        f"去重商品={c['n_unique']:,}",
        f"   覆盖率(去重召回/商品库)={c['coverage']*100:.1f}%   "
        f"热门占比(命中全局Top{HOT_TOP_N})={c['hot_share']*100:.1f}%",
        f"   平均 list_length={avg:.1f}   满{FINAL_K}用户占比={full/len(rows)*100:.1f}%",
        "",
    ]


def _coverage_block(tag: str, rows, flat, library, hot, n_users, n_full, n_cold) -> list[str]:
    c = _coverage(flat, library, hot)
    lens = [r["list_length"] for r in rows]
    avg = sum(lens) / max(1, len(lens))
    return [
        "# 阶段四 · 5 路召回融合 覆盖率报告",
        f"# {tag}  seed={SEED}  N={n_users} 用户  商品库={len(library)}",
        "",
        f"召回总条目           {c['n_rows']:,}",
        f"去重召回商品数       {c['n_unique']:,}",
        f"覆盖率(去重/商品库)  {c['coverage']*100:.2f}%",
        f"热门占比(Top{HOT_TOP_N})   {c['hot_share']*100:.2f}%",
        f"平均 list_length     {avg:.1f}",
        f"满{FINAL_K}用户占比       {n_full/n_users*100:.1f}%",
        f"冷启动用户数         {n_cold}（离线全员有购买，恒 0；在线新用户走热门路并置 is_cold_start=True）",
    ]


if __name__ == "__main__":
    main()
