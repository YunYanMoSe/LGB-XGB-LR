"""阶段六链路 · 多路召回融合 + 阶段四指标 + 实验 A（指导书阶段三/四）。

防泄漏口径（沿用 scripts/05、08，与 scripts/12 可逐格对齐）：
    每用户「最后一笔订单」= 测试篮，更早历史 = 训练。
    商品向量只用训练期订单现重训一份 Skip-gram（词表约 3,046），
    用户向量 / 商品热度 / ItemCF 画像全部只由训练期数据计算。

产出：
1. `data/processed/recall_set.csv` —— 三路召回（热门/相似扩展/用户向量）按配额
   popular10+similar20+user_vec20 融合去重，不足 50 用热门回填；每用户最多 50 行，
   含路贡献(routes)、综合分数(score)、是否多路命中、是否落在训练期全局 Top-100 热门。
2. `outputs/chain/recall_phase4.txt|.json` —— 阶段四指标：
   召回命中率 recall@50、命中用户占比、召回集商品覆盖率、热门占比、多路命中占比、各路配额达成。
3. `outputs/chain/experiment_a.txt|.json` —— 实验 A：同一批测试用户上
   热门-only vs ItemCF-only 的 Precision@5/10、recall@10、hit@10 与 Top10 列表重叠
   （Jaccard 中位/均值），另附 2 个案例用户的 Top10 并排（✔=命中测试篮）。

运行：venv\\Scripts\\python.exe scripts\\11_recall_fusion.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config.settings import COLUMNS, PATHS, SEED, set_seed
from src import recommender as rec

set_seed(SEED)

_USER = COLUMNS["user"]
CHAIN = PROJECT_ROOT / "outputs" / "chain"
CHAIN.mkdir(parents=True, exist_ok=True)
VEC_CACHE = CHAIN / "vectors_train_skip.npz"

FINAL_K = rec._FINAL_K
TOP_K = [5, 10]
TOP100_HOT = 100


def load_and_split():
    df = rec.load_clean()
    train, test = rec.split_last_invoice(df)
    print(f"清洗后 {len(df):,} 行；测试篮用户 {test[_USER].nunique():,}；训练 {len(train):,} 行")
    return df, train, test


def phase4_metrics(ctx, test_map, recalls) -> dict:
    """recalls: user -> 有序候选 code 列表（fusion 结果）。"""
    col_counts = np.asarray(ctx.B.sum(axis=0)).ravel()
    top100 = {ctx.code_of[i] for i in np.argsort(-col_counts)[:TOP100_HOT]}

    hit_rates, covered, recall50s = [], 0, []
    rows_all = 0
    n_multi = 0
    hot_rows = 0
    distinct = set()
    quota_ok = {"popular": 0, "similar": 0, "user_vec": 0}
    n_users = len(recalls)

    for u, cands in recalls.items():
        basket = test_map[u]
        codes = [c["code"] for c in cands]
        hit = set(codes[:FINAL_K]) & basket
        recall50s.append(len(hit) / max(1, len(basket)))
        covered += 1 if hit else 0
        for c in cands:
            rows_all += 1
            distinct.add(c["code"])
            if c["code"] in top100:
                hot_rows += 1
            if len(c["routes"]) >= 2:
                n_multi += 1
            for r in c["routes"]:
                pass
        # 各路配额达成检查（fusion 里该路至少投进一条候选）
        rt = set(r for c in cands for r in c["routes"])
        for r in quota_ok:
            if r in rt:
                quota_ok[r] += 1

    n_items_train = ctx.item_codes.shape[0]
    return {
        "test_users": n_users,
        "recall@50": float(np.mean(recall50s)),
        "命中用户占比": float(covered / max(1, n_users)),
        "平均候选行数/用户": float(rows_all / max(1, n_users)),
        "召回集商品覆盖率": float(len(distinct) / max(1, n_items_train)),
        "去重召回商品数": len(distinct),
        "训练期商品库": int(n_items_train),
        "热门占比": float(hot_rows / max(1, rows_all)),
        "多路命中占比": float(n_multi / max(1, rows_all)),
        "各路由至少一条候选的用户占比": {r: float(v / max(1, n_users)) for r, v in quota_ok.items()},
    }


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a[:10]), set(b[:10])
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def experiment_a(ctx, test_map, users) -> dict:
    """热门-only vs ItemCF-only，Precision@5/10、recall@10、hit@10 + Top10 重叠。"""
    from config.settings import COLUMNS
    metric_rows: list[dict] = []
    pop_top10 = {}
    cf_top10 = {}
    for u in users:
        pop_top10[u] = [c for c, _ in rec.recall_popular_only(ctx, u, 10)]
        cf_top10[u] = [c for c, _ in rec.recall_itemcf_only(ctx, u, 10)]

    def metrics(topns: dict) -> dict:
        agg = {f"precision@{k}": [] for k in TOP_K}
        agg.update({f"recall@{k}": [] for k in TOP_K})
        hit10 = []
        for u in users:
            basket = test_map[u]
            ranked = topns[u]
            for k in TOP_K:
                hits = len(set(ranked[:k]) & basket)
                agg[f"recall@{k}"].append(hits / max(1, len(basket)))
                agg[f"precision@{k}"].append(hits / k)
            hit10.append(1.0 if set(ranked[:10]) & basket else 0.0)
        out = {k: float(np.mean(v)) for k, v in agg.items()}
        out["hit@10"] = float(np.mean(hit10))
        return out

    for name, topns in (("全局热门", pop_top10), ("ItemCF", cf_top10)):
        metric_rows.append({"method": name, **metrics(topns)})

    ov = [jaccard(pop_top10[u], cf_top10[u]) for u in users]
    metric_rows.append({"method": "重叠", "Top10 Jaccard中位": float(np.median(ov)),
                        "Top10 Jaccard均值": float(np.mean(ov))})
    return {"metrics": metric_rows, "pop_top10": pop_top10, "cf_top10": cf_top10}


def pick_cases(ctx, test_map, users, pop_top10, cf_top10) -> list[dict]:
    """选 2 个测试篮大小 3~30 的用户作案例：优先 ItemCF 命中而热门未命中的。"""
    cands = []
    for u in users:
        size = len(test_map[u])
        if 3 <= size <= 30:
            cf_hit = bool(set(cf_top10[u]) & test_map[u])
            pop_hit = bool(set(pop_top10[u]) & test_map[u])
            cands.append((u, size, cf_hit, pop_hit))
    cands.sort(key=lambda t: (t[2] and not t[3], -t[1], t[0]))
    chosen = cands[:2] if len(cands) >= 2 else cands
    out = []
    for u, size, cf_hit, pop_hit in chosen:
        out.append({"user": int(u), "test_basket_size": int(size),
                    "pop_hit": bool(pop_hit), "itemcf_hit": bool(cf_hit),
                    "test_basket": sorted(test_map[u]),
                    "popular_top10": pop_top10[u], "itemcf_top10": cf_top10[u]})
    return out


def main() -> None:
    t0 = time.time()
    df, train, test = load_and_split()

    # 无泄漏 Skip-gram 向量（缓存供 scripts/12 复用）
    keys, vecs = rec.no_leak_skip_vectors(df, cache_path=VEC_CACHE)
    ctx = rec.build_ctx(train, keys, vecs)
    print(f"训练期上下文：{len(ctx.users):,} 用户 / {ctx.item_codes.shape[0]} 商品 / 词表 {len(keys)}")

    test_users = sorted(set(int(x) for x in ctx.users)
                        & set(int(x) for x in test[_USER]))
    test_map = rec.test_basket_map(test)
    print(f"参与评测的测试用户（训练期与测试期都有购买）：{len(test_users):,}")

    # ---- 三路召回融合 ----
    recalls: dict[int, list[dict]] = {}
    for u in test_users:
        recalls[u] = rec.recall_fusion(ctx, u, final_k=FINAL_K)
    rec_rows = []
    for u in test_users:
        for rank, c in enumerate(recalls[u], start=1):
            rec_rows.append({
                "user_id": int(u), "item": c["code"],
                "recall_rank": rank, "routes": "|".join(c["routes"]),
                "score": round(float(c["score"]), 5),
                "n_routes": len(c["routes"]),
            })
    rec_df = pd.DataFrame(rec_rows)
    rec_df.to_csv(PATHS["recall_set"], index=False, encoding="utf-8")
    print(f"[recall_set] {PATHS['recall_set']}  {len(rec_df):,} 行 / {rec_df['user_id'].nunique()} 用户")

    # ---- 阶段四指标 ----
    p4 = phase4_metrics(ctx, test_map, recalls)
    (CHAIN / "recall_phase4.json").write_text(
        json.dumps({"phase4": p4, "recall_set": str(PATHS["recall_set"]),
                    "quotas": {"popular": 10, "similar": 20, "user_vec": 20},
                    "final_k": FINAL_K}, ensure_ascii=False, indent=2), encoding="utf-8")
    _skip = "各路由至少一条候选的用户占比"
    lines = ["阶段四 · 融合召回结果与覆盖率（召回集=recall_set.csv）", "=" * 60]
    for k, v in p4.items():
        if k == _skip:
            continue
        lines.append(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    lines.append(f"{_skip}: " + "，".join(
        f"{r}={v:.2f}" for r, v in p4[_skip].items()))
    (CHAIN / "recall_phase4.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # ---- 实验 A ----
    exp = experiment_a(ctx, test_map, test_users)
    cases = pick_cases(ctx, test_map, test_users, exp["pop_top10"], exp["cf_top10"])
    exp["cases"] = cases
    (CHAIN / "experiment_a.json").write_text(
        json.dumps(exp, ensure_ascii=False, indent=2), encoding="utf-8")

    out = ["实验 A · 热门 vs ItemCF（同一批测试用户，Top10 补全）", "=" * 78]
    m = {r["method"]: r for r in exp["metrics"]}
    head = ["方法", "precision@5", "precision@10", "recall@10", "hit@10"]
    out.append(f"{head[0]:<10}" + "".join(f"{h:>14}" for h in head[1:]))
    for meth in ("全局热门", "ItemCF"):
        r = m[meth]
        out.append(f"{meth:<10}" + "".join(f"{r.get(h, 0):>14.4f}" for h in head[1:]))
    o = m["重叠"]
    out.append(f"Top10重叠   Jaccard 中位 {o['Top10 Jaccard中位']:.4f} / 均值 {o['Top10 Jaccard均值']:.4f}")
    out.append("")
    for case in cases:
        basket = set(case["test_basket"])
        out.append(f"案例用户 {case['user']}（测试篮 {case['test_basket_size']} 种商品，"
                   f"热门命中={case['pop_hit']}，ItemCF命中={case['itemcf_hit']}）")
        out.append(f"  测试篮: {','.join(sorted(basket))}")
        for label, top in (("  热门Top10", case["popular_top10"]),
                           ("  ItemCF Top10", case["itemcf_top10"])):
            mark = lambda c: c if c not in basket else f"{c}✔"
            out.append(f"{label}: " + ", ".join(mark(c) for c in top))
        out.append("")
    (CHAIN / "experiment_a.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))

    print(f"\n总耗时 {time.time()-t0:.0f}s → 见 outputs/chain/recall_phase4.* 与 experiment_a.*")


if __name__ == "__main__":
    main()
