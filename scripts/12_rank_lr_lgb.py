"""阶段六链路 · 召回候选精排（LR + LightGBM）与阶段五/六汇总（指导书阶段五/六）。

协议（与 scripts/11 完全一致，数字可逐格对齐）：
    每用户「最后一笔订单」= 测试篮，更早 = 训练；特征全部由训练期数据计算；
    训练用 Skip-gram 商品向量只用训练期订单重训（读取 outputs/chain/vectors_train_skip.npz
    缓存，无测试篮泄漏）。标签 = 该候选是否落在该用户测试篮里。

流程：
1. 读 recall_set.csv，对 (user,item,routes) 提共享特征矩阵（src/recommender.FEATURES）。
2. 用户级切分 fit/valid（seed=42，避免同一用户的候选跨集泄漏）。
3. LR（StandardScaler + LogisticRegression, class_weight=balanced）→ valid AUC + 系数表；
   LightGBM（同特征同候选）→ valid AUC + gain 重要性；
   两模型每用户 Top10 的 Jaccard 重叠。
4. 汇总指标表（同一批 valid 用户）：热门 top10 / 融合召回 top10 / +LR top10 / +LGB top10
   的 precision@5/10、recall@10、hit@10。
5. 成功 / 失败案例各 1 + 演示用 2 用户，落 JSON。
6. 保存 models/lr_model.pkl(Pipeline)、models/lgb_model.pkl、models/rank_meta.json、
   models/demo_ids.json（含在线排序函数 score_rank 的用法说明）。

运行：venv\\Scripts\\python.exe scripts\\12_rank_lr_lgb.py
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

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import COLUMNS, HYPERPARAMS, PATHS, SEED, set_seed
from src import recommender as rec

set_seed(SEED)

_USER = COLUMNS["user"]
CHAIN = PROJECT_ROOT / "outputs" / "chain"
CHAIN.mkdir(parents=True, exist_ok=True)
VEC_CACHE = CHAIN / "vectors_train_skip.npz"
TOP_K = [5, 10]
VALID_RATIO = 0.3
DEMO_USERS_N = 2
LGB_TRY = True

# 特征显示名 + 一句话含义（供系数表 / Word 素材）
FEAT_ZH = {
    "pop_buyers_log": "商品热度：训练期购买用户数(log1p)",
    "pop_rank_frac": "热门排名位置(0=最热→1=最冷)",
    "price_log": "商品单价中位数(log1p)",
    "emb_sim_uservec": "与用户历史商品向量的余弦(嵌入语义)",
    "cf_agg_sim": "与用户已购集合的 ItemCF 聚合余弦(画像)",
    "n_routes": "召回命中路由数(0~3)",
    "fl_pop": "是否热门路召回",
    "fl_sim": "是否相似扩展路召回",
    "fl_vec": "是否用户向量路召回",
    "user_n_owned_log": "用户历史去重购买品类数(log1p)",
}


def build_features(ctx, test_map, test_users) -> pd.DataFrame:
    """按 recall_set 行序构建特征矩阵与标签。"""
    rows = pd.read_csv(PATHS["recall_set"], dtype={"user_id": "int64", "item": str})
    keep = rows["user_id"].isin(test_users)
    rows = rows[keep].reset_index(drop=True)
    feat_rows = [(int(r.user_id), r.item, str(r.routes).split("|"))
                 for r in rows.itertuples()]
    X = rec.features_for_rows(feat_rows, ctx)
    labels = np.array([
        1.0 if rows.at[i, "item"] in test_map[int(rows.at[i, "user_id"])] else 0.0
        for i in range(len(rows))
    ], dtype=np.float64)
    return rows, X, labels


def split_users(test_users, ratio: float = VALID_RATIO):
    rng = np.random.default_rng(SEED)
    users = np.asarray(sorted(test_users))
    rng.shuffle(users)
    n_valid = max(1, int(round(len(users) * ratio)))
    valid, fit = set(users[:n_valid]), set(users[n_valid:])
    return fit, valid


def auc_on(valid_users, rows, X, labels, clf, scaler=None):
    m = rows["user_id"].isin(valid_users).to_numpy()
    if scaler is not None:
        Xv = scaler.transform(X[m])
    else:
        Xv = X[m]
    p = clf.predict_proba(Xv)[:, 1]
    if labels[m].sum() == 0 or (labels[m] == 0).sum() == 0:
        return float("nan")
    return float(roc_auc_score(labels[m], p))


def top10_metrics(valid_users, test_map, ranked_by_user) -> dict:
    agg = {f"precision@{k}": [] for k in TOP_K}
    agg.update({f"recall@{k}": [] for k in TOP_K})
    hit10 = []
    for u in valid_users:
        basket = test_map[u]
        rk = ranked_by_user[u]
        for k in TOP_K:
            hits = len(set(rk[:k]) & basket)
            agg[f"recall@{k}"].append(hits / max(1, len(basket)))
            agg[f"precision@{k}"].append(hits / k)
        hit10.append(1.0 if set(rk[:10]) & basket else 0.0)
    out = {k: float(np.mean(v)) for k, v in agg.items()}
    out["hit@10"] = float(np.mean(hit10))
    return out


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def coef_table(lr_pipe, X, labels) -> list[dict]:
    scaler = lr_pipe.named_steps["scaler"]
    clf = lr_pipe.named_steps["lr"]
    names = rec.FEATURES
    coef = np.asarray(clf.coef_).ravel()
    # 系数已作用在标准化后的特征上（可直接比较权重），并给“每 +1 标准差 → logit 变化”
    rows_out = []
    for i, name in enumerate(names):
        rows_out.append({
            "feature": name, "zh": FEAT_ZH[name],
            "coef": round(float(coef[i]), 4),
            "coef_std_abs": round(float(np.abs(coef[i])), 4),
        })
    rows_out.sort(key=lambda r: -r["coef_std_abs"])
    return rows_out


def main() -> None:
    t0 = time.time()
    df = rec.load_clean()
    train, test = rec.split_last_invoice(df)
    keys, vecs = rec.no_leak_skip_vectors(df, cache_path=VEC_CACHE)
    ctx = rec.build_ctx(train, keys, vecs)
    test_users = sorted(set(int(x) for x in ctx.users)
                        & set(int(x) for x in test[_USER]))
    test_map = rec.test_basket_map(test)
    print(f"上下文：{len(ctx.users):,} 用户 / {ctx.item_codes.shape[0]} 商品；测试用户 {len(test_users):,}")

    rows, X, labels = build_features(ctx, test_map, test_users)
    print(f"候选样本：{len(rows):,} 行；正样本(命中测试篮) {int(labels.sum()):,} "
          f"({labels.mean()*100:.2f}%)")
    fit_users, valid_users = split_users(test_users)
    print(f"用户级切分(seed={SEED})：fit {len(fit_users)} / valid {len(valid_users)}")

    m_fit = rows["user_id"].isin(fit_users).to_numpy()
    m_val = rows["user_id"].isin(valid_users).to_numpy()

    # ---- LR ----
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", C=1.0, max_iter=3000,
                                  random_state=SEED)),
    ])
    lr_pipe.fit(X[m_fit], labels[m_fit])
    auc_lr = auc_on(valid_users, rows, X, labels, lr_pipe)
    print(f"[LR] valid AUC = {auc_lr:.4f}")

    # ---- LightGBM ----
    auc_lgb, lgb_model, importances = None, None, []
    lgb_enabled = bool(LGB_TRY)
    if lgb_enabled:
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:  # 铁律：装不了就写明原因，不伪造
            lgb_enabled = False
            print(f"[LightGBM] 导入失败：{exc}；跳过，在报告里写文字说明")
        else:
            n_pos = int(labels[m_fit].sum())
            n_neg = int((labels[m_fit] == 0).sum())
            lgb_model = LGBMClassifier(
                n_estimators=300, learning_rate=0.1, num_leaves=31,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=n_neg / max(1, n_pos),
                random_state=SEED, verbosity=-1,
            )
            lgb_model.fit(X[m_fit], labels[m_fit])
            auc_lgb = auc_on(valid_users, rows, X, labels, lgb_model)
            print(f"[LightGBM] valid AUC = {auc_lgb:.4f}")
            gains = lgb_model.booster_.feature_importance(importance_type="gain")
            importances = sorted(
                ({"feature": f, "zh": FEAT_ZH[f], "gain": round(float(g), 1)}
                 for f, g in zip(rec.FEATURES, gains)),
                key=lambda r: -r["gain"])

    # ---- 每用户 Top10（valid 子集内四种方法一致对比）----
    lr_top10 = {u: [c for c, _ in rec.score_rank(ctx, u, lr_pipe)[:10]] for u in valid_users}
    lgb_top10 = None
    if lgb_model is not None:
        lgb_top10 = {u: [c for c, _ in rec.score_rank(ctx, u, lgb_model)[:10]]
                     for u in valid_users}
    fusion_top10 = {}
    pop_top10 = {}
    for u in valid_users:
        rows_u = rows[rows["user_id"] == u].sort_values("recall_rank")
        fusion_top10[u] = rows_u["item"].head(10).tolist()
        pop_top10[u] = [c for c, _ in rec.recall_popular_only(ctx, u, 10)]

    summary = {
        "valid_users": len(valid_users),
        "LR": top10_metrics(valid_users, test_map, lr_top10),
        "融合召回": top10_metrics(valid_users, test_map, fusion_top10),
        "全局热门": top10_metrics(valid_users, test_map, pop_top10),
    }
    if lgb_top10 is not None:
        summary["LGB"] = top10_metrics(valid_users, test_map, lgb_top10)
        ov = [jaccard(lr_top10[u], lgb_top10[u]) for u in valid_users]
    else:
        ov = []
    (CHAIN / "metrics_summary.json").write_text(
        json.dumps({"valid_users": len(valid_users), "protocol": "每用户最后一单=测试篮",
                    "methods": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n汇总指标表（valid 用户", len(valid_users), "人，同一候选池、同一测试篮）")
    hdr = ["方法", "precision@5", "precision@10", "recall@10", "hit@10"]
    print(f"{hdr[0]:<8}" + "".join(f"{h:>13}" for h in hdr[1:]))
    for meth in ("全局热门", "融合召回", "LR", "LGB"):
        if meth in summary:
            r = summary[meth]
            print(f"{meth:<8}" + "".join(f"{r.get(h, 0):>13.4f}" for h in hdr[1:]))

    # ---- 成功/失败案例与演示用户 ----
    def lr_rank_full(u):
        return rec.score_rank(ctx, u, lr_pipe)

    cases, demo = pick_cases(lr_rank_full, lr_top10, test_map, valid_users)
    demo_users = demo

    # ---- 存档 ----
    meta = {
        "protocol": "每用户最后一单=测试篮；特征/向量只用训练期（详见脚本注释与报告）",
        "features": [{"name": f, "zh": FEAT_ZH[f]} for f in rec.FEATURES],
        "n_candidates": int(len(rows)), "n_pos": int(labels.sum()),
        "n_fit_users": len(fit_users), "n_valid_users": len(valid_users),
        "seed": SEED, "valid_ratio": VALID_RATIO,
        "topn": HYPERPARAMS["rank"]["topn"],
        "lr_auc": auc_lr, "lgb_auc": auc_lgb,
        "lr_coef": coef_table(lr_pipe, X, labels),
        "lgb_gain": importances,
        "lgb_fitted": bool(lgb_model is not None),
        "pos_neg_feat_mean": {
            "pos": {f: round(float(X[labels == 1, i].mean()), 4)
                    for i, f in enumerate(rec.FEATURES)},
            "neg": {f: round(float(X[labels == 0, i].mean()), 4)
                    for i, f in enumerate(rec.FEATURES)},
        },
        "lr_vs_lgb_top10_jaccard": {
            "median": float(np.median(ov)) if ov else None,
            "mean": float(np.mean(ov)) if ov else None,
        },
        "success_case": cases[0] if cases else None,
        "failure_case": cases[1] if len(cases) > 1 else None,
    }
    joblib.dump(lr_pipe, str(PATHS["models_dir"] / "lr_model.pkl"))
    joblib.dump(lgb_model, str(PATHS["models_dir"] / "lgb_model.pkl"))
    (PATHS["models_dir"] / "rank_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (PATHS["models_dir"] / "demo_ids.json").write_text(
        json.dumps({"users": demo_users, "note": "演示用户/商品见 Flask / 与报告素材",
                    "products": ["85123A", "22423"]}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n已存 models/lr_model.pkl、models/lgb_model.pkl、rank_meta.json、demo_ids.json")
    print(f"总耗时 {time.time()-t0:.0f}s")


def pick_cases(rank_full, lr_top10, test_map, valid_users):
    """成功案例：LR top1 命中测试篮；失败案例：LR top1 高分未命中但 Top10 有命中。

    只在「训练期订单较多、测试篮适中」的 valid 用户里选，保证例子有说服力。
    """
    cand_s, cand_f = [], []
    for u in sorted(valid_users):
        basket = test_map[u]
        size = len(basket)
        if not (2 <= size <= 12):
            continue
        top = lr_top10[u]
        if not top:
            continue
        hit_ranks = [i + 1 for i, c in enumerate(top) if c in basket]
        if top[0] in basket:
            cand_s.append((u, size, hit_ranks))
        elif hit_ranks and top[0] not in basket:
            cand_f.append((u, size, hit_ranks, top[0]))
    cand_s.sort(key=lambda t: (-len(t[2]), -t[1], t[0]))
    cand_f.sort(key=lambda t: (-t[1], t[0]))
    users_picked = []
    cases = []
    for u, size, hit_ranks in cand_s[:1]:
        full = rank_full(u)
        cases.append(case_json(u, full, test_map[u]))
        users_picked.append(int(u))
    for u, size, hit_ranks, top1 in cand_f[:1]:
        full = rank_full(u)
        cases.append(case_json(u, full, test_map[u], kind="failure"))
        users_picked.append(int(u))
    # 演示用户：优先选两个 LR 有命中的 valid 用户（方便 Flask 演示页展示效果）
    demo = []
    for u, size, hit_ranks in sorted(cand_s, key=lambda t: t[0]):
        if u not in users_picked:
            demo.append(int(u))
        if len(demo) >= DEMO_USERS_N:
            break
    while len(demo) < DEMO_USERS_N:
        for u in sorted(valid_users):
            if u not in demo and u not in users_picked and 2 <= len(test_map[u]) <= 12:
                demo.append(int(u))
            if len(demo) >= DEMO_USERS_N:
                break
        else:
            break
    return cases, demo


def case_json(u, ranked, basket, kind="success"):
    """ranked: [(code, proba)]；拼出可读的 Top10 与测试篮描述。"""
    from src.recommender import load_products_zh
    zh = load_products_zh()
    cat = PROJECT_ROOT / "outputs" / "chain"
    catalog_path = None
    # 复用 script11 建的英文描述 → 这里直接查中文表即可，英文从 zh 表没有则用 code
    top = [
        {"code": c, "proba": round(float(p), 4),
         "desc_zh": zh.get(c, {}).get("desc_zh", "")}
        for c, p in ranked[:10]
    ]
    hits = [i + 1 for i, c in enumerate([t["code"] for t in top]) if c in basket]
    return {
        "user": int(u), "kind": kind, "test_basket_size": int(len(basket)),
        "test_basket": sorted(basket),
        "top10": top, "hit_ranks_top10": hits, "top1_hit": 1 in hits,
        "note": f"该用户 Top10 命中位次 {hits}（LR 打分）",
    }


if __name__ == "__main__":
    main()
