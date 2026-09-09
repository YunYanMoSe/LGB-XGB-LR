"""候选集打分任务 ⑥：离线配置比选（对齐探针锁定的"逐用户 macro 排序"口径）。

探针结论：老师综合分 = 纯排序、逐用户内部序的 macro（跨用户无关、绝对值无关）。
因此本地选择目标 = scripts/21 协议 B（9 月训练 → 10 月从未当标签的未来窗打分）上的
**逐用户 top-k macro**（Hit / Rec / Jacc / NDCG @ k），而非全局 AUC。

本脚本在同一份协议 B 数据上跑若干配置，输出对比表，供挑一个出 v3 正式提交：
  数据只建一次（9 月训练集 + 10 月测试池），各配置只换 LGB 超参 / 样本权重 / 集成。

用法：venv\\Scripts\\python.exe scripts\\23_tune_v3.py  （约 5-15 分钟，向量缓存已存在则快）
产物：outputs/candidate/tune_v3_results.txt
"""
from __future__ import annotations

# 锁单线程数值库（跨进程可复现）
import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
import sys
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import SEED, set_seed
from src import candidate as cand

set_seed(SEED)

TR_CUT = pd.Timestamp("2010-09-01")   # 训练：特征<09-01、标签=9 月
TE_CUT = pd.Timestamp("2010-10-01")   # 测试：特征<10-01、标签=10 月（从未当训练标签）
VEC_TR = PROJECT_ROOT / "outputs" / "candidate" / "vectors_20100901_skip.npz"
KS = (5, 10, 20)
OUT_TXT = PROJECT_ROOT / "outputs" / "candidate" / "tune_v3_results.txt"

INT_KEYS = ("n_estimators", "num_leaves", "min_child_samples", "max_depth",
            "random_state", "verbosity", "n_jobs")
FLOAT_KEYS = ("learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda")


def build_test_pool(clean: pd.DataFrame, cut: pd.Timestamp, n_neg: int = cand.EVAL_NEG):
    """正=未来窗实购；负=窗内未购（含特征期已购未复购）均匀抽 n_neg。返回 (u,c,label)。"""
    n_orders = clean[clean[cand._TIME] < cut].groupby(cand._USER)[cand._INVOICE].nunique()
    cand_users = sorted(int(u) for u, n in n_orders.items() if int(n) >= cand.MIN_FP_ORDERS)
    lp = clean[clean[cand._TIME] >= cut]
    lp_items: dict[int, set[str]] = {}
    for u, gg in lp.groupby(cand._USER):
        lp_items[int(u)] = set(gg[cand._ITEM].astype(str))
    all_items = np.asarray(sorted(set(clean[cand._ITEM].astype(str).unique())), dtype=object)
    rng = np.random.default_rng(SEED)
    users, codes, labels = [], [], []
    for u in cand_users:
        pos = lp_items.get(u, set())
        if not pos:
            continue
        pool = np.asarray(sorted(set(all_items) - pos), dtype=object)
        k = min(n_neg, len(pool))
        neg = pool[rng.choice(len(pool), size=k, replace=False)] if k else np.asarray([], dtype=object)
        users.append(np.full(len(pos) + len(neg), u, dtype="int64"))
        codes.append(np.concatenate([np.asarray(sorted(pos), dtype=object), neg]))
        labels.append(np.concatenate([np.ones(len(pos), dtype="int64"),
                                      np.zeros(len(neg), dtype="int64")]))
    return (np.concatenate(users), np.concatenate(codes), np.concatenate(labels))


def _ndcg(ranked_codes, gt: set, k: int) -> float:
    dcg = idcg = 0.0
    for i, code in enumerate(ranked_codes[:k]):
        rel = 1.0 if code in gt else 0.0
        dcg += rel / math.log2(i + 2)
    for i in range(min(k, len(gt))):
        idcg += 1.0 / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


def metric_block(ev: pd.DataFrame, ks=KS) -> dict:
    out: dict = {}
    out["pair_auc"] = round(float(roc_auc_score(ev["label"], ev["score"])), 4)
    n_u = 0
    hits = {k: 0.0 for k in ks}; rec = {k: 0.0 for k in ks}
    jac = {k: 0.0 for k in ks}; ndc = {k: 0.0 for k in ks}
    for _u, g in ev.groupby("user_id", sort=False):
        g = g.sort_values("score", ascending=False)
        gt = set(g.loc[g["label"] == 1, "stock_code"])
        if not gt:
            continue
        n_u += 1
        codes = list(g["stock_code"])
        for k in ks:
            top = codes[:k]
            inter = set(top) & gt
            hits[k] += 1.0 if inter else 0.0
            rec[k] += len(inter) / len(gt)
            jac[k] += len(inter) / len(set(top) | gt)
            ndc[k] += _ndcg(codes, gt, k)
    m = max(1, n_u)
    out["n_users_pos"] = int(n_u)
    for k in ks:
        out[f"hit@{k}"] = round(hits[k] / m, 4)
        out[f"rec@{k}"] = round(rec[k] / m, 4)
        out[f"jac@{k}"] = round(jac[k] / m, 4)
        out[f"ndcg@{k}"] = round(ndc[k] / m, 4)
    return out


def _lgb_params(over: dict) -> dict:
    p = {"n_estimators": 300, "learning_rate": 0.1, "num_leaves": 31,
         "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
         "random_state": SEED, "verbosity": -1, "deterministic": True, "n_jobs": 1}
    p.update(over)
    for k in INT_KEYS:
        if k in p:
            p[k] = int(p[k])
    for k in FLOAT_KEYS:
        if k in p:
            p[k] = float(p[k])
    return p


def main() -> None:
    t0 = time.time()
    clean = cand.load_clean_train()
    feats = cand.FEATURES_CAND

    # ---- 训练窗（9 月标签）一次建好 ----
    print("[1/3] 构建 9 月训练集（向量走缓存）…", flush=True)
    bundle_tr = cand.build_bundle(clean, cut=TR_CUT, vec_cache=VEC_TR)
    ds = cand.build_pair_dataset(clean, bundle_tr, cut=TR_CUT)
    st = ds.attrs["stats"]
    X = ds[feats].to_numpy(dtype=np.float64)
    y = ds["label"].to_numpy()
    print(f"      训练样本：正 {st['pos_n']:,} / 负 {st['neg_n']:,}（有正用户 {st['n_users_with_pos']:,}）",
          flush=True)
    # 每用户行数（macro 加权变体用）
    nrow_u = ds.groupby("user_id").size()
    w_macro = (1.0 / np.sqrt(nrow_u.reindex(ds["user_id"]).to_numpy(dtype="float64"))).astype("float64")

    # ---- 测试池（10 月）特征一次建好 ----
    print("[2/3] 构建 10 月测试池特征…", flush=True)
    bundle_te = cand.build_bundle(clean, cut=TE_CUT, vec_cache=cand.VEC_CACHE)
    u, c, l = build_test_pool(clean, TE_CUT)
    X_ev = cand.features_for_pairs(bundle_te, u, c)
    ev_base = pd.DataFrame({"user_id": u, "stock_code": c, "label": l})
    print(f"      测试池 {len(u):,} 行", flush=True)

    # ---- 配置 ----
    CFGS = [
        ("anchor(v1超参)",    "lgb", {}, None),          # 复现协议 B 基线，锚点
        ("c1_more_tree_lowlr","lgb", {"n_estimators": 1000, "learning_rate": 0.05,
                                      "num_leaves": 63, "min_child_samples": 30}, None),
        ("c2_deep_reg",       "lgb", {"n_estimators": 700, "learning_rate": 0.06,
                                      "num_leaves": 127, "min_child_samples": 40,
                                      "subsample": 0.7, "colsample_bytree": 0.6}, None),
        ("c3_smallclassic",   "lgb", {"n_estimators": 1500, "learning_rate": 0.03,
                                      "num_leaves": 31, "min_child_samples": 10,
                                      "subsample": 0.9, "colsample_bytree": 0.9}, None),
        ("c1+macro_user_wt",  "lgb", {"n_estimators": 1000, "learning_rate": 0.05,
                                      "num_leaves": 63, "min_child_samples": 30}, "macro"),
    ]
    from lightgbm import LGBMClassifier
    spw = float((y == 0).sum() / max(1, int(y.sum())))

    rows: list[tuple] = []
    evals: dict[str, pd.DataFrame] = {}

    def run_lgb(name, over, weight_name) -> pd.DataFrame:
        par = _lgb_params(over)
        par["scale_pos_weight"] = spw
        clf = LGBMClassifier(**par)
        w = ds["weight_macro"] if weight_name == "macro" else None
        clf.fit(X, y, sample_weight=w)
        ev = ev_base.copy()
        ev["score"] = clf.predict_proba(X_ev)[:, 1]
        return ev

    ds = ds.assign(weight_macro=w_macro)  # 列现挂上去，供 macro 变体取
    print("[3/3] 逐配置训练与评测…", flush=True)
    for name, kind, over, wt in CFGS:
        ev = run_lgb(name, over, wt)
        evals[name] = ev
        b = metric_block(ev)
        rows.append((name, kind, b))
        print(f"      {name:22s} " + "  ".join(
            f"rec@{k}={b[f'rec@{k}']:.3f}" for k in KS) + f"   ndcg@10={b['ndcg@10']:.3f}   "
            f"hit@10={b['hit@10']:.3f}   pairAUC={b['pair_auc']:.4f}", flush=True)

    # LR + 两个集成
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0,
                                                            random_state=SEED, n_jobs=1))
    lr.fit(X, y)
    ev_lr = ev_base.copy(); ev_lr["score"] = lr.predict_proba(X_ev)[:, 1]
    evals["lr"] = ev_lr
    b = metric_block(ev_lr)
    rows.append(("lr_baseline", "lr", b))
    print(f"      {'lr_baseline':22s} " + "  ".join(
        f"rec@{k}={b[f'rec@{k}']:.3f}" for k in KS) + f"   ndcg@10={b['ndcg@10']:.3f}   "
        f"hit@10={b['hit@10']:.3f}   pairAUC={b['pair_auc']:.4f}", flush=True)

    for ename in ("c1_more_tree_lowlr", "c2_deep_reg"):
        ev_ens = ev_base.copy()
        ev_ens["score"] = 0.5 * evals[ename]["score"].to_numpy() + 0.5 * ev_lr["score"].to_numpy()
        evals[f"ens({ename}+lr)"] = ev_ens
        b = metric_block(ev_ens)
        rows.append((f"ens({ename}+lr)", "ens", b))
        print(f"      {f'ens({ename}+lr)':22s} " + "  ".join(
            f"rec@{k}={b[f'rec@{k}']:.3f}" for k in KS) + f"   ndcg@10={b['ndcg@10']:.3f}   "
            f"hit@10={b['hit@10']:.3f}   pairAUC={b['pair_auc']:.4f}", flush=True)

    # ---- 汇总表 ----
    df = pd.DataFrame([
        {"config": n, "kind": k,
         **{f"rec@{kk}": b[f"rec@{kk}"] for kk in KS},
         **{f"ndcg@{kk}": b[f"ndcg@{kk}"] for kk in KS},
         **{f"hit@{kk}": b[f"hit@{kk}"] for kk in KS},
         "jac@10": b["jac@10"], "pair_auc": b["pair_auc"], "n_users": b["n_users_pos"]}
        for n, k, b in rows])
    df["avg_rec"] = df[[f"rec@{kk}" for kk in KS]].mean(axis=1)
    df["avg_ndcg"] = df[[f"ndcg@{kk}" for kk in KS]].mean(axis=1)
    df = df.sort_values("avg_rec", ascending=False).reset_index(drop=True)

    cols = ["config", "avg_rec", "avg_ndcg", "rec@5", "rec@10", "rec@20",
            "ndcg@5", "ndcg@10", "ndcg@20", "hit@5", "hit@10", "hit@20",
            "jac@10", "pair_auc", "n_users"]
    df[cols].to_string(OUT_TXT, index=False)
    print("\n结果已写 " + str(OUT_TXT))
    print(df[cols].round(4).to_string(index=False))
    print(f"总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
