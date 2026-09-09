"""候选集打分任务 ④：本地预估分（假定老师按 AUC/覆盖率/重叠率评分）。

老师真实榜单 = 训练数据(train.csv)之后一个不可见的未来窗。本地无法精确复现，只能给
"若老师口径与本脚本一致时会得多少分"的估算，因此分两栏：
    A) 同窗用户留出（转导）：scripts/19 已算、存于 candidate_meta.json（valid 用户看不到
       训练标签，但标签窗与训练相同）——乐观上界；
    B) 滚动时间外推：用「特征<2010-09-01、标签=9 月」训练一个同参 LGB，再在
       「特征<2010-10-01、标签=10 月」——该窗从未作为任何训练标签——上打分。
       这最接近"用已知预测一个没见过的未来窗"，作为主预估。

口径（K=10，均为"按 score 降序每用户取 Top-K 为推荐表"）：
    pair-AUC        全候选对排序质量（正=未来窗实购）；
    Hit@10          至少命中 1 件的用户占比（macro）；
    Recall@10       |TopK∩真实|/|真实| 的用户均值；
    覆盖率@10(item)  distinct(TopK∩真实) / distinct(真实)；
    重叠率@10(Jacc)  |TopK∩真实| / |TopK∪真实| 的用户均值。

用法：venv\\Scripts\\python.exe scripts\\21_estimate_score.py（约 1–3 分钟，含 9 月向量重训）
产物：outputs/candidate/score_estimate.txt
"""
from __future__ import annotations

# 锁单线程数值库（与 19/20 相同），保证跨进程可复现
import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

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
from sklearn.metrics import roc_auc_score

from config.settings import SEED, set_seed
from src import candidate as cand

set_seed(SEED)

K = 10
TR_CUT = pd.Timestamp("2010-09-01")   # 训练窗：特征<9-01、标签=9 月
TE_CUT = pd.Timestamp("2010-10-01")   # 测试窗：特征<10-01、标签=10 月（未参与任何训练标签）
VEC_TR = PROJECT_ROOT / "outputs" / "candidate" / "vectors_20100901_skip.npz"
OUT_TXT = PROJECT_ROOT / "outputs" / "candidate" / "score_estimate.txt"


def build_test_pool(clean: pd.DataFrame, cut: pd.Timestamp,
                    n_neg: int = cand.EVAL_NEG):
    """返回 (users, codes, labels) 元组数组：对 fp(订单≥2)∩未来窗有购买的用户建候选池。

    正=未来窗实购；负=窗内未购（含该用户特征期已购未复购）均匀抽 n_neg 个。
    """
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


def metric_block(ev: pd.DataFrame, k: int = K) -> dict:
    auc = float(roc_auc_score(ev["label"], ev["score"]))
    n_u = hits = prec = rec = jac = 0.0
    cov_num = set(); cov_den = set()
    for _u, g in ev.groupby("user_id", sort=False):
        g = g.sort_values("score", ascending=False)
        top = set(g["stock_code"].head(k))
        gt = set(g.loc[g["label"] == 1, "stock_code"])
        if not gt:
            continue
        n_u += 1
        inter = top & gt
        hits += 1.0 if inter else 0.0
        prec += len(inter) / min(k, len(g))
        rec += len(inter) / len(gt)
        jac += len(inter) / len(top | gt)
        cov_num |= inter
        cov_den |= gt
    m = max(1.0, n_u)
    return {"pair_auc": round(auc, 4),
            f"hit@{k}": round(hits / m, 4),
            f"rec@{k}": round(rec / m, 4),
            f"cov_item@{k}": round(len(cov_num) / max(1, len(cov_den)), 4),
            f"jac@{k}": round(jac / m, 4),
            "n_users_pos": int(n_u)}


def fmt(b: dict) -> str:
    return (f"pair-AUC {b['pair_auc']:.4f}  Hit@{K} {b[f'hit@{K}']:.3f}  "
            f"Rec@{K} {b[f'rec@{K}']:.3f}  cov_item@{K} {b[f'cov_item@{K}']:.3f}  "
            f"Jacc@{K} {b[f'jac@{K}']:.3f}  (正用户 {b['n_users_pos']})")


def main() -> None:
    t0 = time.time()
    meta = json.loads(cand.CAND_META.read_text(encoding="utf-8"))
    feats = meta["feature_order"]
    clean = cand.load_clean_train()
    lm = meta["metrics"]["lgb"]

    lines = ["候选集打分 · 本地预估分（假定老师按 AUC/覆盖率/重叠率，K=10）", "=" * 68,
             "口径：正=候选对在未来窗被实购；Top-K=按 score 每用户取前 K；",
             "  覆盖率@K(item 级)=distinct(TopK∩真实)/distinct(真实)；",
             "  重叠率@K=Jaccard |TopK∩真实|/|TopK∪真实| 的用户均值。",
             "  老师真实榜单窗不可见，下表为本地估算，非最终分。", ""]

    # ---- A 同窗用户留出（meta 已有，乐观上界）----
    lines.append("[A] 同窗用户留出（cand-lgb-v1 / valid 用户留出 / 标签=训练同窗 Oct）")
    lines.append(f"    valid pair-AUC {lm['valid_pair_auc']:.4f}   候选池 AUC {lm['pool_auc']:.4f}   "
                 f"Hit@10 {lm['hit@10']:.3f}   Rec@10 {lm['rec@10']:.3f}   "
                 f"(本行来自 candidate_meta.json，无 cov/Jacc 列)")

    # ---- B 滚动时间外推 ----
    lines.append("")
    lines.append("[B] 滚动时间外推：训练标签=2010-09（特征<09-01）；测试=2010-10"
                 "（特征<10-01，从未当过训练标签）")
    print(f"[B] 构建 9 月训练窗（含向量重训缓存 {VEC_TR.name}）…")
    bundle_tr = cand.build_bundle(clean, cut=TR_CUT, vec_cache=VEC_TR)
    ds = cand.build_pair_dataset(clean, bundle_tr, cut=TR_CUT)
    st = ds.attrs["stats"]
    print(f"[B] 训练样本：正 {st['pos_n']:,} / 负 {st['neg_n']:,}（有正用户 {st['n_users_with_pos']:,}）")
    X = ds[feats].to_numpy(dtype=np.float64)
    y = ds["label"].to_numpy()

    from lightgbm import LGBMClassifier
    params = {k: v for k, v in meta["lgb_params"].items() if k != "scale_pos_weight"}
    # meta 里整型参数按 float 存储（scripts/19 落盘口径），LightGBM 对 verbosity 等严格要 int
    for _k in ("n_estimators", "num_leaves", "min_child_samples", "random_state",
               "verbosity", "n_jobs", "max_depth"):
        if _k in params:
            params[_k] = int(params[_k])
    params["deterministic"] = bool(params.get("deterministic", False))
    params["scale_pos_weight"] = float((y == 0).sum() / max(1, int(y.sum())))
    clf = LGBMClassifier(**params).fit(X, y)
    print("[B] 9 月 LGB 训练完成")

    bundle_te = cand.build_bundle(clean, cut=TE_CUT, vec_cache=cand.VEC_CACHE)
    u, c, l = build_test_pool(clean, TE_CUT)
    print(f"[B] 10 月测试候选池 {len(u):,} 行…")
    X_ev = cand.features_for_pairs(bundle_te, u, c)
    assert X_ev.isna().sum().sum() == 0, "测试特征含 NaN"
    ev = pd.DataFrame({"user_id": u, "stock_code": c, "label": l,
                       "score": clf.predict_proba(X_ev.to_numpy(dtype=np.float64))[:, 1]})
    b = metric_block(ev, K)
    print("[B] " + fmt(b))
    lines.append(f"    测试池 {len(u):,} 行；" + fmt(b))

    lines += ["", "读法：A 是「标签窗与训练相同」的乐观上界；B 用「从未当标签的 10 月」外推，更接近老师",
              "      「预测未来」的评法，建议以 B 为主估。若老师分母含无购买用户/不同候选池，数字会变。"]
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n已写 " + str(OUT_TXT))
    print(f"总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
