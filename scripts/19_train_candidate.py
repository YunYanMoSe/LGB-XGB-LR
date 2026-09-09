"""候选集打分任务 ②：LGB / LR 训练 + 本地评测（用户级 70/30）。

读 data/processed/clean_train.csv（scripts/18 产物），以切窗 cut=CUT_DEFAULT=2010-10-01
把数据切成「特征期(< cut) + 尾窗标签(>= cut)」：
    标签语义 = 用户在标签窗内是否购买过该商品（**含复购**，贴近候选集含历史已购对）；
    负样本   = 1:NEG_RATIO 配对，从「特征期已购未复购」与「从未购买」两个池掺入（OWNED_NEG_FRAC 贴候选形态）；
    特征     = fe.FEATURES(31) + ui_owned/ui_owned_recency_log = 33 列，训练与打分共用 features_for_pairs。

本地评测：pair-AUC + 每用户再补 EVAL_NEG 个全新负样本构成候选池，算 Hit/Recall/Precision@5,10。

产物（全部新文件）：
    models/candidate_lgb.pkl / candidate_lr.pkl / candidate_meta.json（含 feature_order/版本号/指标/重要性）
    outputs/candidate/candidate_importance.csv

运行：venv\\Scripts\\python.exe scripts\\19_train_candidate.py（缺失 clean_train 会自动先跑 18）
"""
from __future__ import annotations

# 锁单线程数值库：numpy/OpenBLAS 多线程求和顺序不确定会造成跨进程结果漂移，
# 这里必须在任何 numpy 导入前生效（连同 LGB 的 deterministic=True）保证逐位可复现。
import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import subprocess
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

from config.settings import SEED, set_seed
from src import candidate as cand

set_seed(SEED)

VALID_RATIO = 0.3
LGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.1, num_leaves=31, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbosity=-1,
    deterministic=True, n_jobs=1,  # 单线程 + deterministic：跨运行逐位可复现
)
IMP_CSV = PROJECT_ROOT / "outputs" / "candidate" / "candidate_importance.csv"
MODEL_FILES = {"lgb": cand.LGB_PKL, "lr": cand.LR_PKL}


def _eval_pool(
    clf, bundle: dict, clean: pd.DataFrame, valid_users: np.ndarray,
    *, cut: pd.Timestamp, eval_neg: int = cand.EVAL_NEG,
) -> pd.DataFrame:
    """对每个 valid 用户构造候选池：正 = 标签窗实购；负 = 池内均匀抽 EVAL_NEG 个"标签窗未购"。

    与真实候选集形态一致：池里会自然包含该用户特征期已购（未复购）的商品。
    """
    lp = clean[clean[cand._TIME] >= cut]
    lp_items: dict[int, set[str]] = {}
    for u, gg in lp.groupby(cand._USER):
        lp_items[int(u)] = set(gg[cand._ITEM].astype(str))
    all_items = np.asarray(sorted(set(clean[cand._ITEM].astype(str).unique())), dtype=object)
    rng = np.random.default_rng(SEED)

    users, codes, labels = [], [], []
    for u in valid_users:
        u = int(u)
        pos = lp_items.get(u, set())
        if not pos:
            continue
        pool = np.asarray(sorted(set(all_items) - pos), dtype=object)
        n_neg = min(eval_neg, len(pool))
        negs = pool[rng.choice(len(pool), size=n_neg, replace=False)] if n_neg > 0 else np.asarray([], dtype=object)
        us = np.full(len(pos) + len(negs), u, dtype="int64")
        cs = np.concatenate([np.asarray(sorted(pos), dtype=object), negs])
        ls = np.concatenate([np.ones(len(pos), dtype="int64"), np.zeros(len(negs), dtype="int64")])
        users.append(us); codes.append(cs); labels.append(ls)
    u_all = np.concatenate(users); c_all = np.concatenate(codes); l_all = np.concatenate(labels)
    X = cand.features_for_pairs(bundle, u_all, c_all)
    p = clf.predict_proba(X.to_numpy(dtype=np.float64))[:, 1]
    ev = pd.DataFrame({"user_id": u_all, "stock_code": c_all, "score": p, "label": l_all})
    return ev


def main() -> None:
    t0 = time.time()
    if not cand.CLEAN_TRAIN.exists():
        print("clean_train.csv 缺失，自动先跑 scripts/18…")
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "18_clean_train.py")], check=True)

    clean = cand.load_clean_train()
    cut = cand.CUT_DEFAULT
    print(f"[数据] clean_train {len(clean):,} 行；切窗 cut={cut.date()}；"
          f"特征期 {int((clean[cand._TIME] < cut).sum()):,} / "
          f"标签窗 {int((clean[cand._TIME] >= cut).sum()):,} 行")

    # ---- 画像（含特征期重训向量缓存）----
    bundle = cand.build_bundle(clean, cut=cut, vec_cache=cand.VEC_CACHE)
    print(f"[画像] fp 用户 {len(bundle['users_all']):,}；目录商品 {len(bundle['catalog']):,}；"
          f"向量词表 {bundle['V']:,}")

    # ---- 数据集 ----
    ds = cand.build_pair_dataset(clean, bundle, cut=cut)
    stats = ds.attrs["stats"]
    print(f"[样本] 候选用户 {stats['n_users_total']:,}（fp 订单≥2），有正样本 "
          f"{stats['n_users_with_pos']:,}；正 {stats['pos_n']:,} / 负 {stats['neg_n']:,}（1:"
          f"{stats['neg_n'] / max(1, stats['pos_n']):.1f}）")

    feats = list(cand.FEATURES_CAND)
    assert list(ds.columns[2:-1]) == feats, "数据集列序 ≠ FEATURES_CAND"
    assert ds[feats].isna().sum().sum() == 0, "特征含 NaN"

    # 确定性自检：任意行重算 features_for_pairs 应与表中行一致（训练特征≡打分特征）
    head = ds.head(5)
    X_again = cand.features_for_pairs(bundle, head["user_id"].to_numpy(), head["stock_code"].to_numpy())
    assert np.allclose(X_again.to_numpy(dtype=np.float64),
                       head[feats].to_numpy(dtype=np.float64)), "features_for_pairs 不一致！"
    print("[自检] features_for_pairs 重算与宽表一致（训练特征 ≡ 打分特征），无 NaN，列序正确")

    X = ds[feats].to_numpy(dtype=np.float64)
    y = ds["label"].to_numpy()
    fit_mask, valid_mask = cand.user_split(ds["user_id"].to_numpy(), ratio=VALID_RATIO, seed=SEED)
    n_fit_u = int(np.unique(ds["user_id"].to_numpy()[fit_mask]).size)
    n_val_u = int(np.unique(ds["user_id"].to_numpy()[valid_mask]).size)
    n_pos_fit = int(y[fit_mask].sum()); n_neg_fit = int((y[fit_mask] == 0).sum())
    print(f"[切分] 用户级 {1 - VALID_RATIO:.0%}/{VALID_RATIO:.0%}（seed=42）：fit {n_fit_u:,} 用户 / "
          f"valid {n_val_u:,} 用户；fit 样本 {int(fit_mask.sum()):,}（正 {n_pos_fit:,}/负 {n_neg_fit:,}）")

    # ---- LR 基线 ----
    lr = Pipeline([("sc", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=SEED))])
    lr.fit(X[fit_mask], y[fit_mask])
    auc_lr = float(roc_auc_score(y[valid_mask], lr.predict_proba(X[valid_mask])[:, 1]))

    # ---- LightGBM ----
    from lightgbm import LGBMClassifier
    params = dict(LGB_PARAMS)
    params["scale_pos_weight"] = n_neg_fit / max(1, n_pos_fit)
    lgb = LGBMClassifier(**params)
    lgb.fit(X[fit_mask], y[fit_mask])
    p_val = lgb.predict_proba(X[valid_mask])[:, 1]
    auc_lgb = float(roc_auc_score(y[valid_mask], p_val))
    print(f"[训练] valid pair-AUC：LR={auc_lr:.4f}  LGB={auc_lgb:.4f}")

    # ---- 候选池式评测（每 valid 用户补 EVAL_NEG 个全新负样本）----
    valid_users = np.sort(np.unique(ds["user_id"].to_numpy()[valid_mask]))
    ev_lgb = _eval_pool(lgb, bundle, clean, valid_users, cut=cut)
    ev_lr = _eval_pool(lr, bundle, clean, valid_users, cut=cut)
    m_lgb = cand.rank_metrics(ev_lgb)
    m_lr = cand.rank_metrics(ev_lr)
    auc_pool_lgb = float(roc_auc_score(ev_lgb["label"], ev_lgb["score"]))
    auc_pool_lr = float(roc_auc_score(ev_lr["label"], ev_lr["score"]))
    print(f"[评测] 候选池 {len(ev_lgb):,} 行（valid 用户 {ev_lgb['user_id'].nunique()}，"
          f"正 {int(ev_lgb['label'].sum()):,}/负 {int((ev_lgb['label']==0).sum()):,}）")
    print(f"{'模型':<6}{'pool AUC':>10}{'Hit@5':>8}{'Rec@5':>8}{'P@5':>8}{'Hit@10':>9}{'Rec@10':>9}{'P@10':>9}")
    for name, ev, mm, a in (("LR", ev_lr, m_lr, auc_pool_lr), ("LGB", ev_lgb, m_lgb, auc_pool_lgb)):
        print(f"{name:<6}{a:>10.4f}"
              f"{mm['hit@5']:>8.3f}{mm['rec@5']:>8.3f}{mm['prec@5']:>8.3f}"
              f"{mm['hit@10']:>9.3f}{mm['rec@10']:>9.3f}{mm['prec@10']:>9.3f}")

    # ---- 重要性 ----
    booster = lgb.booster_
    table = pd.DataFrame({
        "feature": feats,
        "gain": booster.feature_importance(importance_type="gain").astype(float),
        "weight": booster.feature_importance(importance_type="split").astype(float),
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    IMP_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(IMP_CSV, index=False, encoding="utf-8")

    # ---- meta ----
    meta = {
        "task": "train.csv candidate scoring（候选集 user_id,item_id → score）",
        "model_version": cand.MODEL_VERSION,
        "cut": str(cut.date()),
        "label_semantics": "标签窗(>=cut)是否购买该商品，含复购（正）；掺 fp 已购未复购 + 从未购买（负）",
        "n_clean_rows": int(len(clean)),
        "bundle": {"vec_cache": str(cand.VEC_CACHE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                   "n_fp_users": int(len(bundle["users_all"])),
                   "catalog_n": int(len(bundle["catalog"]))},
        "negative_sampling": {"neg_ratio": cand.NEG_RATIO,
                              "owned_neg_frac": cand.OWNED_NEG_FRAC,
                              "seed": SEED},
        "dataset": {**stats, "actual_neg_per_pos": round(stats["neg_n"] / max(1, stats["pos_n"]), 2)},
        "feature_order": feats,
        "n_feature_cols": len(feats),
        "extra_features": cand.EXTRA_FEATS,
        "split": {"ratio": VALID_RATIO, "seed": SEED,
                  "n_fit_users": n_fit_u, "n_valid_users": n_val_u},
        "eval_pool": {"per_user_neg": cand.EVAL_NEG,
                      "rows": int(len(ev_lgb)), "pool_users": int(ev_lgb["user_id"].nunique()),
                      "n_pos": int(ev_lgb["label"].sum())},
        "metrics": {
            "lr": {"valid_pair_auc": round(auc_lr, 4), "pool_auc": round(auc_pool_lr, 4), **m_lr},
            "lgb": {"valid_pair_auc": round(auc_lgb, 4), "pool_auc": round(auc_pool_lgb, 4), **m_lgb},
        },
        "lgb_params": {k: (float(v) if isinstance(v, (int, float)) else v)
                       for k, v in params.items()},
        "feature_importance": table.to_dict("records"),
    }
    cand.CAND_META.parent.mkdir(parents=True, exist_ok=True)
    cand.CAND_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump(lgb, str(MODEL_FILES["lgb"]))
    joblib.dump(lr, str(MODEL_FILES["lr"]))
    print(f"[输出] LGB -> {MODEL_FILES['lgb'].name}；LR -> {MODEL_FILES['lr'].name}；"
          f"meta -> {cand.CAND_META.name}；重要性 -> {IMP_CSV.name}")

    print("\n特征重要性 Top12（按 gain 降序）")
    print(f"{'rank':<5}{'特征':<24}{'gain':>12}{'weight':>9}")
    for i, r in table.head(12).iterrows():
        print(f"{i + 1:<5}{r['feature']:<24}{r['gain']:>12.0f}{r['weight']:>9.0f}")
    print(f"\n总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
