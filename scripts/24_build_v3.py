"""候选集打分任务 ⑦：按 scripts/23 选定的配置重训正式 v3 模型并产出提交文件。

与 v1(v2) 训练协议差异（全部更贴老师口径、且不碰旧文件）：
    * 训练切窗不变 2010-10-01（历史<10-01、标签=10 月整月），但**用全部用户**训练
      （v1 为留 valid 只用了 70% 用户；本地代理/B 也全量，故正式模型应全量）；
    * 超参 = scripts/23 在本地代理上按"逐用户 macro 排序"选出的配置（v3_config.json）；
    * 打分特征切窗 = 2010-11-01（用尽整份 train.csv 近况，同 v2）；
    * 产物全新文件 models/candidate_<name>.pkl + sample_submission_<name>.csv，
      cand-lgb-v1 系列文件一律不覆盖。

配置：outputs/candidate/v3_config.json 形如
    {"name": "cand-lgb-v3", "desc": "…",
     "over": {"n_estimators": 1000, "learning_rate": 0.05, ...},
     "ensemble_lr": true|false}
ensemble_lr=true 时分数 = 0.5*LGB + 0.5*LR(同特征同窗标准化)。

用法：venv\\Scripts\\python.exe scripts\\24_build_v3.py
"""
from __future__ import annotations

# 锁单线程数值库（跨进程可复现）
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import SEED, set_seed
from src import candidate as cand

set_seed(SEED)

INT_KEYS = ("n_estimators", "num_leaves", "min_child_samples", "max_depth",
            "random_state", "verbosity", "n_jobs")
FLOAT_KEYS = ("learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda")

TR_CUT = cand.CUT_DEFAULT                      # 2010-10-01：特征<10-01、标签=10 月
SC_CUT = pd.Timestamp("2010-11-01")            # 打分切窗：用尽整份 train.csv
VEC_SC = PROJECT_ROOT / "outputs" / "candidate" / "vectors_20101101_skip.npz"
CAND_INPUT = PROJECT_ROOT / "train_data" / "sample_candidates.csv"
CFG_FILE = PROJECT_ROOT / "outputs" / "candidate" / "v3_config.json"


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
    cfg = json.loads(CFG_FILE.read_text(encoding="utf-8"))
    name = cfg["name"]
    feats = cand.FEATURES_CAND

    clean = cand.load_clean_train()

    # ---- 1. 全量重训（切窗 10-01，全部用户）----
    print("[1/3] 构建训练集（cut=10-01，全部用户）…", flush=True)
    bundle_tr = cand.build_bundle(clean, cut=TR_CUT, vec_cache=cand.VEC_CACHE)
    ds = cand.build_pair_dataset(clean, bundle_tr, cut=TR_CUT)
    st = ds.attrs["stats"]
    X = ds[feats].to_numpy(dtype=np.float64)
    y = ds["label"].to_numpy()
    spw = float((y == 0).sum() / max(1, int(y.sum())))
    print(f"      训练样本：正 {st['pos_n']:,} / 负 {st['neg_n']:,}（用户 {st['n_users_total']:,}，"
          f"含正 {st['n_users_with_pos']:,}）", flush=True)

    # macro 加权：每行权重 ∝ 1/sqrt(该用户总行数)，让每用户在损失中等权（贴"逐用户 macro"口径）
    weight = cfg.get("weight", "none")
    sw = None
    if weight == "macro":
        nrow_u = ds.groupby("user_id").size()
        sw = (1.0 / np.sqrt(nrow_u.reindex(ds["user_id"]).to_numpy(dtype="float64"))).astype("float64")

    from lightgbm import LGBMClassifier
    par = _lgb_params(cfg.get("over", {}))
    par["scale_pos_weight"] = spw
    clf = LGBMClassifier(**par).fit(X, y, sample_weight=sw)
    print(f"[2/3] LGB 训练完成（{par['n_estimators']} 树 / lr {par['learning_rate']} / "
          f"leaves {par['num_leaves']}）", flush=True)

    lr = None
    if cfg.get("ensemble_lr", False):
        lr = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=2000, C=1.0,
                                              random_state=SEED, n_jobs=1)).fit(X, y)
        print("      已加训 LR（同特征，用于集成）", flush=True)

    # ---- 3. 打分：候选集在切窗 11-01 特征 ----
    print("[3/3] 候选集打分（cut=2010-11-01，向量缓存复用）…", flush=True)
    bundle_sc = cand.build_bundle(clean, cut=SC_CUT, vec_cache=VEC_SC)
    raw = pd.read_csv(CAND_INPUT, dtype={c: str for c in pd.read_csv(CAND_INPUT, nrows=0).columns})
    # 兼容列名
    lower = {c: str(c).lower() for c in raw.columns}
    uc = next(c for c in raw.columns if lower[c] in {"user_id", "customerid"})
    ic = next(c for c in raw.columns if lower[c] in {"item_id", "stockcode", "stock_code"})
    df = raw[[uc, ic]].copy()
    df.columns = ["user_id", "item_id"]
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("int64")
    df = df.dropna(subset=["user_id", "item_id"]).reset_index(drop=True)
    users = df["user_id"].to_numpy(dtype="int64")
    codes = df["item_id"].astype(str).to_numpy(dtype=object)

    Xs = cand.features_for_pairs(bundle_sc, users, codes)
    assert list(Xs.columns) == feats and Xs.isna().sum().sum() == 0
    Xa = Xs.to_numpy(dtype=np.float64)
    score = clf.predict_proba(Xa)[:, 1].astype(np.float64)
    if lr is not None:
        score = 0.5 * score + 0.5 * lr.predict_proba(Xa)[:, 1].astype(np.float64)

    out = pd.DataFrame({"user_id": df["user_id"], "item_id": df["item_id"],
                        "score": np.round(score, 6)})
    assert len(out) == len(df) and out.isna().sum().sum() == 0
    out_path = PROJECT_ROOT / "outputs" / "candidate" / f"sample_submission_{name}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")

    # ---- 4. 存模型 + meta（全新文件，不覆盖 v1）----
    model_path = PROJECT_ROOT / "models" / f"candidate_{name}.pkl"
    import joblib
    joblib.dump(clf, str(model_path))
    meta = {
        "name": name, "desc": cfg.get("desc", ""), "train_cut": str(TR_CUT.date()),
        "score_cut": str(SC_CUT.date()), "model_version": name,
        "feature_order": feats, "n_feature_cols": len(feats),
        "lgb_params": {k: float(v) if isinstance(v, (int, float)) and k not in ("n_jobs",)
                       else v for k, v in par.items()},
        "ensemble_lr": bool(cfg.get("ensemble_lr", False)),
        "weight": weight,
        "train_stats": st,
    }
    meta_path = PROJECT_ROOT / "models" / f"candidate_meta_{name}.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    owned = sum(1 for u, c in zip(users, codes)
                if int(u) in bundle_sc["owned_fp"] and str(c) in bundle_sc["owned_fp"][int(u)])
    print(f"[版本] {name}（模型 {model_path.name}）→ {out_path}")
    print(f"[覆盖] 输出 {len(out):,} 行；fp 历史已购对 {owned:,}（~{owned / len(out):.1%}）")
    print(f"[分数] min={score.min():.4f} mean={score.mean():.4f} max={score.max():.4f}")
    print(f"[meta] {meta_path.name} 已写；总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
