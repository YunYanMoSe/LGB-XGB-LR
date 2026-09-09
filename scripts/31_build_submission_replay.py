"""候选集打分任务 ⑩（v9 对标线）：正式提交 = 候选对齐回放 LGB（全回放重训 → cut 11-01 打分）。

与 v1-v4 的本质区别：训练样本不再是从目录随机负采样（25% 基率、丢无正样本用户），
而是把固定候选 58,205 行在 2010-06..10 五个截点逐月回放、标签=当月真实购买
（4.37% 真实基率、覆盖当月全部可计算候选行）—— 训练分布与老师打分的候选分布对齐。

流程（复刻组长 v9，可复现、不覆盖任何旧产物）：
    * 数据 = leader_clean（组长口径，327,827 行，保留服务码；scripts/29 已建）；
    * 回放宽表 = outputs/candidate/replay/wide_*.csv（scripts/29 已建）；
    * 训练 = 全部 5 个月回放样本，LGB；权重 --weight 可选 scale_pos/macro/macro_sp
      （默认 macro_sp = v3 配方 1/sqrt(用户行数) × scale_pos，OOF 上逐用户 macro 最优）；
    * 打分 = cut=2010-11-01（用尽 leader_clean），对全部 58,205 候选行 features_for_pairs
      （同一套 33 特征装配；冷用户/商品走同 v3 冷路径）→ 全行覆盖、全精度 float 输出。

用法：venv\\Scripts\\python.exe scripts\\31_build_submission_replay.py --weight macro_sp --version cand-align-v1
产物：outputs/candidate/sample_submission_<version>.csv + models/candidate_<version>.pkl + meta json
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SEED, set_seed
from src import candidate as cand

set_seed(SEED)

REPLAY_DIR = PROJECT_ROOT / "outputs" / "candidate" / "replay"
LEADER_CLEAN = PROJECT_ROOT / "data" / "processed" / "leader_clean.csv"
CAND_INPUT = PROJECT_ROOT / "train_data" / "sample_candidates.csv"
SC_CUT = pd.Timestamp("2010-11-01")
SC_VEC = PROJECT_ROOT / "outputs" / "candidate" / "v9_vecs_20101101_skip.npz"
FEATS = cand.FEATURES_CAND
MONTHS = [f"2010-{m:02d}-01" for m in range(6, 11)]


def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="macro_sp", choices=["scale_pos", "macro", "macro_sp"])
    ap.add_argument("--version", default="cand-align-v1")
    ap.add_argument("--n_estimators", type=int, default=1000)
    args = ap.parse_args()
    name = args.version

    # ---- 1. 训练数据：全部回放样本 ----
    tr = pd.concat([pd.read_csv(REPLAY_DIR / f"wide_{m[:7]}.csv",
                                dtype={"item_id": str}) for m in MONTHS],
                   ignore_index=True)
    spw = float((tr["label"] == 0).sum() / max(1, int(tr["label"].sum())))
    sw = None
    use_spw = 1.0
    if args.weight == "scale_pos":
        use_spw = spw
    elif args.weight in ("macro", "macro_sp"):
        nrow_u = tr.groupby("user_id").size()
        macro = (1.0 / np.sqrt(nrow_u.reindex(tr["user_id"]).to_numpy(dtype="float64")))
        sw = macro if args.weight == "macro" else \
            (macro * np.where(tr["label"] == 1, spw, 1.0)).astype("float64")
    print(f"[1/4] 回放训练 {len(tr):,} 行 / 正 {int(tr['label'].sum()):,} "
          f"({tr['label'].mean():.4%})；weight={args.weight}", flush=True)

    from lightgbm import LGBMClassifier
    clf = LGBMClassifier(
        n_estimators=args.n_estimators, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=use_spw, random_state=SEED, verbosity=-1,
        deterministic=True, n_jobs=1)
    clf.fit(tr[FEATS], tr["label"], sample_weight=sw)
    print(f"[2/4] LGB 训练完成（{clf.n_estimators} 树）", flush=True)

    # ---- 3. 打分：cut 11-01（leader_clean 全量历史）----
    print("[3/4] 候选集打分（cut=11-01，leader_clean）…", flush=True)
    clean = cand.load_clean_train(LEADER_CLEAN)
    bundle = cand.build_bundle(clean, cut=SC_CUT, vec_cache=SC_VEC)

    raw = pd.read_csv(CAND_INPUT, dtype={c: str for c in ["user_id", "item_id"]})
    df = raw[["user_id", "item_id"]].copy()
    df["user_id"] = pd.to_numeric(df["user_id"]).astype("int64")
    df["item_id"] = df["item_id"].astype(str)
    Xs = cand.features_for_pairs(bundle, df["user_id"].to_numpy(), df["item_id"].to_numpy())
    assert list(Xs.columns) == FEATS and Xs.isna().sum().sum() == 0
    score = clf.predict_proba(Xs)[:, 1].astype(np.float64)

    out = pd.DataFrame({"user_id": df["user_id"], "item_id": df["item_id"],
                        "score": score})
    assert len(out) == len(df) and out.isna().sum().sum() == 0
    out_path = PROJECT_ROOT / "outputs" / "candidate" / f"sample_submission_{name}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")

    # ---- 4. 存模型 + meta ----
    model_path = PROJECT_ROOT / "models" / f"candidate_{name}.pkl"
    import joblib
    joblib.dump(clf, str(model_path))
    owned = sum(1 for u, c in zip(df["user_id"], df["item_id"])
                if int(u) in bundle["owned_fp"] and str(c) in bundle["owned_fp"][int(u)])
    meta = {
        "name": name, "desc": "candidate-aligned replay LGB (leader v9 method)",
        "train_months": MONTHS, "n_train_rows": int(len(tr)), "pos_rate": float(tr["label"].mean()),
        "weight": args.weight, "score_cut": str(SC_CUT.date()),
        "feature_order": FEATS, "n_feature_cols": len(FEATS),
        "lgb_params": {"n_estimators": int(clf.n_estimators), "learning_rate": 0.05,
                       "num_leaves": 31, "scale_pos_weight": float(use_spw)},
        "train_stats": {"pos": int(tr["label"].sum()), "neg": int((tr["label"] == 0).sum())},
    }
    meta_path = PROJECT_ROOT / "models" / f"candidate_meta_{name}.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[版本] {name} → {out_path}")
    print(f"[覆盖] 输出 {len(out):,} 行；fp 历史已购 {owned:,} (~{owned/len(out):.2%})")
    print(f"[分数] min={score.min():.4f} mean={score.mean():.4f} max={score.max():.4f}")
    print(f"[meta] {meta_path.name}；总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
