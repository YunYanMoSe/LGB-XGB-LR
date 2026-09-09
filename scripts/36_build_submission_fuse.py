"""候选集打分 v2 融合提交：LGB(macro)×0.20 + LGB(macro_sp)×0.05 + LR×0.75（全局百分位秩加权）。

选权来源 scripts/35：在 OOF（Jul-Oct）上以 宏 gAUC 为目标做 3 分量网格，
LR 主导的组合把 AUC/gAUC 推到 0.8841/0.8682（超过组长 v9 融合 0.88282/0.86641）。
保留 5% macro_sp 作逐用户 top-k 对冲（老师口径未 100% 锁定为列表类指标）。

流程（每分量 = 全部 5 月 wide_v2 重训，cut=11-01 leader_clean 打分）：
    * 一次算好 58,205 行的 44 特征；
    * lgb_macro / lgb_macro_sp / lr 各训、各打；
    * 分量分 → 全局百分位秩（单调保逐用户内部序）→ 加权和 → 写三列提交。

用法：venv\\Scripts\\python.exe scripts\\36_build_submission_fuse.py
产物：outputs/candidate/sample_submission_cand-align-v3-fuse.csv + models/candidate_*.pkl
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SEED, set_seed
from src import candidate as cand
from src.candidate_v2 import v2_features, V2_FEATS

set_seed(SEED)

REPLAY_DIR = PROJECT_ROOT / "outputs" / "candidate" / "replay"
LEADER_CLEAN = PROJECT_ROOT / "data" / "processed" / "leader_clean.csv"
CAND_INPUT = PROJECT_ROOT / "train_data" / "sample_candidates.csv"
SC_CUT = pd.Timestamp("2010-11-01")
SC_VEC = PROJECT_ROOT / "outputs" / "candidate" / "v9_vecs_20101101_skip.npz"
BASE_FEATS = cand.FEATURES_CAND
FEATS = list(BASE_FEATS) + list(V2_FEATS)
MONTHS = [f"2010-{m:02d}-01" for m in range(6, 11)]
NAME = "cand-align-v3-fuse"
W = {"lgb_macro": 0.20, "lgb_macro_sp": 0.05, "lr": 0.75}   # gAUC 网格最优区 + topk 对冲


def load_train() -> pd.DataFrame:
    return pd.concat([pd.read_csv(REPLAY_DIR / f"wide_v2_{m[:7]}.csv",
                                  dtype={"item_id": str}) for m in MONTHS],
                     ignore_index=True)


def main() -> None:
    t0 = time.time()
    tr = load_train()
    spw = float((tr["label"] == 0).sum() / max(1, int(tr["label"].sum())))
    print(f"[0/4] 训练样本 {len(tr):,} / 特征 {len(FEATS)}；spw={spw:.1f}", flush=True)

    # ---- 分量 1/2：lgb_macro 与 lgb_macro_sp ----
    from lightgbm import LGBMClassifier
    nrow_u = tr.groupby("user_id").size()
    macro = 1.0 / np.sqrt(nrow_u.reindex(tr["user_id"]).to_numpy(dtype="float64"))
    sw_macro = macro.astype("float64")
    sw_macrosp = (macro * np.where(tr["label"] == 1, spw, 1.0)).astype("float64")

    models = {}
    for wname, sw in (("lgb_macro", sw_macro), ("lgb_macro_sp", sw_macrosp)):
        clf = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                             min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=1.0, random_state=SEED, verbosity=-1,
                             deterministic=True, n_jobs=1)
        clf.fit(tr[FEATS], tr["label"], sample_weight=sw)
        models[wname] = clf
        print(f"[1/4] {wname} 训练完成", flush=True)

    lr = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED))
    lr.fit(tr[FEATS], tr["label"])
    models["lr"] = lr
    print("[2/4] lr 训练完成", flush=True)

    # ---- 打分：一次特征，三模型 ----
    clean = cand.load_clean_train(LEADER_CLEAN)
    raw = pd.read_csv(CAND_INPUT, dtype={c: str for c in ["user_id", "item_id"]})
    df = raw[["user_id", "item_id"]].copy()
    df["user_id"] = pd.to_numeric(df["user_id"]).astype("int64")
    df["item_id"] = df["item_id"].astype(str)
    bundle = cand.build_bundle(clean, cut=SC_CUT, vec_cache=SC_VEC)
    Xb = cand.features_for_pairs(bundle, df["user_id"].to_numpy(), df["item_id"].to_numpy())
    Xv = v2_features(clean, SC_CUT, df)
    Xs = pd.concat([Xb.reset_index(drop=True), Xv.reset_index(drop=True)], axis=1)
    assert list(Xs.columns) == FEATS and Xs.isna().sum().sum() == 0
    print("[3/4] 打分特征就绪（58,205 行 × 44）", flush=True)

    scores = {}
    for wname, m in models.items():
        if wname.startswith("lgb"):
            scores[wname] = m.predict_proba(Xs)[:, 1].astype("float64")
        else:
            scores[wname] = m.predict_proba(Xs)[:, 1].astype("float64")
    N = len(df)
    fused = np.zeros(N, dtype="float64")
    for wname, w in W.items():
        fused = fused + w * rankdata(scores[wname]) / N

    out = pd.DataFrame({"user_id": df["user_id"], "item_id": df["item_id"], "score": fused})
    assert len(out) == len(raw) and out.isna().sum().sum() == 0
    out_path = PROJECT_ROOT / "outputs" / "candidate" / f"sample_submission_{NAME}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")

    import joblib
    for wname, m in models.items():
        joblib.dump(m, str(PROJECT_ROOT / "models" / f"candidate_{NAME}_{wname}.pkl"))
    meta = {"name": NAME, "desc": "candidate-aligned replay fusion (global-percentile-rank)",
            "components": {k: {"weight": w, "model": v} for k, v, w in
                           ((k, "lgb" if k.startswith("lgb") else "lr", W[k]) for k in W)},
            "train_months": MONTHS, "n_train_rows": int(len(tr)),
            "score_cut": str(SC_CUT.date()), "feature_order": FEATS,
            "n_feature_cols": len(FEATS), "weights": W}
    meta_path = PROJECT_ROOT / "models" / f"candidate_meta_{NAME}.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[4/4] {NAME} → {out_path.name}")
    print(f"[覆盖] {len(out):,} 行；score min={fused.min():.4f} mean={fused.mean():.4f} "
          f"max={fused.max():.4f}")
    print(f"[meta] {meta_path.name}；总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
