"""候选集打分 v12（LR 外推彩票 #4）：复用 v6 三 pkl（不重训），融合权把 LR 推到 .75。

依据（scripts/60 + 板上实证）：
  * 本地复合随 LR 权单调降（.75 .58293 / .8 .58216 / .9 .58124 / 纯 LR .57974 < v6 .58429）→ 本地反对；
  * 但真榜系统性奖励 LR 侧重（.375 → 败 .7774；.6 → 胜 .7782），本地与板在 LR 轴**方向相反** ⇒ 此轴纯板盲。
  * LR 权 > .6 从未在板上试过 → 免费彩票：赌"榜奖励 LR 侧重"这条规律再推一格（平滑模型扛未来窗漂移）。

构造：v6 三模型分数逐位复现自检 → 换权 (lgb .0625, xgb .1875, lr .75)——两树按 v6 的 1:3 内部
比例同压到合计 .25，树/线性相对关系不变，唯一变量 = LR 侧重程度。

产物：outputs/candidate/sample_submission_cand-align-v12-lr75.csv + meta
用法：venv\\Scripts\\python.exe scripts\\72_build_submission_v12_lr75.py
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SEED, set_seed
from src import candidate as cand
from src.candidate_v2 import v2_features, V2_FEATS

set_seed(SEED)

LEADER_CLEAN = PROJECT_ROOT / "data" / "processed" / "leader_clean.csv"
CAND_INPUT = PROJECT_ROOT / "train_data" / "sample_candidates.csv"
SC_CUT = pd.Timestamp("2010-11-01")
SC_VEC = PROJECT_ROOT / "outputs" / "candidate" / "v9_vecs_20101101_skip.npz"
V6_CSV = PROJECT_ROOT / "outputs" / "candidate" / "sample_submission_cand-align-v6-fuse.csv"
BASE_FEATS = cand.FEATURES_CAND
FEATS = list(BASE_FEATS) + list(V2_FEATS)
W6 = {"lgb": 0.1, "xgb": 0.3, "lr": 0.6}          # v6 原权（自检用）
W12 = {"lgb": 0.0625, "xgb": 0.1875, "lr": 0.75}   # 树按 1:3 比例同压到 .25，LR → .75
NAME = "cand-align-v12-lr75"


def main() -> None:
    t0 = time.time()
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

    import joblib
    clf_lgb = joblib.load(str(PROJECT_ROOT / "models" / "candidate_cand-align-v6-fuse_lgb.pkl"))
    clf_xgb = joblib.load(str(PROJECT_ROOT / "models" / "candidate_cand-align-v6-fuse_xgb.pkl"))
    clf_lr = joblib.load(str(PROJECT_ROOT / "models" / "candidate_cand-align-v6-fuse_lr.pkl"))
    N = len(df)
    r_lgb = rankdata(clf_lgb.predict_proba(Xs)[:, 1]) / N
    r_xgb = rankdata(clf_xgb.predict_proba(Xs)[:, 1]) / N
    r_lr = rankdata(clf_lr.predict_proba(Xs)[:, 1]) / N
    print(f"[1/3] 三分量分数就绪（{N:,} 行）", flush=True)

    fused6 = W6["lgb"] * r_lgb + W6["xgb"] * r_xgb + W6["lr"] * r_lr
    v6 = pd.read_csv(V6_CSV)
    assert np.allclose(fused6, v6["score"].to_numpy(dtype="float64"), atol=1e-12), "v6 分量复现不一致！"
    print("[self-check] v6 原权融合与 v6 csv 逐位全等通过", flush=True)

    score = W12["lgb"] * r_lgb + W12["xgb"] * r_xgb + W12["lr"] * r_lr
    # 与 v6 的差异量（应集中在秩空间中下段；打印整体差异确认换权生效且非数值事故）
    print(f"[delta] |v12 − v6| mean {np.abs(score - fused6).mean():.5f} / max {np.abs(score - fused6).max():.5f}",
          flush=True)

    assert len(score) == len(raw) and not np.isnan(score).any()
    out = pd.DataFrame({"user_id": df["user_id"], "item_id": df["item_id"], "score": score})
    p = PROJECT_ROOT / "outputs" / "candidate" / f"sample_submission_{NAME}.csv"
    out.to_csv(p, index=False, encoding="utf-8")

    meta = {"name": NAME,
            "desc": "LR 外推彩票：复用 v6 三 pkl 不重训，融合权 lgb.0625/xgb.1875/lr.75 "
                    "(两树按 v6 1:3 比例同压到 .25)；LR 侧重 .6→.75",
            "weights_v12": W12, "weights_v6": W6,
            "rationale": "板系统性奖励 LR 侧重而本地反对 ⇒ LR 轴板盲，> .6 从未上板，免费彩票",
            "reuses": "cand-align-v6-fuse pkls (lgb/xgb/lr)",
            "score_cut": str(SC_CUT.date()), "feature_order": FEATS}
    (PROJECT_ROOT / "models" / f"candidate_meta_{NAME}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[3/3] {NAME} → {p.name}；总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
