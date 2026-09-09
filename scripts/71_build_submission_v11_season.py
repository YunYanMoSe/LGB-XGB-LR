"""候选集打分 v11（季节性 12 月彩票）：复用 v6 三模型（不重训），serve 侧圣诞内容 boost。

前提（scripts/70 已证伪「12 月独立需求结构」——批发商忙季在 9-10 月且在 v6 训练池内）。用户仍按
「自由提交+榜保留最佳 ⇒ 非负 EV」裁决烧一发彩票：赌老师测试窗 Nov(+Dec) 的下游需求中，圣诞内容
商品（描述含 CHRISTMAS/XMAS/SANTA...）被 v6 系统性低估（训练月全是 1-10 月，模型从未见过「假期
目标月」）。

构造：
  * fused = v6 三 pkl 的全局百分位秩融合（与 v6 serve 逐位一致，先自检 == v6 csv）；
  * seasonal 集合 = 候选商品的描述命中圣诞关键词 且 该商品 9/10 月在售（≥1 units，防 boost 停售/
    首月伪影旧货——诊断里 lift≥2 的主要是这类）；
  * score = fused + DELTA×1_{seasonal}（DELTA=0.10；秩空间内对在售圣诞品做 ~10-percentile 抬升）。

产物：outputs/candidate/sample_submission_cand-align-v11-season.csv + meta
用法：venv\\Scripts\\python.exe scripts\\71_build_submission_v11_season.py
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import re
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
W = {"lgb": 0.1, "xgb": 0.3, "lr": 0.6}
NAME = "cand-align-v11-season"
DELTA = 0.10                                   # 圣诞在售商品的秩抬升幅度（百分位）
KW = re.compile(
    r"CHRISTMAS|XMAS|SANTA|MERRY|SNOWMAN|NATIVITY|CRACKER|STOCKING|FESTIVE|DECORATION|TREE|WRAP|ANGEL")
ALIVE_MONTHS = ("2010-09", "2010-10")          # 商品须在这两月仍在售（≥1 units）
U, I, T, Q, D = "CustomerID", "StockCode", "InvoiceDate", "Quantity", "Description"


def main() -> None:
    t0 = time.time()
    clean = cand.load_clean_train(LEADER_CLEAN)

    # ---- 圣诞关键词商品集（内容 ∩ 在售）----
    desc_all = clean.dropna(subset=[D]).groupby(I, sort=True)[D].first().str.upper()
    is_kw = desc_all.str.contains(KW, regex=True)
    alive = clean[clean[T].dt.strftime("%Y-%m").isin(ALIVE_MONTHS)][I].unique()
    content_only = set(desc_all[is_kw].index)
    seasonal_set = {c for c in content_only if c in alive}
    print(f"[seasonal] 关键词商品 {len(content_only)}；∩在售(9/10月) = {len(seasonal_set)}", flush=True)

    # ---- 打分特征（与 v6 serve 全同）----
    raw = pd.read_csv(CAND_INPUT, dtype={c: str for c in ["user_id", "item_id"]})
    df = raw[["user_id", "item_id"]].copy()
    df["user_id"] = pd.to_numeric(df["user_id"]).astype("int64")
    df["item_id"] = df["item_id"].astype(str)
    bundle = cand.build_bundle(clean, cut=SC_CUT, vec_cache=SC_VEC)
    Xb = cand.features_for_pairs(bundle, df["user_id"].to_numpy(), df["item_id"].to_numpy())
    Xv = v2_features(clean, SC_CUT, df)
    Xs = pd.concat([Xb.reset_index(drop=True), Xv.reset_index(drop=True)], axis=1)
    assert list(Xs.columns) == FEATS and Xs.isna().sum().sum() == 0

    # ---- v6 三模型 pkl → fused（先与 v6 csv 全等自检）----
    import joblib
    clf_lgb = joblib.load(str(PROJECT_ROOT / "models" / "candidate_cand-align-v6-fuse_lgb.pkl"))
    clf_xgb = joblib.load(str(PROJECT_ROOT / "models" / "candidate_cand-align-v6-fuse_xgb.pkl"))
    clf_lr = joblib.load(str(PROJECT_ROOT / "models" / "candidate_cand-align-v6-fuse_lr.pkl"))
    N = len(df)
    fused = (W["lgb"] * rankdata(clf_lgb.predict_proba(Xs)[:, 1]) / N
             + W["xgb"] * rankdata(clf_xgb.predict_proba(Xs)[:, 1]) / N
             + W["lr"] * rankdata(clf_lr.predict_proba(Xs)[:, 1]) / N)
    v6 = pd.read_csv(V6_CSV)
    assert np.allclose(fused, v6["score"].to_numpy(dtype="float64"), atol=1e-12), "v6 复现不一致！"
    print(f"[self-check] fused 与 v6 csv 全等通过（{N:,} 行）", flush=True)

    # ---- seasonal 提升 ----
    in_set = df["item_id"].isin(seasonal_set).to_numpy(dtype=bool)
    print(f"[coverage] 候选行 in-seasonal: {int(in_set.sum())} ({in_set.mean():.2%})，"
          f"用户 {int(df.loc[in_set, 'user_id'].nunique())} 人", flush=True)

    # 提升导致的 top-10 变动诊断（per-user list；δ=0 基线即 v6）
    score = fused + DELTA * in_set.astype("float64")
    tmp = pd.DataFrame({"u": df["user_id"].to_numpy(), "in": in_set,
                        "base": fused, "d10": fused + DELTA * in_set.astype("float64")})
    order = np.argsort(tmp["u"].to_numpy(), kind="stable")
    u_arr = tmp["u"].to_numpy(); b_arr = tmp["base"].to_numpy(); d_arr = tmp["d10"].to_numpy()
    i_arr = tmp["in"].to_numpy()
    n_swap = n_enter_in = n_exit_in = 0
    for u_val, sl in _user_slices(u_arr, order):
        idx = order[sl]
        base_top = set(idx[np.argpartition(-b_arr[idx], 10)[:10]].tolist())
        new_top = set(idx[np.argpartition(-d_arr[idx], 10)[:10]].tolist())
        enter = new_top - base_top
        exit_ = base_top - new_top
        n_swap += len(enter)
        n_enter_in += int(sum(i_arr[e] for e in enter))
        n_exit_in += int(sum(i_arr[e] for e in exit_))
    print(f"[δ={DELTA}] top-10 变动 {n_swap} 席：换入季节品 {n_enter_in}、被顶出的季节品 {n_exit_in}"
          f"（净流入季节品 {n_enter_in - n_exit_in}）", flush=True)

    assert len(score) == len(raw) and not np.isnan(score).any()
    out = pd.DataFrame({"user_id": df["user_id"], "item_id": df["item_id"], "score": score})
    p = PROJECT_ROOT / "outputs" / "candidate" / f"sample_submission_{NAME}.csv"
    out.to_csv(p, index=False, encoding="utf-8")

    meta = {"name": NAME,
            "desc": f"v6 fused + DELTA({DELTA})×1_{{seasonal在售}}；圣诞内容∈在售(9/10月)共 "
                    f"{len(seasonal_set)} 商品/{int(in_set.sum())} 行",
            "weights": W, "reuses": "cand-align-v6-fuse pkls (lgb/xgb/lr), 不重训",
            "keyword": KW.pattern, "alive_months": list(ALIVE_MONTHS),
            "n_seasonal_items": len(seasonal_set), "n_seasonal_rows": int(in_set.sum()),
            "score_cut": str(SC_CUT.date()), "feature_order": FEATS}
    (PROJECT_ROOT / "models" / f"candidate_meta_{NAME}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[4/4] {NAME} → {p.name}；总耗时 {time.time()-t0:.0f}s", flush=True)


def _user_slices(u_arr, order):
    """按 order（已按 u 稳定排序）切出每个 user 的 [start,end)。"""
    su = u_arr[order]
    n = len(su)
    start = 0
    while start < n:
        end = start + 1
        while end < n and su[end] == su[start]:
            end += 1
        yield su[start], slice(start, end)
        start = end


if __name__ == "__main__":
    main()
