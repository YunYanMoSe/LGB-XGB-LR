"""候选集打分 v13（跨架构秩融合）：组长 v15 提交 × 我方 v6，多个 v15 权重烧上板。

背景（2026-09-09）：组长 v15（traincsv-v15-activity-moe 分解式：活动门控 + 首购/复购双专家
+ 前向校准 stacker）榜分 0.7845 >> 我方 v6 0.7782（+.0063）。组长已把 Nov 提交 csv
（data/submission_traincsv_v15_activity_moe_blend.csv）与代码 bundle 分享给我们。

跨架构融合依据：我方 v6（44 特征 pointwise eq-group 秩融合）与组长 v15（分解式）归纳偏置不同，
Nov 提交上 within-user rank Spearman=0.877、per-user top10 平均只重合 7.9/10 ⇒ 分歧席 ~2/用户，
若分歧是互补的，w·v15+(1−w)·v6 可望 > v15 单点。本地无法给 Nov 打分，用榜保留最佳+无限提交，
按自由彩票纪律烧 3 个权重（w_v15 ∈ {0.70, 0.85, 0.92}）。

产物：outputs/candidate/sample_submission_cand-align-v13-v15blend{70,85,92}.csv + meta
用法：venv\\Scripts\\python.exe scripts\\73_build_submission_v13_v15blend.py
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

ROOT = Path(__file__).resolve().parents[1]

LEADER_SUB = ROOT / "data" / "submission_traincsv_v15_activity_moe_blend.csv"
V6_SUB = ROOT / "outputs" / "candidate" / "sample_submission_cand-align-v6-fuse.csv"
CAND = ROOT / "train_data" / "sample_candidates.csv"
CLEAN = ROOT / "data" / "processed" / "leader_clean.csv"
WEIGHTS = [0.70, 0.85, 0.92]
CUT = pd.Timestamp("2010-11-01")


def main() -> None:
    t0 = time.time()
    cand = pd.read_csv(CAND, dtype={"user_id": str, "item_id": str})
    v15 = pd.read_csv(LEADER_SUB)
    v6 = pd.read_csv(V6_SUB)
    for df in (v15, v6):
        df["user_id"] = df["user_id"].astype(str)
        df["item_id"] = df["item_id"].astype(str)
    for nm, df in (("cand", cand), ("v15", v15), ("v6", v6)):
        assert df[["user_id", "item_id"]].equals(cand[["user_id", "item_id"]]), f"{nm} 对齐失败"

    N = len(cand)
    r15 = rankdata(v15["score"].to_numpy(float)) / N
    r6 = rankdata(v6["score"].to_numpy(float)) / N

    # ---- 分歧席诊断（v15 vs v6 每用户 top10，按 owned / 商品热度分类）----
    cl = pd.read_csv(CLEAN)
    cl["InvoiceDate"] = pd.to_datetime(cl["InvoiceDate"])
    hist = cl[cl["InvoiceDate"] < CUT]
    owned = set(zip(hist["CustomerID"].astype(str), hist["StockCode"].astype(str)))
    owned_arr = np.fromiter(
        (1 if (u, i) in owned else 0 for u, i in zip(cand["user_id"], cand["item_id"])),
        int, N)
    pop = hist.groupby("StockCode")["Quantity"].sum()
    pop_arr = np.fromiter(
        (float(pop.get(i, 0.0)) for i in cand["item_id"]), float, N)

    def top10_of(score_arr):
        cols = pd.DataFrame({
            "u": cand["user_id"].astype(str).to_numpy(),
            "row": np.arange(N),
            "s": np.asarray(score_arr, dtype=float),
        })
        top = {}
        for u, grp in cols.groupby("u", sort=False):
            top[u] = grp.nlargest(10, "s")["row"].to_numpy(int)
        return top

    def seat_stats(seat_rows):
        if len(seat_rows) == 0:
            return {"n": 0, "owned": 0.0, "log_pop_mean": 0.0}
        return {"n": len(seat_rows),
                "owned": float(owned_arr[seat_rows].mean()),
                "log_pop_mean": float(np.log1p(pop_arr[seat_rows]).mean())}

    t15 = top10_of(v15["score"].to_numpy(float))
    t6 = top10_of(v6["score"].to_numpy(float))
    v15_only_rows = []
    v6_only_rows = []
    for u in t15:
        s15, s6 = set(t15[u]), set(t6[u])
        v15_only_rows += [r for r in (s15 - s6)]
        v6_only_rows += [r for r in (s6 - s15)]
    v15_only_rows = np.array(v15_only_rows, dtype=int)
    v6_only_rows = np.array(v6_only_rows, dtype=int)
    print(f"[diag] top10 分歧席：v15 独占 {len(v15_only_rows)} / v6 独占 {len(v6_only_rows)}")
    for nm, rows in (("v15 独占", v15_only_rows), ("v6 独占", v6_only_rows)):
        st = seat_stats(rows)
        if st:
            print(f"  {nm}: n={st['n']} owned比例={st['owned']:.2f} "
                  f"log1p(units)均={st['log_pop_mean']:.2f}", flush=True)

    # ---- 融合写出 ----
    for w in WEIGHTS:
        score = w * r15 + (1.0 - w) * r6
        name = f"cand-align-v13-v15blend{int(w * 100):02d}"
        p = ROOT / "outputs" / "candidate" / f"sample_submission_{name}.csv"
        pd.DataFrame({"user_id": cand["user_id"], "item_id": cand["item_id"], "score": score}
                     ).to_csv(p, index=False, encoding="utf-8")
        meta = {"name": name,
                "desc": f"{w}×v15(v15 组长 submission) 秩 + {1.0-w:.2f}×v6(我方) 秩 全局百分位融合",
                "v15_source": str(LEADER_SUB.name), "v15_board": 0.7845,
                "v6_board": 0.7782, "w_v15": w,
                "within_user_rank_spearman_v15_vs_v6": 0.8767,
                "top10_overlap_mean_per_user": 7.94,
                "note": "leader 共享 submission，跨架构融合彩票；榜保留最佳，EV>=0"}
        (ROOT / "models" / f"candidate_meta_{name}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[out] {name} → {p.name}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
