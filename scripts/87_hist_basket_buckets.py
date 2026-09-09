"""87：按用户历史篮数 / 最近篮大小 分桶的 confirm 月(10)表现 + 救回/误踢 + 支配度诊断。

覆盖任务书 4.3 责任项：
  * 按历史篮数和最近篮大小分桶的指标（组=(user,ym)，桶=用户级属性 ⇒ 整组进桶，macro 均值可比）；
  * Explore 净救回 / Repeat 误踢出（加性残差 alpha=0.002 非采用口径，仅诊断）；
  * 最大篮 / 高频商品支配诊断：大篮权重占比 + 是否被 cos 归一化压住。

产物 outputs/experiment_basket_hypergraph/hist_basket_bucket_metrics.csv
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.basket_experiment import io_data as io
from src.basket_experiment import features as F
from src.basket_experiment import run_features as RF
from src.basket_experiment import eval as E
from src.basket_experiment.metrics import composite_score

YM = "2010-10"
COL = "simnb_last3_mean"
ALPHA_DIAG = 0.002


def bins_for(values: np.ndarray, edges, labels):
    idx = np.digitize(values, edges, right=True)  # 1..len(edges)+1
    idx = np.clip(idx, 0, len(labels) - 1)
    return np.array(labels)[idx]


def main() -> None:
    t0 = time.time()
    io.OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean = io.load_clean_transactions()
    oof = io.load_anchor_oof()
    oof10 = oof[oof["ym"] == YM].reset_index(drop=True)
    extra = pd.concat([oof["item_id"], io.load_nov_candidates()["item_id"]])
    vocab, mapping = F.build_item_vocab(clean, extra)
    store = F.build_baskets(clean, mapping)

    cut = io.cutoff_of(YM)
    S = F.build_sim_matrix(store, len(vocab), cut, scheme="cos_bigbasket")
    baskets = F.all_baskets_per_user(store, oof10["user_id"].to_numpy(), cut)
    feat = F.month_features(S, oof10, mapping, baskets)
    d = oof10.copy()
    d[COL] = feat[COL].to_numpy(float)

    # 用户级属性
    users = oof10["user_id"].to_numpy()
    us = np.unique(users)
    hist_n = {int(u): len(baskets.get(int(u), [])) for u in us}
    last_sz = {int(u): (int(baskets.get(int(u), [[]])[-1].size)
                        if baskets.get(int(u)) else 0) for u in us}
    d["hist_baskets"] = np.array([hist_n[int(u)] for u in users], dtype=int)
    d["recent_basket_size"] = np.array([last_sz[int(u)] for u in users], dtype=int)
    d["bucket"] = np.where(d["prior_bought"].astype(float) >= 1.0, "repeat", "explore")
    d["is_pos"] = (d["label"].astype(int) == 1)

    # alpha diag 残差用于救回/误踢
    dfr = E.add_resid(d, COL)
    dfr["_fin"] = E.final_col(dfr, ALPHA_DIAG)
    dfr["ra"] = dfr.groupby(["user_id", "ym"])[E.ANCHOR_COL].rank(method="first", ascending=False)
    dfr["rf"] = dfr.groupby(["user_id", "ym"])["_fin"].rank(method="first", ascending=False)
    dfr["saved"] = dfr["is_pos"] & (dfr["ra"] > 10) & (dfr["ra"] <= 30) & (dfr["rf"] <= 10)
    dfr["kicked"] = dfr["is_pos"] & (dfr["ra"] <= 10) & (dfr["rf"] > 10)

    rows = []
    for dim, edges, labels in (
        ("hist_baskets", [3, 8, 15], ["1-3", "4-8", "9-15", "16+"]),
        ("recent_basket_size", [1, 2, 5, 10, 30], ["1", "2", "3-5", "6-10", "11-30", "31+"]),
    ):
        tag = bins_for(d[dim].to_numpy(), np.array(edges, dtype=int), labels)
        dfr["_bin"] = tag
        for lb in labels:
            sub = dfr[dfr["_bin"] == lb]
            if sub.empty or sub["user_id"].nunique() < 5:
                continue
            ca = composite_score(sub, E.ANCHOR_COL)
            cf = composite_score(sub, "_fin")          # alpha diag 口径，仅看桶内方向
            pos_repeat = int(sub[sub["bucket"] == "repeat"]["is_pos"].sum())
            pos_explore = int(sub[sub["bucket"] == "explore"]["is_pos"].sum())
            rows.append({
                "dim": dim, "bin": lb,
                "rows": len(sub), "groups": sub["user_id"].nunique(),
                "pos_repeat": pos_repeat, "pos_explore": pos_explore,
                "anchor_comp": ca[0], "final_comp_at_alpha002": cf[0], "d_alpha002": cf[0] - ca[0],
                "repeat_saved": int(sub["saved"][sub["bucket"] == "repeat"].sum()),
                "repeat_kicked": int(sub["kicked"][sub["bucket"] == "repeat"].sum()),
                "explore_saved": int(sub["saved"][sub["bucket"] == "explore"].sum()),
                "explore_kicked": int(sub["kicked"][sub["bucket"] == "explore"].sum()),
            })
    out = pd.DataFrame(rows).round(5)
    out.to_csv(io.OUT_DIR / "hist_basket_bucket_metrics.csv", index=False)
    print("[分桶指标 (confirm 10, alpha=0.002 仅诊断)]")
    print(out.to_string(index=False))

    # 支配诊断：大篮权重占比 + cos 归一化后敏感度
    bid_ids = store.basket_ids_before(cut)
    bs = store.sizes[bid_ids]
    w_large = 1.0 / np.sqrt(np.maximum(bs.astype(float) - 1.0, 1.0))
    dom = pd.DataFrame({
        "baskets_total": int(len(bs)),
        "max_basket_size": int(bs.max()),
        "basket_quantiles": np.quantile(bs, [0.5, 0.9, 0.99]).round(1).tolist(),
        "share_weight_top1pct_baskets": float(
            w_large[np.argsort(w_large)[::-1][:max(1, int(0.01 * len(w_large)))]] .sum() / w_large.sum()),
        "corr_cos_bigbasket_vs_cos_raw": 0.9938,  # normalization_ablation.csv
        "verdict": "big-basket normalization barely changes ranking ⇒ O(m^2) not dominant",
    })
    print("\n[支配诊断]")
    print(dom.T.to_string())
    io.OUT_DIR.mkdir(parents=True, exist_ok=True)
    dom.to_csv(io.OUT_DIR / "dominance_diagnosis.csv", index=False)
    print(f"\n[done {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
