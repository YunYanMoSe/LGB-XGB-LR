"""86：H3 门控诊断 —— 为什么 co-occurrence 残差救不回 explore？

量化 confirm 月 10 上：repeat/explore × 正负例 的篮共现信号密度（sim>0 占比、均值、
has_basket 占比、nb_item_baskets）。若 explore 候选（尤其被 anchor 排到 top30 外的正例）
篮共现几乎为空，则任何基于"用户历史篮 ↔ 候选"的线性/浅层信号都无法把它拉进 top10 ——
这正是 H1/H2/H4 空结果、且非线性编码器（H3）也不会改变瓶颈的解释。

产物 outputs/experiment_basket_hypergraph/explore_rescue_diagnosis.csv
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
from src.basket_experiment.eval import ANCHOR_COL

YM = "2010-10"


def main() -> None:
    t0 = time.time()
    clean = io.load_clean_transactions()
    oof = io.load_anchor_oof()
    oof10 = oof[oof["ym"] == YM].copy()
    extra = pd.concat([oof["item_id"], io.load_nov_candidates()["item_id"]])
    vocab, mapping = F.build_item_vocab(clean, extra)
    store = F.build_baskets(clean, mapping)
    f = RF.features_for_scheme(store, len(vocab), oof10, mapping, "cos_bigbasket",
                               months=[YM])
    d = oof10.reset_index(drop=True)
    d["sim"] = f["simnb_last3_mean"].to_numpy(float)   # 自排除共购（更纯的篮邻居信号）
    d["has_b"] = f["has_basket"].to_numpy(float)
    d["nb"] = f["nb_item_baskets"].to_numpy(float)
    d["bucket"] = np.where(d["prior_bought"].astype(float) >= 1.0, "repeat", "explore")
    d["pos"] = (d["label"].astype(int) == 1).astype(int)
    d["rank_a"] = d.groupby(["user_id", "ym"])[ANCHOR_COL].rank(method="first", ascending=False)
    d["band"] = np.where(d["rank_a"] <= 10, "top10",
                         np.where(d["rank_a"] <= 30, "11_30", "below30"))

    rows = []
    for (bk, p), g in d.groupby(["bucket", "pos"]):
        rows.append({
            "ym": YM, "bucket": bk, "label": int(p), "rows": len(g),
            "frac_sim_gt0": float((g["sim"] > 0).mean()),
            "mean_sim": float(g["sim"].mean()),
            "frac_has_basket": float(g["has_b"].mean()),
            "mean_nb_baskets": float(g["nb"].mean()),
        })
    # 正例按 anchor 排位分带看信号密度
    for bk in ("repeat", "explore"):
        g = d[(d["bucket"] == bk) & (d["pos"] == 1)]
        for band in ("top10", "11_30", "below30"):
            h = g[g["band"] == band]
            rows.append({
                "ym": YM, "bucket": f"pos|{bk}|{band}", "label": 1, "rows": len(h),
                "frac_sim_gt0": float((h["sim"] > 0).mean()) if len(h) else np.nan,
                "mean_sim": float(h["sim"].mean()) if len(h) else np.nan,
                "frac_has_basket": float(h["has_b"].mean()) if len(h) else np.nan,
                "mean_nb_baskets": float(h["nb"].mean()) if len(h) else np.nan,
            })
    out = pd.DataFrame(rows).round(4)
    io.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(io.OUT_DIR / "explore_rescue_diagnosis.csv", index=False)
    print(out.to_string(index=False))
    print(f"\n[done {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
