"""Exp-H2：线性 Hypergraph 对照（item→basket→item 篮度归一传播 vs Exp-H1 规范化共现）。

H2 相似度 S_hg：对 incidence W(1/sqrt(|B|)) 做 C=W Wᵀ 再余弦对称归一 —— 等价于"消息沿超边
先按篮规模平均、再按商品参与度对称归一"。核心问题：线性超图传播是否只是另一种共现归一化。

对照口径与 H1 完全一致：同月份同 cutoff、同候选行、residual=组内百分位、alpha 在 09 选择 10 确认。

产物 outputs/experiment_basket_hypergraph/cooccurrence_vs_hypergraph.csv
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
from src.basket_experiment.metrics import composite_score, monthly_composite
from scipy.stats import spearmanr

H1 = "cos_bigbasket"
H2 = "hg_linear"
COL = "simnb_last3_mean"      # 自排除共购聚合（不依赖候选商品自身在篮里）
ALPHAS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.2]


def per_group_spearman_avg(a: pd.Series, b: pd.Series, grp: pd.Series) -> float:
    """组内(≥3 行且变化) Spearman 均值（衡量两者排序一致性，排除组间差异）。"""
    d = pd.DataFrame({"a": a, "b": b, "g": grp})
    rs = []
    for _, g in d.groupby("g"):
        if len(g) < 3 or g["a"].nunique() < 2 or g["b"].nunique() < 2:
            continue
        r = spearmanr(g["a"], g["b"]).correlation
        if np.isfinite(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def top10_overlap(a: pd.Series, b: pd.Series, grp: pd.Series) -> float:
    """组内按 a / b 各自 rank 的 top10 商品集 Jaccard 均值。"""
    d = pd.DataFrame({"a": a, "b": b, "g": grp})
    d["ra"] = d.groupby("g")["a"].rank(method="first", ascending=False)
    d["rb"] = d.groupby("g")["b"].rank(method="first", ascending=False)
    ov = []
    for _, g in d.groupby("g"):
        n = len(g)
        if n < 2:
            continue
        ta = set(g.loc[g["ra"] <= 10].index)
        tb = set(g.loc[g["rb"] <= 10].index)
        ov.append(len(ta & tb) / min(10, n))
    return float(np.mean(ov)) if ov else float("nan")


def main() -> None:
    t0 = time.time()
    io.OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean = io.load_clean_transactions()
    oof = io.load_anchor_oof()
    extra = pd.concat([oof["item_id"], io.load_nov_candidates()["item_id"]])
    vocab, mapping = F.build_item_vocab(clean, extra)
    store = F.build_baskets(clean, mapping)

    df1 = RF.features_for_scheme(store, len(vocab), oof, mapping, H1)
    df2 = RF.features_for_scheme(store, len(vocab), oof, mapping, H2)
    print(f"[feats] H1/H2 done {time.time()-t0:.0f}s", flush=True)

    j = df1[["ym", "user_id", "item_id", COL]].merge(
        df2[["ym", "user_id", "item_id", COL]], on=["ym", "user_id", "item_id"],
        suffixes=("_h1", "_h2"))
    r = j[[COL + "_h1", COL + "_h2"]].corr(method="pearson").iloc[0, 1]
    sp = per_group_spearman_avg(j[COL + "_h1"], j[COL + "_h2"],
                                j["user_id"].astype(str) + "_" + j["ym"])
    ov = top10_overlap(j[COL + "_h1"], j[COL + "_h2"],
                       j["user_id"].astype(str) + "_" + j["ym"])

    rows = []
    for name, df in (("h1_norm_cooc", df1), ("h2_linear_hg", df2)):
        df = E.add_resid(df, COL)
        rec = {"route": name, "scheme": (H1 if name.startswith("h1") else H2)}
        # alpha* 在 09
        sw9 = E.sweep(df[df["ym"] == "2010-09"], ALPHAS)
        a09 = sw9.loc[sw9["composite"].idxmax(), "alpha"]
        rec["alpha_on09"] = a09
        for ym in io.YM_ALL:
            sub = df[df["ym"] == ym]
            ca = composite_score(sub, E.ANCHOR_COL)
            cf = composite_score(sub.assign(_f=E.final_col(sub, a09)), "_f")
            rec[f"anchor_comp_{ym[-2:]}"] = ca[0]
            rec[f"final_comp_{ym[-2:]}"] = cf[0]
            rec[f"d_{ym[-2:]}"] = cf[0] - ca[0]
        rows.append(rec)

    out = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "comparison": "H1 vs H2",
        "pearson_rows": float(r),
        "spearman_group_mean": sp,
        "residual_top10_overlap": ov,
        "note": "高相关/高重叠 ⇒ 线性超图≈另一种共现归一化",
    }])
    final = pd.concat([summary, out], axis=1)
    final.to_csv(io.OUT_DIR / "cooccurrence_vs_hypergraph.csv", index=False)
    print(final.round(5).to_string(index=False))
    print(f"\n[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
