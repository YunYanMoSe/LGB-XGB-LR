"""组长 v15 带标签 OOF 复合重算 + 跨架构融合选权（老师口径 0.4 GAUC + 0.4 NDCG@10 + 0.2 Recall@10）。

数据：
  * 组长 OOF data/candidate_aligned_oof_v15.csv（146,530 行 / 2010-08/09/10，正例 4.55%）
    ——含其内部分数 v15_calibrated / decomposed_product / v11_blend / blend_v15。
  * 我方 v6 分量 OOF outputs/candidate/replay/oof_eqg_preds.csv（187,730 行 Jul..Oct），
    每月行数与组长 OOF 逐位一致（45,399/48,097/53,034）⇒ (user,month,item) 可直接对齐。

目的：
  A. 组长各内部分数在老师复合（全等权 macro，scripts/59 同口径）下的真实分——它的 AUC 选权
     blend_v15 是不是真复合口径下的最优？还白丢多少分？
  B. 我方 v6 融合在同一对齐集上的复合分（基准）。
  C. 跨架构秩融合选权：leader×v6 在老师复合口径下的 argmax 权重（替代盲烧 .70/.85/.92），
     并读 w=0(纯 v6)/w=1(纯 leader) 两端 + 三子分解。

组 = (user, snapshot_month)；per-group 指标对组内单调变换不变 ⇒ 融合一律用全局百分位秩。

产物：outputs/candidate/replay/leader_oof_composite_report.txt + .json
用法：venv\\Scripts\\python.exe scripts\\74_leader_oof_composite.py
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
LEADER = ROOT / "data" / "candidate_aligned_oof_v15.csv"
OURS = ROOT / "outputs" / "candidate" / "replay" / "oof_eqg_preds.csv"
OUT = ROOT / "outputs" / "candidate" / "replay" / "leader_oof_composite_report.txt"
V6_W = {"lgb_macro_eq": 0.1, "xgb_macro_eq": 0.3, "lr_eq_l2_sp": 0.6}


# ---- scripts/59 同口径（逐 (user,month) 组等权 macro）----
def group_meta(df):
    g, _ = pd.factorize(df["user_id"].astype(str) + "_" + df["ym"].astype(str))
    lbl = df["label"].to_numpy(dtype="int64")
    n_g = np.bincount(g)
    p_g = np.bincount(g, weights=lbl.astype("float64"))
    return g, n_g, p_g, lbl


def group_metrics(s, g, n_g, p_g, lbl):
    sub = pd.DataFrame({"g": g, "s": np.asarray(s, dtype="float64"), "lbl": lbl})
    sub["ra"] = sub.groupby("g")["s"].rank(method="average")
    sp = np.bincount(g, weights=(sub["ra"].to_numpy() * lbl.astype("float64")))
    nn = n_g - p_g
    with np.errstate(divide="ignore", invalid="ignore"):
        auc_g = np.where((p_g > 0) & (nn > 0),
                         (sp - p_g * (p_g + 1.0) / 2.0) / (p_g * nn), np.nan)
    sub["rd"] = sub.groupby("g")["s"].rank(method="first", ascending=False)
    top = (sub["rd"].to_numpy() <= 10.0)
    hit = top & (lbl == 1)
    hits_g = np.bincount(g, weights=hit.astype("float64"))
    dcg_g = np.bincount(g, weights=np.where(
        hit, 1.0 / np.log2(sub["rd"].to_numpy() + 1.0), 0.0))
    denom = np.log2(np.arange(1, 11, dtype="float64") + 1.0)
    idcg = np.array([denom[:int(min(v, 10))].sum() for v in p_g])
    with np.errstate(divide="ignore", invalid="ignore"):
        ndcg_g = np.where(p_g > 0, dcg_g / np.where(idcg > 0, idcg, np.nan), np.nan)
        rec_g = np.where(p_g > 0, hits_g / p_g, np.nan)
    return auc_g, ndcg_g, rec_g


def comp(s, g, n_g, p_g, lbl):
    a, d, r = group_metrics(s, g, n_g, p_g, lbl)
    def macro(m):
        m = m[~np.isnan(m)]
        return float(m.mean()) if len(m) else float("nan")
    GAUC, NDCG, REC = macro(a), macro(d), macro(r)
    return 0.4 * GAUC + 0.4 * NDCG + 0.2 * REC, GAUC, NDCG, REC


def main() -> None:
    t0 = time.time()
    ld = pd.read_csv(LEADER)
    ow = pd.read_csv(OURS)
    ld["ym"] = pd.to_datetime(ld["snapshot_month"]).dt.strftime("%Y-%m")
    ow["ym"] = pd.to_datetime(ow["month"]).dt.strftime("%Y-%m")
    ld["user_id"] = ld["user_id"].astype("int64"); ld["item_id"] = ld["item_id"].astype(str)
    ow["user_id"] = ow["user_id"].astype("int64"); ow["item_id"] = ow["item_id"].astype(str)

    sub = ld.merge(
        ow[["ym", "user_id", "item_id", "lgb_macro_eq", "xgb_macro_eq", "lr_eq_l2_sp"]],
        on=["ym", "user_id", "item_id"], how="left", validate="one_to_one")
    assert len(sub) == len(ld) and sub[["lgb_macro_eq", "xgb_macro_eq", "lr_eq_l2_sp"]].notna().all().all()
    print(f"[align] 组长 OOF {len(ld):,} 行全部与 v6 OOF 对齐；每月核对 ok", flush=True)

    g, n_g, p_g, lbl = group_meta(sub)

    def pct(a): return rankdata(np.asarray(a, float)) / len(a)

    # 我方 v6 融合 = v6 原权×全局百分位秩（提交同管线）
    v6f = (V6_W["lgb_macro_eq"] * pct(sub.lgb_macro_eq.to_numpy())
           + V6_W["xgb_macro_eq"] * pct(sub.xgb_macro_eq.to_numpy())
           + V6_W["lr_eq_l2_sp"] * pct(sub.lr_eq_l2_sp.to_numpy()))
    sub["v6_fused"] = v6f

    L = "\n".join
    lines = []
    # ---- A/B. 各候选分数（leader 内部 + 我方 v6）在老师复合口径下的分 ----
    lines.append("## 老师复合（全等权 macro，scripts/59 口径）| 组=(user,snapshot_month) Aug..Oct 146,530 行")
    lines.append("composite = 0.4 GAUC + 0.4 NDCG@10 + 0.2 Recall@10")
    lines.append("")
    cand_cols = ["v15_calibrated", "decomposed_product", "v11_blend", "blend_v15",
                 "p_item_given_active", "p_activity", "v6_fused"]
    rows = []
    for c in cand_cols:
        cval, ga, nd, rc = comp(sub[c].to_numpy(float), g, n_g, p_g, lbl)
        rows.append((c, cval, ga, nd, rc))
    lines.append("| 分数 | composite | GAUC | NDCG@10 | Recall@10 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for c, cval, ga, nd, rc in sorted(rows, key=lambda x: -x[1]):
        lines.append(f"| {c:20s} | {cval:.5f} | {ga:.4f} | {nd:.4f} | {rc:.4f} |")
    lines.append("")

    # ---- C. 跨架构秩融合选权（leader 各候选 × 我方 v6）----
    def grid(nameA, arrA, nameB, arrB, wmax=1.0, step=0.05):
        pa, pb = pct(np.asarray(arrA, float)), pct(np.asarray(arrB, float))
        res = []
        for i in range(int(round(wmax / step)) + 1):
            w = round(i * step, 2)
            s = w * pa + (1.0 - w) * pb
            cval, ga, nd, rc = comp(s, g, n_g, p_g, lbl)
            res.append((w, cval, ga, nd, rc))
        res.sort(key=lambda x: -x[1])
        return res

    lines.append("## 跨架构秩融合选权（w = leader 权重；w=1 纯 leader，w=0 纯 v6）")
    for nameA, nameB, colA, colB in [
            ("leader blend_v15", "v6", "blend_v15", "v6_fused"),
            ("leader v15_cal", "v6", "v15_calibrated", "v6_fused"),
            ("leader v11_blend", "v6", "v11_blend", "v6_fused")]:
        res = grid(nameA, sub[colA].to_numpy(float), nameB, sub[colB].to_numpy(float))
        lines.append(f"-- {nameA} × {nameB} --")
        for w, cval, ga, nd, rc in res[:6]:
            lines.append(f"   w={w:.2f}  comp={cval:.5f}  (GAUC {ga:.4f} NDCG {nd:.4f} Rec {rc:.4f})")
        # 端点
        for w in (0.0, 1.0):
            pass
        lines.append("")
    # 细网格最佳段（leader blend × v6）
    res = grid("leader blend_v15", sub["blend_v15"].to_numpy(float),
               "v6", sub["v6_fused"].to_numpy(float), wmax=1.0, step=0.01)
    best = res[0]
    w1 = comp(sub["blend_v15"].to_numpy(float), g, n_g, p_g, lbl)[0]
    w0 = comp(sub["v6_fused"].to_numpy(float), g, n_g, p_g, lbl)[0]
    lines.append(f"[细网格 step .01] best w_leader={best[0]:.2f} comp={best[1]:.5f} "
                 f"| 纯 leader={w1:.5f} | 纯 v6={w0:.5f} | 跨融合增益 "
                 f"{best[1] - max(w0, w1):+.5f}")
    txt = "\n".join(lines) + "\n"
    OUT.write_text(txt, encoding="utf-8")
    print(txt)
    json.dump({"best_w_leader_vs_v6": best[0], "best_comp": best[1],
               "pure_leader_comp": w1, "pure_v6_comp": w0,
               "internal": {c: rows_row for c, rows_row in [(r[0], r[1:]) for r in rows]}},
              open(OUT.with_suffix(".json"), "w"), indent=2, ensure_ascii=False)
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
