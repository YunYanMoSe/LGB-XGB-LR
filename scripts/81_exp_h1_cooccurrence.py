"""Exp-H1：规范化篮共现基线（购物篮 → 商品相似度 → 候选残差信号）。

主方案 cos_bigbasket（大篮归一化 + 余弦）。消融方案见 ABLATIONS。
流程：特征只用 < snapshot 月的交易（时间安全）→ residual=组内百分位 → final=anchor+alpha·residual，
alpha 在 2010-09 上按 composite argmax 选择，2010-10 冻结确认（08 仅排错/诊断）。

产物 outputs/experiment_basket_hypergraph/:
    normalization_ablation.csv
    alpha_sweep_main.csv
    monthly_metrics_main_h1.csv
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

MAIN = "cos_bigbasket"
ABLATIONS = ["cos_raw", "jaccard", "cond", "cos_bigbasket_t90", "cos_bigbasket_t30"]
SCORE_COL = "sim_last3_mean"
ALPHAS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]


def setup():
    clean = io.load_clean_transactions()
    oof = io.load_anchor_oof()
    extra = pd.concat([oof["item_id"], io.load_nov_candidates()["item_id"]])
    vocab, mapping = F.build_item_vocab(clean, extra)
    store = F.build_baskets(clean, mapping)
    return clean, oof, vocab, mapping, store


def main() -> None:
    t0 = time.time()
    io.OUT_DIR.mkdir(parents=True, exist_ok=True)
    _clean, oof, vocab, mapping, store = setup()

    print(f"[data] vocab={len(vocab)} baskets={len(store.users)}", flush=True)

    # ---------- 主方案：逐月 alpha 稳定性 ----------
    print(f"[H1] main={MAIN}", flush=True)
    t1 = time.time()
    df_main = RF.features_for_scheme(store, len(vocab), oof, mapping, MAIN)
    df_main = E.add_resid(df_main, SCORE_COL)
    print(f"  features done {time.time()-t1:.0f}s rows={len(df_main)}", flush=True)

    # alpha_sweep：09 选型 / 10 确认 / 总体
    rows = []
    for a in ALPHAS:
        rec = {"alpha": a}
        for ym in io.YM_ALL:
            sub = df_main[df_main["ym"] == ym]
            c = composite_score(sub.assign(_f=E.final_col(sub, a)), "_f")
            rec[f"comp_{ym[-2:]}"] = c[0]
        c_all = composite_score(df_main.assign(_f=E.final_col(df_main, a)), "_f")
        rec["comp_all"] = c_all[0]
        rows.append(rec)
    sw = pd.DataFrame(rows)
    sw.to_csv(io.OUT_DIR / "alpha_sweep_main.csv", index=False)
    print("\n[alpha sweep main] 09 选型（粗览，含各月）:")
    print(sw.round(5).to_string(index=False))

    # 09 上选 alpha：整体 composite 最高的正 alpha（若 argmax=0 说明无提升）
    best = sw.loc[sw["comp_09"].idxmax(), "alpha"]
    print(f"\n[select] alpha* on 09 = {best}  (anchor 09 comp={sw.loc[sw.alpha==0,'comp_09'].iloc[0]:.5f})")

    mt = E.monthly_gain(df_main, float(best))
    mt.to_csv(io.OUT_DIR / "monthly_metrics_main_h1.csv", index=False)
    print("\n[monthly gain at alpha*]")
    print(mt.round(5).to_string(index=False))

    # ---------- 归一化消融 ----------
    print("\n[H1] ablations ...", flush=True)
    df_ref = df_main  # main 的 sim 列用于相关性
    ab_rows = []
    for sch in [MAIN] + ABLATIONS:
        t2 = time.time()
        df_s = RF.features_for_scheme(store, len(vocab), oof, mapping, sch)
        df_s = E.add_resid(df_s, SCORE_COL)
        # 该方案 09 上 alpha*
        sw9 = E.sweep(df_s[df_s["ym"] == "2010-09"], ALPHAS)
        a09 = sw9.loc[sw9["composite"].idxmax(), "alpha"]
        g09 = E.monthly_gain(df_s[df_s["ym"] == "2010-09"], float(a09), months=["2010-09"])
        g10 = E.monthly_gain(df_s[df_s["ym"] == "2010-10"], float(a09), months=["2010-10"])
        # 与主方案 sim 行级相关（仅 >0 且同月同键）
        j = df_ref[["ym", "user_id", "item_id", SCORE_COL]].merge(
            df_s[["ym", "user_id", "item_id", SCORE_COL]], on=["ym", "user_id", "item_id"],
            suffixes=("_main", "_s"))
        corr = j[j[SCORE_COL + "_s"] > 0][SCORE_COL + "_main"].corr(
            j[j[SCORE_COL + "_s"] > 0][SCORE_COL + "_s"])
        ab_rows.append({
            "scheme": sch, "alpha09": a09,
            "comp09_anchor": g09.anchor_comp.iloc[0], "comp09_final": g09.final_comp.iloc[0],
            "d09": g09.d_comp.iloc[0],
            "comp10_anchor": g10.anchor_comp.iloc[0], "comp10_final": g10.final_comp.iloc[0],
            "d10": g10.d_comp.iloc[0],
            "corr_sim_with_main": float(corr) if pd.notna(corr) else np.nan,
        })
        print(f"  {sch:20s} alpha09={a09:.3f} d09={g09.d_comp.iloc[0]:+.5f} "
              f"d10={g10.d_comp.iloc[0]:+.5f} corr={corr:.3f}  [{time.time()-t2:.0f}s]", flush=True)
    abl = pd.DataFrame(ab_rows)
    abl.to_csv(io.OUT_DIR / "normalization_ablation.csv", index=False)
    print("\n[normalization ablation]")
    print(abl.round(5).to_string(index=False))

    # anchor 月度参照
    ma = monthly_composite(oof, "blend_v15")
    ma.to_csv(io.OUT_DIR / "monthly_metrics_anchor.csv", index=False)
    print(f"\n[done] {time.time()-t0:.0f}s → {io.OUT_DIR}")


if __name__ == "__main__":
    main()
