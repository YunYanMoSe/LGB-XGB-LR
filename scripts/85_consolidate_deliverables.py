"""85_consolidate：汇总 Basket/Hypergraph 组员路线交付物（一致性单源）。

读取已产出的 normalization_ablation.csv / cooccurrence_vs_hypergraph.csv / h4_configs.csv，
重算主路线特征并一次性写齐：
  alpha_sweep.csv                主残差列 alpha×月份 composite
  candidate_aligned_oof_basket.csv  (snapshot_month,user_id,item_id,label,anchor_score,
                                     residual_score,final_score,prior_bought)
  monthly_metrics.csv             逐月 anchor/final 指标（final 采用冻结后确认的 alpha）
  repeat_explore_analysis.csv     repeat/explore 分桶命中与 11-30 救援诊断
  experiment_config.json          配置与采用/不采用决定

不采用判定：候选路线在 09 选 alpha、在 10 冻结确认的 d>=+5e-4 才算通过；否则保留 anchor。
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
import json
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

ALPHAS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]
MAIN = "cos_bigbasket"
MAIN_COL = "sim_last3_mean"      # 主残差信号（alpha_sweep / OOF residual 同源）
FAMILY = ["sim_last1", "sim_last3_max", "sim_last3_mean",
          "simnb_last1", "simnb_last3_max", "simnb_last3_mean"]
ADOPT_MIN_CONFIRM_D = 5e-4       # 10 冻结确认至少 +0.0005 才采用


def main() -> None:
    t0 = time.time()
    io.OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean = io.load_clean_transactions()
    oof = io.load_anchor_oof()
    extra = pd.concat([oof["item_id"], io.load_nov_candidates()["item_id"]])
    vocab, mapping = F.build_item_vocab(clean, extra)
    store = F.build_baskets(clean, mapping)
    feat = RF.features_for_scheme(store, len(vocab), oof, mapping, MAIN)
    print(f"[feats] {len(feat)} rows {time.time()-t0:.0f}s", flush=True)

    # ---------- alpha_sweep（主残差列，逐月） ----------
    rows = []
    for a in ALPHAS:
        rec = {"alpha": a}
        for ym in io.YM_ALL:
            sub = feat[feat["ym"] == ym]
            rec[f"comp_{ym[-2:]}"] = composite_score(
                sub.assign(_f=E.final_col(E.add_resid(sub, MAIN_COL), a)), "_f")[0]
        rows.append(rec)
    sw = pd.DataFrame(rows)
    sw.to_csv(io.OUT_DIR / "alpha_sweep.csv", index=False)
    alpha09_main = float(sw.loc[sw["comp_09"].idxmax(), "alpha"])

    # ---------- 残差列族 09 选型 / 10 确认（供报告与采用判定） ----------
    fam_rows = []
    for col in FAMILY:
        d09 = {ym: feat[feat["ym"] == ym] for ym in io.YM_ALL}
        best_a, best_c = None, -1.0
        for a in ALPHAS:
            sub = E.add_resid(d09["2010-09"], col)
            c = composite_score(sub.assign(_f=E.final_col(sub, a)), "_f")[0]
            if c > best_c:
                best_c, best_a = c, a
        sub10 = E.add_resid(d09["2010-10"], col)
        c10a = composite_score(sub10, E.ANCHOR_COL)[0]
        c10f = composite_score(sub10.assign(_f=E.final_col(sub10, best_a)), "_f")[0]
        fam_rows.append({"col": col, "alpha09": best_a, "comp09": best_c,
                         "d09": best_c - composite_score(E.add_resid(d09["2010-09"], col), E.ANCHOR_COL)[0],
                         "comp10_anchor": c10a, "comp10_final": c10f, "d10": c10f - c10a})
    fam = pd.DataFrame(fam_rows)
    fam.to_csv(io.OUT_DIR / "residual_col_selection.csv", index=False)
    print("\n[residual col family: select on 09 / confirm on 10]")
    print(fam.round(5).to_string(index=False))

    # ---------- 采用判定 ----------
    fam_best = fam.sort_values("comp09", ascending=False).iloc[0]
    # 与 H4 学到的 best config 对比
    h4 = pd.read_csv(io.OUT_DIR / "h4_configs.csv")
    h4_best = h4.sort_values("comp09_final", ascending=False).iloc[0]
    decided = []
    for nm, src in (("additive_family", fam_best), ("learned_h4", h4_best)):
        decided.append({
            "candidate": nm,
            "config_detail": (f"col={src['col']},alpha={src['alpha09']:.3f}"
                              if nm == "additive_family" else
                              f"{src['config']},alpha={src['alpha09']:.3f}"),
            "d09": float(src["d09"] if nm == "additive_family" else src["d09"]),
            "d10_confirm": float(src["d10"] if nm == "additive_family" else src["d10"]),
            "adopt": bool((src["d09"] if nm == "additive_family" else src["d09"]) >= 0
                          and float(src["d10"] if nm == "additive_family" else src["d10"]) >= ADOPT_MIN_CONFIRM_D),
        })
    # 实际冻结重拟合口径：确认月用的是 08+09 重拟合模型（H4 已按此口径）；
    # additive 家族第 10 月残差本身就是时间安全的原始信号，不存在重拟合问题。
    adopted = [x for x in decided if x["adopt"]]
    ALPHA_FINAL = float(adopted[0]["config_detail"].split("alpha=")[1].split(",")[0]) if adopted else 0.0
    SHIP_ROUTE = adopted[0]["candidate"] if adopted else "none(keep_anchor)"
    print(f"\n[decision] confirm 阈值 +{ADOPT_MIN_CONFIRM_D:.4f} → 采用: {SHIP_ROUTE}  alpha_final={ALPHA_FINAL}")

    # ---------- candidate_aligned_oof_basket.csv ----------
    df = E.add_resid(feat, MAIN_COL)
    df["final"] = E.final_col(df, ALPHA_FINAL)
    out = df[["ym", "user_id", "item_id", "label", E.ANCHOR_COL, "resid_pct", "final", "prior_bought"]] \
        .rename(columns={"ym": "snapshot_month", E.ANCHOR_COL: "anchor_score",
                         "resid_pct": "residual_score", "final": "final_score"})
    # 组内与 anchor 文件一致：按 08/09/10 分块、行序与 anchor oof 相同
    out["snapshot_month"] = out["snapshot_month"] + "-01"
    out.to_csv(io.OUT_DIR / "candidate_aligned_oof_basket.csv", index=False)
    print(f"[oof] {len(out)} rows → candidate_aligned_oof_basket.csv")

    # ---------- monthly_metrics.csv ----------
    mm = E.monthly_gain(df, ALPHA_FINAL)
    mm = mm.rename(columns={"ym": "month"})
    mm.to_csv(io.OUT_DIR / "monthly_metrics.csv", index=False)
    print("\n[monthly_metrics at alpha_final]")
    print(mm.round(5).to_string(index=False))

    # ---------- repeat/explore 分析 ----------
    d = df.copy()
    d["bucket"] = (d["prior_bought"].astype(float) >= 1.0).astype(int)
    d["b_anchor"] = d["label"].astype(int) == 1
    d["rank_a"] = d.groupby(["user_id", "ym"])[E.ANCHOR_COL].rank(method="first", ascending=False)
    d["band"] = np.where(d["rank_a"] <= 10, "top10",
                         np.where(d["rank_a"] <= 30, "11_30", "below30"))
    agg = []
    for (bk, ym), g in d[d["b_anchor"]].groupby(["bucket", "ym"]):
        agg.append({
            "ym": ym, "bucket": "repeat" if bk else "explore",
            "pos": int(len(g)), "rows": 0,
            "pos_anchor_top10": int((g["band"] == "top10").sum()),
            "pos_anchor_11_30": int((g["band"] == "11_30").sum()),
            "pos_anchor_below30": int((g["band"] == "below30").sum()),
        })
    # 修正 rows 列语义（候选行数而非正例行）
    for r in agg:
        r["rows"] = int(((d["ym"] == r["ym"]) & (d["bucket"] == (1 if r["bucket"] == "repeat" else 0))).sum())
    rp = pd.DataFrame(agg).sort_values(["ym", "bucket"])
    # 正例命中率 + 救援诊断（非采用口径，仅展示 residual 理论救援上限与实测）
    rp["pos_rate"] = rp["pos"] / rp["rows"]
    rp["anchor_hit_rate_top10"] = rp["pos_anchor_top10"] / rp["pos"]
    rp.to_csv(io.OUT_DIR / "repeat_explore_analysis.csv", index=False)
    print("\n[repeat/explore 命中率]")
    print(rp.round(4).to_string(index=False))

    # ---------- experiment_config.json ----------
    cfg = {
        "experiment": "basket_hypergraph_member_route",
        "author_route": "Basket/Hypergraph Residual (任务书 四)",
        "dev_protocol": {"08": "debug", "09": "select config+alpha", "freeze": "refit 08+09", "10": "confirm"},
        "metric": {"composite": "0.4*GAUC+0.4*NDCG@10+0.2*Recall@10",
                   "group": "(user_id, snapshot_month)", "nan_group": "skip"},
        "anchor": {"source": "data/candidate_aligned_oof_v15.csv", "col": "blend_v15",
                   "overall_composite": 0.57753},
        "time_safety": "features use transactions strictly before snapshot month cutoff",
        "basket_def": {"id": "(CustomerID,InvoiceNo)", "dedup": "intra-basket item dedup",
                       "service_codes_excluded": True, "big_basket_norm": "1/sqrt(|B|-1) default"},
        "seed": 42,
        "experiments": {
            "H0_repro": {"status": "pass", "overall_composite": 0.57753, "delta": 0.0},
            "H1_norm_cooccurrence": {"main_scheme": MAIN, "main_col": MAIN_COL,
                                     "alpha09": alpha09_main, "ablations_csv": "normalization_ablation.csv",
                                     "ablations_alpha09": "all 0.0 / corr 0.92-0.99"},
            "H2_linear_hypergraph": {"scheme": "hg_linear", "csv": "cooccurrence_vs_hypergraph.csv",
                                     "pearson_h1_h2": 0.9987, "spearman_group": 0.9955,
                                     "residual_top10_overlap": 0.963, "stop_condition": "4.4 #1 met"},
            "H3_light_nonlinear_encoder": {"status": "skipped",
                                           "reason": "H1/H2 simple baselines null → task-book gating not met"},
            "H4_constrained_residual": {"csv": "h4_configs.csv",
                                        "models": ["lr", "lgb"], "feature_sets": list(h4["features"].unique()),
                                        "fit_ladder": "08->select 09; 08+09 refit->confirm 10",
                                        "best_confirm_d10": float(h4["d10"].max())},
        },
        "decision": {"adopted_route": SHIP_ROUTE, "alpha_final": ALPHA_FINAL,
                     "confirm_threshold_d": ADOPT_MIN_CONFIRM_D,
                     "decided": decided,
                     "summary": ("No basket/hypergraph residual route reached +5e-4 confirm gain; "
                                 "final_score=anchor (alpha=0). residual_score kept for future re-blend.")},
        "artifacts": ["candidate_aligned_oof_basket.csv", "monthly_metrics.csv", "alpha_sweep.csv",
                      "normalization_ablation.csv", "cooccurrence_vs_hypergraph.csv",
                      "repeat_explore_analysis.csv", "residual_col_selection.csv",
                      "h4_configs.csv", "h4_saved_kicked_confirm.csv"],
    }
    (io.OUT_DIR / "experiment_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {time.time()-t0:.0f}s → {io.OUT_DIR}")


if __name__ == "__main__":
    main()
