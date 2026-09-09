"""Exp-H4：受约束残差学习（learned residual）—— H1/H2 特征 × {LR, 浅层 LGB} 对照。

理念（任务书 4.3 H4）：用一个比 anchor 小得多的可解释模型拟合"残差"，即预测哪些候选
被 anchor 低估（label=1 但 anchor 排低）。受约束 = 只允许篮共现族特征、极浅模型、
最后仍走同一 alpha 融合预算（final = anchor + alpha*group_percentile(pred)）。

协议（时间安全 & 冻结）：
  08 拟合初步排错 → 09 上选 alpha* 与配置 → 配置/alpha 冻结 → 用 08+09 重拟合 → 10 确认。
  模型特征只依赖 < snapshot 月的交易（同 H1/H2 特征管线）。

产物 outputs/experiment_basket_hypergraph/:
  h4_configs.csv                配置×（alpha09, 09/10 指标, 救回/误踢诊断）
  h4_saved_kicked_confirm.csv   确认月 10 上 anchor vs final 的分桶命中（diagnostic）
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.basket_experiment import io_data as io
from src.basket_experiment import features as F
from src.basket_experiment import run_features as RF
from src.basket_experiment import eval as E
from src.basket_experiment.metrics import composite_score

SEED = 42
ALPHAS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
MODEL_FEATS = RF.RESID_COLS + ["has_basket"]
SCHEMES = {"h1_norm_cooc": "cos_bigbasket", "h2_linear_hg": "hg_linear"}


def make_model(kind: str):
    if kind == "lr":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=0.1, max_iter=2000, random_state=SEED))
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=80, learning_rate=0.05, max_depth=2,
        num_leaves=4, min_child_samples=200, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.9, n_jobs=1, random_state=SEED, verbosity=-1)


def fit_score(df_tr, df_va, kind: str) -> np.ndarray:
    """在 df_tr（某月）行上 fit，返回 df_va 行的 predict_proba[:,1]。"""
    m = make_model(kind)
    Xtr = df_tr[MODEL_FEATS].to_numpy(float)
    m.fit(Xtr, df_tr["label"].to_numpy(float))
    return m.predict_proba(df_va[MODEL_FEATS].to_numpy(float))[:, 1]


def main() -> None:
    t0 = time.time()
    io.OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean = io.load_clean_transactions()
    oof = io.load_anchor_oof()
    extra = pd.concat([oof["item_id"], io.load_nov_candidates()["item_id"]])
    vocab, mapping = F.build_item_vocab(clean, extra)
    store = F.build_baskets(clean, mapping)

    feat = {}
    for tag, sch in SCHEMES.items():
        t1 = time.time()
        d = RF.features_for_scheme(store, len(vocab), oof, mapping, sch)
        feat[tag] = d
        print(f"[feats] {tag}: {len(d)} rows, {time.time()-t1:.0f}s", flush=True)

    # 每次 fit 都用严格更早的月份做特征（features_for_scheme 已按各 snapshot 月 cutoff 计算）
    df08 = {k: d[d["ym"] == "2010-08"] for k, d in feat.items()}
    df09 = {k: d[d["ym"] == "2010-09"] for k, d in feat.items()}
    df10 = {k: d[d["ym"] == "2010-10"] for k, d in feat.items()}

    rows = []
    sk_rows = []
    for tag in SCHEMES:
        for kind in ("lr", "lgb"):
            # --- 选型：fit on 08 → score 09 ---
            pr09 = fit_score(df08[tag], df09[tag], kind)
            df9 = df09[tag].copy()
            df9["model_p"] = pr09
            df9 = E.add_resid(df9, "model_p")
            sw = E.sweep(df9, ALPHAS)
            best = sw.loc[sw["composite"].idxmax(), "alpha"]
            # --- 确认：fit on 08+09 → score 10，用冻结的 best ---
            tr = pd.concat([df08[tag], df09[tag]], ignore_index=True)
            pr10 = fit_score(tr, df10[tag], kind)
            df10c = df10[tag].copy()
            df10c["model_p"] = pr10
            df10c = E.add_resid(df10c, "model_p")
            f10 = E.final_col(df10c, float(best))
            c10a = composite_score(df10c, E.ANCHOR_COL)
            c10f = composite_score(df10c.assign(_f=f10), "_f")
            # 09 自身 frozen-alpha 参考（供报告对比，不参与选择）
            f9 = E.final_col(df9, float(best))
            c9a = composite_score(df9, E.ANCHOR_COL)
            c9f = composite_score(df9.assign(_f=f9), "_f")

            # 确认月救回/误踢诊断（bucket 口径与 H1 相同）
            try:
                b10 = E.bucket_topk_metrics(df10c.assign(label=df10c["label"]),
                                            float(best), topk=10)
                b10.insert(0, "config", f"{tag}|{kind}")
                b10.insert(1, "ym", "2010-10")
                b10.insert(2, "alpha", float(best))
                sk_rows.append(b10)
            except Exception as e:
                print(f"  bucket_topk fail: {e}", flush=True)

            rows.append({
                "config": f"{tag}|{kind}", "features": tag, "model": kind,
                "alpha09": float(best),
                "comp09_anchor": c9a[0], "comp09_final": c9f[0], "d09": c9f[0] - c9a[0],
                "comp10_anchor": c10a[0], "comp10_final": c10f[0], "d10": c10f[0] - c10a[0],
            })
            print(f"  {tag}|{kind}: alpha09={best:.4f} d09={c9f[0]-c9a[0]:+.5f} "
                  f"d10(refit, frozen a)={c10f[0]-c10a[0]:+.5f}  [{time.time()-t0:.0f}s]", flush=True)

    cfg = pd.DataFrame(rows)
    cfg.to_csv(io.OUT_DIR / "h4_configs.csv", index=False)
    print("\n[h4_configs]")
    print(cfg.round(5).to_string(index=False))

    if sk_rows:
        sk = pd.concat(sk_rows, ignore_index=True)
        sk.to_csv(io.OUT_DIR / "h4_saved_kicked_confirm.csv", index=False)
        print("\n[confirm 10 bucket saved/kicked]")
        print(sk.round(5).to_string(index=False))

    # 记录配置清单摘要（供 consolidate 引用）
    (io.OUT_DIR / "h4_configs_summary.json").write_text(
        json.dumps({"seed": SEED, "features": list(SCHEMES), "models": ["lr", "lgb"],
                    "fit_ladder": "08->09 select; 08+09 refit -> 10 confirm",
                    "best_on09_config": cfg.loc[cfg["comp09_final"].idxmax(), "config"]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {time.time()-t0:.0f}s → {io.OUT_DIR}")


if __name__ == "__main__":
    main()
