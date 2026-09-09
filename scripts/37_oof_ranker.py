"""候选集打分 v2：LGBMRanker(group=user×month) 时间序 OOF —— 列表式学习对照 binary。

动机（真榜 0.7707 = v3-fuse 验证）：老师口径 ≈ 逐用户整列表质量（组内 AUC/NDCG 类），
LR（组 AUC .8659）胜过 top-K 向的 macro_sp。LGBMRanker 以 lambdarank/NDCG 直接在
每用户列表内学习排序 —— 与口径同构，理论上应进一步抬升逐用户列表质量。

协议与 scripts/34 相同：对目标月只用更早月份训练；行按 (user,month) 排序、组大小喂给 ranker；
对比项 = lgb_macro（scripts/34 结果）。评估同一套：global AUC / wgAUC / mgAUC + top-k。

用法：venv\\Scripts\\python.exe scripts\\37_oof_ranker.py
产物：outputs/candidate/replay/oof_ranker_report.txt + oof_ranker_preds.csv
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
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SEED
from src import candidate as cand
from src.candidate_v2 import V2_FEATS
from src.candidate import rank_metrics

REPLAY_DIR = PROJECT_ROOT / "outputs" / "candidate" / "replay"
FEATS = cand.FEATURES_CAND + V2_FEATS
MONTHS = [f"2010-{m:02d}-01" for m in range(6, 11)]
KS = (5, 10, 20)


def load_wide() -> dict[str, pd.DataFrame]:
    return {m: pd.read_csv(REPLAY_DIR / f"wide_v2_{m[:7]}.csv", dtype={"item_id": str})
            for m in MONTHS}


def gauc_by_group(df: pd.DataFrame, score_col: str, group_cols=("user_id", "month")):
    """逐组 AUC（python 循环，只用于 4 次评估，可接受）。"""
    ws, ms, ws_, cnt = 0.0, 0.0, 0.0, 0
    for _g, g in df.groupby(list(group_cols)):
        if g["label"].nunique() < 2:
            continue
        a = roc_auc_score(g["label"], g[score_col])
        w = len(g)
        ws += w * a
        ws_ += w
        ms += a
        cnt += 1
    return ws / ws_ if ws_ else float("nan"), ms / cnt if cnt else float("nan"), cnt


def eval_frame(df: pd.DataFrame, score_col: str) -> dict:
    out = {}
    out["global_auc"] = roc_auc_score(df["label"], df[score_col])
    wg, mg, ngrp = gauc_by_group(df, score_col)
    out["gAUC_w"], out["gAUC_m"] = wg, mg
    r = rank_metrics(df.rename(columns={"item_id": "stock_code"}), score_col, ks=KS)
    for k in KS:
        for m_ in ("hit", "prec", "rec"):
            out[f"{m_}@{k}"] = r.get(f"{m_}@{k}", float("nan"))
    return out


def _sorted_groups(df: pd.DataFrame):
    """按 (user,month) 排序，返回 (排序df, group_sizes)。"""
    df = df.sort_values(["user_id", "month"]).reset_index(drop=True)
    sizes = df.groupby(["user_id", "month"], sort=True).size().to_numpy(dtype="int64")
    return df, sizes


def run_ranker_oof(wide: dict[str, pd.DataFrame], report: list[str]) -> pd.DataFrame:
    from lightgbm import LGBMRanker
    months_ordered = sorted(wide)
    pred_parts = []
    for ti, tgt in enumerate(months_ordered):
        tr_m = months_ordered[:ti]
        if not tr_m:
            continue
        tr0 = pd.concat([wide[m] for m in tr_m], ignore_index=True)
        tr, q = _sorted_groups(tr0)
        te = wide[tgt]
        rk = LGBMRanker(
            objective="lambdarank", n_estimators=1000, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbosity=-1, deterministic=True, n_jobs=1)
        rk.fit(tr[FEATS], tr["label"], group=q)
        part = te[["user_id", "item_id", "label", "month"]].copy()
        part["score"] = rk.predict(te[FEATS]).astype(np.float64)
        pred_parts.append(part)
        report.append(f"  Ranker train={tr_m[0][:7]}..{tr_m[-1][:7]} -> eval {tgt[:7]}")
    return pd.concat(pred_parts, ignore_index=True)


def main() -> None:
    t0 = time.time()
    wide = load_wide()
    report: list[str] = []
    pf = run_ranker_oof(wide, report)
    ev = eval_frame(pf, "score")
    line = "  ".join(f"{k}={v:.4f}" for k, v in ev.items())
    print(f"[lgbm_ranker] {line}", flush=True)
    pf.to_csv(REPLAY_DIR / "oof_ranker_preds.csv", index=False, encoding="utf-8")

    lines = (["# LGBMRanker(group=user×month) v2 OOF；目标月 Jul-Oct；与 lgb_macro 对照",
              "# lgb_macro(scripts/34): global_auc=.8676 gAUC_w=.8428 gAUC_m=.8556 "
              "rec@10=.6320 hit@10=.9715", ""] +
             report + [f"LGBMRanker {line}", f"\n总耗时 {time.time()-t0:.0f}s"])
    (REPLAY_DIR / "oof_ranker_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[save] oof_ranker_report.txt；总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
