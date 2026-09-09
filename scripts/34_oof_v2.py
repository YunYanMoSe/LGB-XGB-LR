"""候选集打分 v2：时间序 OOF 评测（协议同 scripts/30，但读 wide_v2_*、特征=33base+11v2）。

评测同一套：global AUC / 加权 gAUC / 宏 gAUC + 逐用户 macro top-k（老师口径）。
对目标月只用在它之前月份的样本训练 → 无泄漏。报告与 v1 差异用于特征增益判定。
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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SEED
from src import candidate as cand
from src.candidate_v2 import V2_FEATS
from src.candidate import rank_metrics

REPLAY_DIR = PROJECT_ROOT / "outputs" / "candidate" / "replay"
FEATS = cand.FEATURES_CAND + V2_FEATS          # 44
MONTHS = [f"2010-{m:02d}-01" for m in range(6, 11)]
KS = (5, 10, 20)
PFX = "wide_v2"


def load_wide() -> dict[str, pd.DataFrame]:
    out = {}
    for m in MONTHS:
        out[m] = pd.read_csv(REPLAY_DIR / f"{PFX}_{m[:7]}.csv", dtype={"item_id": str})
    return out


def gauc_by_group(df: pd.DataFrame, score_col: str, group_cols=("user_id", "month")):
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
    return (ws / ws_ if ws_ else float("nan"),
            ms / cnt if cnt else float("nan"), cnt)


def eval_frame(df: pd.DataFrame, score_col: str) -> dict:
    out = {}
    out["global_auc"] = roc_auc_score(df["label"], df[score_col])
    wg, mg, ngrp = gauc_by_group(df, score_col)
    out["gAUC_w"], out["gAUC_m"], out["n_groups"] = wg, mg, ngrp
    r = rank_metrics(df.rename(columns={"item_id": "stock_code"}), score_col, ks=KS)
    for k in KS:
        for m_ in ("hit", "prec", "rec"):
            out[f"{m_}@{k}"] = r.get(f"{m_}@{k}", float("nan"))
    return out


def _sw(weight: str, df: pd.DataFrame, spw: float) -> np.ndarray | None:
    if weight in ("none", "scale_pos"):
        return None
    nrow_u = df.groupby("user_id").size()
    macro = (1.0 / np.sqrt(nrow_u.reindex(df["user_id"]).to_numpy(dtype="float64")))
    if weight == "macro":
        return macro.astype("float64")
    if weight == "macro_sp":
        return (macro * np.where(df["label"] == 1, spw, 1.0)).astype("float64")
    raise ValueError(weight)


def run_lgb_oof(wide: dict[str, pd.DataFrame], weight: str, report: list[str]) -> pd.DataFrame:
    from lightgbm import LGBMClassifier
    months_ordered = sorted(wide)
    pred_parts = []
    for ti, tgt in enumerate(months_ordered):
        tr_m = months_ordered[:ti]
        if not tr_m:
            continue
        tr = pd.concat([wide[m] for m in tr_m], ignore_index=True)
        te = wide[tgt]
        spw = float((tr["label"] == 0).sum() / max(1, int(tr["label"].sum())))
        use_spw = 1.0 if weight == "macro_sp" else (spw if weight == "scale_pos" else 1.0)
        sw = _sw(weight, tr, spw)
        clf = LGBMClassifier(
            n_estimators=1000, learning_rate=0.05, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=use_spw,
            random_state=SEED, verbosity=-1, deterministic=True, n_jobs=1)
        clf.fit(tr[FEATS], tr["label"], sample_weight=sw)
        p = clf.predict_proba(te[FEATS])[:, 1].astype(np.float64)
        part = te[["user_id", "item_id", "label", "month"]].copy()
        part["score"] = p
        pred_parts.append(part)
    return pd.concat(pred_parts, ignore_index=True)


def main() -> None:
    t0 = time.time()
    wide = load_wide()
    report: list[str] = []
    preds: dict[str, pd.DataFrame] = {}
    for weight in ("scale_pos", "macro", "macro_sp"):
        pf = run_lgb_oof(wide, weight, report)
        preds[f"lgb_{weight}"] = pf
        ev = eval_frame(pf, "score")
        line = "  ".join(f"{k}={v:.4f}" for k, v in ev.items())
        report.append(f"LGB[{weight}] {line}")
        print(f"[lgb_{weight}] {line}", flush=True)

    from sklearn.linear_model import LogisticRegression as _LR
    months_ordered = sorted(wide)
    pred_parts = []
    for ti, tgt in enumerate(months_ordered):
        tr_m = months_ordered[:ti]
        if not tr_m:
            continue
        tr = pd.concat([wide[m] for m in tr_m], ignore_index=True)
        te = wide[tgt]
        lr = make_pipeline(StandardScaler(),
                           _LR(max_iter=2000, C=1.0, class_weight="balanced",
                               random_state=SEED, n_jobs=1))
        lr.fit(tr[FEATS], tr["label"])
        part = te[["user_id", "item_id", "label", "month"]].copy()
        part["score"] = lr.predict_proba(te[FEATS])[:, 1].astype(np.float64)
        pred_parts.append(part)
    pf = pd.concat(pred_parts, ignore_index=True)
    preds["lr"] = pf
    ev = eval_frame(pf, "score")
    report.append("LR[balanced] " + "  ".join(f"{k}={v:.4f}" for k, v in ev.items()))
    print("[lr] " + " ".join(f"{k}={v:.4f}" for k, v in ev.items()), flush=True)

    base = preds["lgb_scale_pos"][["user_id", "item_id", "label", "month"]].copy()
    for name, pf in preds.items():
        base[name] = pf["score"].to_numpy()
    base.to_csv(REPLAY_DIR / "oof_v2_preds.csv", index=False, encoding="utf-8")
    lines = (["# v2 OOF（44 特征 = 33 base + 11 v2；wide_v2_*；组=user×month；前月训练→后月预测）",
              ""] + report + [f"\n总耗时 {time.time()-t0:.0f}s"])
    (REPLAY_DIR / "oof_v2_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[save] oof_v2_preds.csv + oof_v2_report.txt；总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
