"""候选集打分任务 ⑩（v9 对标线）：候选对齐回放 —— 时间序 OOF 评测（前月训练→后月预测）。

回放样本按时间折训练：对目标月 t，只用「严格早于 t 的月份」训练，预测 t 月的候选行。
这复刻组长 v9 的「模型不能使用目标月或更晚月份产生该月预测」，避免任何目标月信息泄漏。

评估口径（同时报告，便于双对照）：
    * 全局 AUC / 加权 gAUC / 宏 gAUC —— 组长表 IV 同款，用来对「OOF 绝对量级」是否 ~0.88/0.85/0.86；
    * 逐用户 macro top-k（hit/prec/rec@k，按用户等权）—— 我们用探针 A/B 锁定的老师口径。
gAUC 的用户-月份分组：同一 (user, month) 为一组，组内算 AUC 再加权/宏平均（回放天然每用户每月一组）。

权重策略：
    * scale_pos  = 训练集负/正比（回放 4.37% 基率 → ~22×），保留回放低基率的本来面目；
    * macro      = 每行 1/sqrt(该用户当月行数) —— 贴「逐用户 macro」口径，让每用户(每月)在损失中近似等权；
    * macro×sp   = 两者相乘（v3 同款配方，放到回放上复验）。

用法：venv\\Scripts\\python.exe scripts\\30_oof_replay.py
产物：outputs/candidate/replay/oof_{lgb,lr}.txt + 预测宽表（供融合/堆叠复用）
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
from src.candidate import rank_metrics

REPLAY_DIR = PROJECT_ROOT / "outputs" / "candidate" / "replay"
FEATS = cand.FEATURES_CAND
MONTHS = [f"2010-{m:02d}-01" for m in range(6, 11)]
KS = (5, 10, 20)


def load_wide() -> dict[str, pd.DataFrame]:
    out = {}
    for m in MONTHS:
        out[m] = pd.read_csv(REPLAY_DIR / f"wide_{m[:7]}.csv", dtype={"item_id": str})
    return out


def gauc_by_group(df: pd.DataFrame, score_col: str, group_cols=("user_id", "month")):
    """逐 (user,month) 组 AUC；返回 (weighted_gAUC, macro_gAUC, n_groups_ok, group_rows)。
    weight = 组内样本数；AUC 不可算（只一类）的组跳过（leader 同款语义）。"""
    ws, ms, ws_, cnt = 0.0, 0.0, 0.0, 0
    group_rows = 0
    for _g, g in df.groupby(list(group_cols)):
        if g["label"].nunique() < 2:
            continue
        a = roc_auc_score(g["label"], g[score_col])
        w = len(g)
        ws += w * a
        ws_ += w
        ms += a
        cnt += 1
        group_rows += w
    return (ws / ws_ if ws_ else float("nan"),
            ms / cnt if cnt else float("nan"), cnt, group_rows)


def eval_frame(df: pd.DataFrame, score_col: str) -> dict:
    """单帧综合评估：全局 AUC / gAUC 两种 + 逐用户 macro top-k。"""
    out = {}
    out["global_auc"] = roc_auc_score(df["label"], df[score_col])
    wg, mg, ngrp, nrow = gauc_by_group(df, score_col)
    out["gAUC_w"] = wg
    out["gAUC_m"] = mg
    out["n_groups"] = ngrp
    out["n_rows"] = nrow
    r = rank_metrics(df.rename(columns={"item_id": "stock_code"}), score_col, ks=KS)
    for k in KS:
        for m_ in ("hit", "prec", "rec"):
            out[f"{m_}@{k}"] = r.get(f"{m_}@{k}", float("nan"))
    return out


def _sw(weight: str, df: pd.DataFrame, spw: float) -> np.ndarray | None:
    if weight in ("none", "scale_pos"):
        return None
    nrow_u = df.groupby("user_id")["month"].size() if False else df.groupby("user_id").size()
    macro = (1.0 / np.sqrt(nrow_u.reindex(df["user_id"]).to_numpy(dtype="float64")))
    if weight == "macro":
        return macro.astype("float64")
    if weight == "macro_sp":
        return (macro * np.where(df["label"] == 1, spw, 1.0)).astype("float64")
    raise ValueError(weight)


def run_lgb_oof(wide: dict[str, pd.DataFrame], weight: str,
                report: list[str]) -> pd.DataFrame:
    from lightgbm import LGBMClassifier
    months_ordered = sorted(wide)
    pred_parts = []
    for ti, tgt in enumerate(months_ordered):
        tr_m = months_ordered[:ti]
        if not tr_m:
            continue  # 目标月之前必须至少有一个训练月
        tr = pd.concat([wide[m] for m in tr_m], ignore_index=True)
        te = wide[tgt]
        spw = float((tr["label"] == 0).sum() / max(1, int(tr["label"].sum())))
        # scale_pos 走 LGB 内建 scale_pos_weight；macro / macro_sp 走 sample_weight。
        # macro_sp 的行权重已含 spw ⇒ scale_pos_weight 保持 1.0，避免二次放大。
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
        rep = f"  LGB[{weight}] train={tr_m[0][:7]}..{tr_m[-1][:7]} -> eval {tgt[:7]}"
        report.append(rep)
    return pd.concat(pred_parts, ignore_index=True)


def run_lr_oof(wide: dict[str, pd.DataFrame], report: list[str]) -> pd.DataFrame:
    months_ordered = sorted(wide)
    pred_parts = []
    for ti, tgt in enumerate(months_ordered):
        tr_m = months_ordered[:ti]
        if not tr_m:
            continue
        tr = pd.concat([wide[m] for m in tr_m], ignore_index=True)
        te = wide[tgt]
        spw = float((tr["label"] == 0).sum() / max(1, int(tr["label"].sum())))
        lr = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=2000, C=1.0,
                                              class_weight="balanced",
                                              random_state=SEED, n_jobs=1))
        lr.fit(tr[FEATS], tr["label"])
        p = lr.predict_proba(te[FEATS])[:, 1].astype(np.float64)
        part = te[["user_id", "item_id", "label", "month"]].copy()
        part["score"] = p
        pred_parts.append(part)
    return pd.concat(pred_parts, ignore_index=True)


def main() -> None:
    t0 = time.time()
    wide = load_wide()
    print(f"[load] 月份：{sorted(wide)}", flush=True)
    report: list[str] = []

    preds: dict[str, pd.DataFrame] = {}
    for weight in ("scale_pos", "macro", "macro_sp"):
        pf = run_lgb_oof(wide, weight, report)
        preds[f"lgb_{weight}"] = pf
        ev = eval_frame(pf, "score")
        report.append(f"  ==> LGB[{weight}] " + "  ".join(
            f"{k}={v:.4f}" for k, v in ev.items()))
        print(f"[lgb_{weight}] " + "  ".join(f"{k}={v:.4f}" for k, v in ev.items()), flush=True)

    pf = run_lr_oof(wide, report)
    preds["lr"] = pf
    ev = eval_frame(pf, "score")
    report.append("  ==> LR[balanced] " + "  ".join(f"{k}={v:.4f}" for k, v in ev.items()))
    print("[lr] " + " ".join(f"{k}={v:.4f}" for k, v in ev.items()), flush=True)

    # 落盘预测宽表（融合/堆叠复用）—— 只合并 user/item/label/month + 各模型 score
    base = preds["lgb_scale_pos"][["user_id", "item_id", "label", "month"]].copy()
    for name, pf in preds.items():
        base[name] = pf["score"].to_numpy()
    base.to_csv(REPLAY_DIR / "oof_preds.csv", index=False, encoding="utf-8")

    lines = [f"# 候选对齐回放 时间序 OOF（前月训练→后月预测，组=user×month）",
             f"# 评估：global AUC / 加权 gAUC / 宏 gAUC / 逐用户 macro top-k（探针口径）",
             f"# 训练月序：{sorted(wide)}；对目标月只用严格早于它的月份",
             ""] + report + [f"\n总耗时 {time.time()-t0:.0f}s"]
    (REPLAY_DIR / "oof_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[save] oof_preds.csv + oof_report.txt；总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
