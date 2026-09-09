"""候选集打分 v2：OOF 融合选权（向量化 gAUC，在 oof_v2_preds.csv 上，不烧提交）。

候选分量：lgb_macro / lgb_macro_sp / lr（各自时间序 OOF 预测）。
融合：全局百分位秩（单调 ⇒ 保留各分量逐用户内部序）→ 加权平均。
选权网格在「宏 gAUC（逐 user×month 组 AUC 的宏平均，组内只一类则剔除）」上做
（快；老师口径=逐用户内部序，宏 gAUC 是该序的列表级代理，组长亦以 0.15 权重用它），
再对 gAUC_m 前列与单模型补全逐用户 macro top-k（hit/rec@k）与全局 AUC 作对照表。

用法：venv\\Scripts\\python.exe scripts\\35_fusion_oof.py
产物：outputs/candidate/replay/fusion_v2_report.txt
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.candidate import rank_metrics

REPLAY_DIR = PROJECT_ROOT / "outputs" / "candidate" / "replay"
COMPONENTS = ["lgb_macro", "lgb_macro_sp", "lr"]
KS = (5, 10, 20)


def vectorized_gAUC(df: pd.DataFrame, s: np.ndarray):
    """逐 (user,month) 组 AUC，加权与宏两种平均；组内只一类剔除。向量化，ms 级。"""
    sub = df[["user_id", "month"]].copy()
    sub["s"] = np.asarray(s, dtype="float64")
    sub["lbl"] = df["label"].to_numpy(dtype="int64")
    grp, uniq = pd.factorize(sub["user_id"].astype(str) + "_" + sub["month"].astype(str))
    sub["g"] = grp
    sub["r"] = sub.groupby("g")["s"].rank(method="average").to_numpy()
    is_pos = sub["lbl"] == 1
    pos = sub[is_pos]
    agg = sub.groupby("g").agg(n=("lbl", "size"), p=("lbl", "sum"))
    sp = pos.groupby("g")["r"].sum()
    agg = agg.join(sp.rename("sp"), how="left").fillna({"sp": 0.0})
    agg["n_neg"] = agg["n"] - agg["p"]
    ok = agg[(agg["p"] > 0) & (agg["n_neg"] > 0)].copy()
    ok["auc"] = (ok["sp"] - ok["p"] * (ok["p"] + 1.0) / 2.0) / (ok["p"] * ok["n_neg"])
    w = ok["n"]
    return float((w * ok["auc"]).sum() / w.sum()), float(ok["auc"].mean())


def main() -> None:
    preds = pd.read_csv(REPLAY_DIR / "oof_v2_preds.csv", dtype={"item_id": str})
    df = preds[["user_id", "item_id", "label", "month"]].copy()
    label = df["label"].to_numpy(dtype="int64")
    rmat = np.column_stack([rankdata(preds[c]) / len(preds) for c in COMPONENTS])

    lines = ["# v2 OOF 融合选权（全局百分位秩加权；组件=lgb_macro/lgb_macro_sp/lr）",
             f"样本 {len(df):,} / 目标月 Jul-Oct；行组=user×month", ""]
    # 单模型基准（同一向量化 gAUC + 全局 AUC）
    for j, c in enumerate(COMPONENTS):
        wg, mg = vectorized_gAUC(df, rmat[:, j])
        lines.append(f"单模型 {c:>13}: AUC={roc_auc_score(label, rmat[:, j]):.4f} "
                     f"gAUC_w={wg:.4f} gAUC_m={mg:.4f}")
    lines.append("")

    step = 0.05
    n = int(1 / step) + 1
    rows = []
    for a, b in itertools.product(range(n), repeat=2):
        c = n - 1 - a - b
        if c < 0:
            continue
        w = np.array([a, b, c], dtype="float64") * step
        if w.sum() <= 0:
            continue
        s = rmat @ w
        wg, mg = vectorized_gAUC(df, s)
        rows.append((tuple(round(float(x), 2) for x in w),
                     roc_auc_score(label, s), wg, mg))
    res = pd.DataFrame(rows, columns=["w", "auc", "gAUC_w", "gAUC_m"])
    res = res.sort_values("gAUC_m", ascending=False).reset_index(drop=True)

    lines.append("== 按 gAUC_m 降序 top8 ==")
    best_rows = []
    for _, r in res.head(8).iterrows():
        best_rows.append(r)
        lines.append(f"w={r['w']}  AUC={r['auc']:.4f} gAUC_w={r['gAUC_w']:.4f} "
                     f"gAUC_m={r['gAUC_m']:.4f}")

    lines.append("")
    lines.append("== top 组合与单模型 → 逐用户 macro top-k 对照 ==")
    # 给 top4 组合 + 单模型补 topk（rank_metrics 逐组但只跑几次）
    cand_w = [r["w"] for r in best_rows[:4]]
    combo = [("单 lgb_macro", np.array([1.0, 0.0, 0.0])),
             ("单 lgb_macro_sp", np.array([0.0, 1.0, 0.0])),
             ("单 lr", np.array([0.0, 0.0, 1.0]))] + \
            [(f"融合 {w}", np.array(w)) for w in cand_w]
    for nm, w in combo:
        s = rmat @ w
        df["s"] = s
        r = rank_metrics(df.rename(columns={"item_id": "stock_code"}), "s", ks=KS)
        lines.append(f"{nm:>24}: rec@5={r['rec@5']:.4f} rec@10={r['rec@10']:.4f} "
                     f"rec@20={r['rec@20']:.4f} | hit@5={r['hit@5']:.4f} "
                     f"hit@10={r['hit@10']:.4f} | prec@10={r['prec@10']:.4f}")

    txt = "\n".join(lines) + "\n"
    (REPLAY_DIR / "fusion_v2_report.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print("[save] fusion_v2_report.txt")


if __name__ == "__main__":
    main()
