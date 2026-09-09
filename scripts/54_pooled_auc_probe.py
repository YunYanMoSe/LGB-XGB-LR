"""探针信息验证：老师口径≈全局池化 AUC ⇒ 重新用池化 AUC 选融合权。

组长 3 探针测得老师评分主要是全局池化 AUC（非逐组宏平均）。我们此前按 gAUC_m（逐
(user,month) 组 AUC 宏平均）选权，6 轮与榜一致；但池化 AUC 与 gAUC_m 的最优权向量
可能不同。存量 OOF（187,730 行 Jul..Oct，不重训）上：
    * 每个分量同时报 池化AUC 与 gAUC_m —— 看谁全局校准更强；
    * 三分量 fine-grid 分别按 池化AUC 与 gAUC_m 取 argmax，对比冠军权向量与提升幅度。

若池化AUC-argmax 权向量明显偏离 (.1,.3,.6) 且池化AUC 涨得可观 → 用已存 v6 模型
（不重训）按新权重重融合产 v7 上板验证组长口径。

用法：venv\\Scripts\\python.exe scripts\\54_pooled_auc_probe.py
产物：outputs/candidate/replay/pooled_auc_report.txt
"""
from __future__ import annotations

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
REPLAY_DIR = PROJECT_ROOT / "outputs" / "candidate" / "replay"


def gauc_macro(df: pd.DataFrame, s: np.ndarray) -> float:
    sub = df[["user_id", "month"]].copy()
    sub["s"] = np.asarray(s, dtype="float64")
    sub["lbl"] = df["label"].to_numpy(dtype="int64")
    grp, _ = pd.factorize(sub["user_id"].astype(str) + "_" + sub["month"].astype(str))
    sub["g"] = grp
    sub["r"] = sub.groupby("g")["s"].rank(method="average").to_numpy()
    pos = sub[sub["lbl"] == 1]
    agg = sub.groupby("g").agg(n=("lbl", "size"), p=("lbl", "sum"))
    sp = pos.groupby("g")["r"].sum()
    agg = agg.join(sp.rename("sp"), how="left").fillna({"sp": 0.0})
    agg["nn"] = agg["n"] - agg["p"]
    ok = agg[(agg["p"] > 0) & (agg["nn"] > 0)].copy()
    ok["auc"] = (ok["sp"] - ok["p"] * (ok["p"] + 1.0) / 2.0) / (ok["p"] * ok["nn"])
    return float(ok["auc"].mean())


def main() -> None:
    sets = [
        ("v6 分量(oof_eqg)", "oof_eqg_preds.csv",
         ["lgb_macro_eq", "xgb_macro_eq", "lr_eq_l2_sp"]),
        ("重调分量(oof_eqtune)", "oof_eqtune_preds.csv",
         ["lgb_lb_base", "xgb_xb_d5", "lr_w2_a1p0_C3p0"]),
    ]
    lines = ["# 池化AUC vs gAUC_m 探针（OOF 187,730 行 Jul..Oct）", ""]
    for tag, fn, comps in sets:
        df = pd.read_csv(REPLAY_DIR / fn, dtype={"item_id": str})
        label = df["label"].to_numpy()
        lines.append(f"===== {tag}  comps={comps} =====")
        for c in comps:
            r = rankdata(df[c]) / len(df)
            lines.append(f"  单 {c:16s}: 池化AUC={roc_auc_score(label, r):.4f}  "
                         f"gAUC_m={gauc_macro(df, r):.4f}")
        rmat = np.column_stack([rankdata(df[c]) / len(df) for c in comps])

        def grid(arg: str):
            step = 0.025
            n = int(round(1.0 / step)) + 1
            rows = []
            for idx in itertools.product(range(n), repeat=2):
                if sum(idx) > n - 1:
                    continue
                w = np.array([idx[0], idx[1], n - 1 - idx[0] - idx[1]],
                             dtype="float64") * step
                s = rmat @ w
                rows.append((tuple(round(float(x), 3) for x in w),
                             roc_auc_score(label, s), gauc_macro(df, s)))
            g = pd.DataFrame(rows, columns=["w", "pooled_auc", "gAUC_m"])
            g["score"] = g[arg]
            return g.sort_values("score", ascending=False).reset_index(drop=True)

        ga = grid("pooled_auc")
        gm = grid("gAUC_m")
        lines.append("  按 池化AUC argmax top5:")
        for _, r in ga.head(5).iterrows():
            lines.append(f"    w={r['w']}  pooledAUC={r['pooled_auc']:.4f}  "
                         f"gAUC_m={r['gAUC_m']:.4f}")
        lines.append("  按 gAUC_m argmax top5:")
        for _, r in gm.head(5).iterrows():
            lines.append(f"    w={r['w']}  pooledAUC={r['pooled_auc']:.4f}  "
                         f"gAUC_m={r['gAUC_m']:.4f}")
        w_A = ga.iloc[0]["w"]
        lines.append(f"  → 池化AUC-argmax 冠军 w={w_A} pooledAUC={ga.iloc[0]['pooled_auc']:.4f} "
                     f"(gAUC_m={ga.iloc[0]['gAUC_m']:.4f})")
        lines.append("")
    txt = "\n".join(lines) + "\n"
    (REPLAY_DIR / "pooled_auc_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
