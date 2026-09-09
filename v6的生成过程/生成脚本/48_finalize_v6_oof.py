"""v6 冠军组合精挑：在 oof_eqg_preds 上对结构化候选组合做 step .025 细网格。

候选组件（全部列已在盘）：
    lgb_macro / lgb_macro_eq / xgb_macro / xgb_macro_eq / lr(=macro_sp last2) / lr_eq_l2_sp
单模型 gAUC_m：lgb_macro .8556｜xgb_macro .8586｜xgb_macro_eq .8665｜lr .8684｜lr_eq_l2_sp .8687。
三分量全组等权 best .8721（> v5 .8710）。此脚本细网格确认稳健冠军并写 weights json。

用法：venv\\Scripts\\python.exe scripts\\48_finalize_v6_oof.py
产物：outputs/candidate/replay/fusion_v6_report.txt + fusion_v6_weights.json
"""
from __future__ import annotations

import itertools
import json
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

REPLAY_DIR = PROJECT_ROOT / "outputs" / "candidate" / "replay"


def gauc_vec(df: pd.DataFrame, s: np.ndarray):
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
    w = ok["n"]
    return float((w * ok["auc"]).sum() / w.sum()), float(ok["auc"].mean())


def grid(base, df, label, comps, tag, out_lines, step=0.025):
    rmat = np.column_stack([rankdata(base[c]) / len(base) for c in comps])
    m = len(comps)
    n = int(round(1.0 / step)) + 1
    rows = []
    for idx in itertools.product(range(n), repeat=m - 1):
        if sum(idx) > n - 1:
            continue
        w = np.array(list(idx) + [n - 1 - sum(idx)], dtype="float64") * step
        s = rmat @ w
        wg, mg = gauc_vec(df, s)
        rows.append((tuple(round(float(x), 3) for x in w),
                     roc_auc_score(label, s), wg, mg))
    res = pd.DataFrame(rows, columns=["w", "auc", "gAUC_w", "gAUC_m"])
    res = res.sort_values("gAUC_m", ascending=False).reset_index(drop=True)
    out_lines.append(f"== {tag}（{comps}）step{step} top4 ==")
    for _, r in res.head(4).iterrows():
        out_lines.append(f"w={r['w']}  AUC={r['auc']:.4f} gAUC_w={r['gAUC_w']:.4f} "
                         f"gAUC_m={r['gAUC_m']:.4f}")
    out_lines.append("")
    return res


def main() -> None:
    base = pd.read_csv(REPLAY_DIR / "oof_eqg_preds.csv", dtype={"item_id": str})
    df = base[["user_id", "item_id", "label", "month"]].copy()
    label = df["label"].to_numpy()
    lines = ["# v6 冠军精挑（step .025；对照 v5=(lgb_macro,xgb_macro,lr)=.8710）", ""]

    combos = [
        (["xgb_macro_eq", "lr_eq_l2_sp"], "二分量 eqXGB+eqLR"),
        (["lgb_macro", "xgb_macro_eq", "lr_eq_l2_sp"], "原LGB+eqXGB+eqLR"),
        (["lgb_macro_eq", "xgb_macro_eq", "lr_eq_l2_sp"], "eq三分量"),
        (["xgb_macro", "xgb_macro_eq", "lr_eq_l2_sp"], "双XGB变体+eqLR"),
        (["lgb_macro", "xgb_macro", "lr_eq_l2_sp"], "原树+eqLR"),
        (["lgb_macro", "xgb_macro_eq", "lr"], "原LR(1/√n)+eqXGB"),
    ]
    results = [grid(base, df, label, c, t, lines) for c, t in combos]
    best_rows = [r.iloc[0] for r in results]
    best_idx = int(np.argmax([r["gAUC_m"] for r in best_rows]))
    br = best_rows[best_idx]
    comps = combos[best_idx][0]
    wdict = {c: float(w) for c, w in zip(comps, br["w"])}
    json.dump({"components": comps, "weights": wdict,
               "auc": float(br["auc"]), "gAUC_w": float(br["gAUC_w"]),
               "gAUC_m": float(br["gAUC_m"])},
              open(REPLAY_DIR / "fusion_v6_weights.json", "w"), indent=2)
    lines.append(f"== 全场 best gAUC_m={br['gAUC_m']:.4f} w={wdict}"
                 f"（> .8710 → 产 v6 提交）==")
    txt = "\n".join(lines) + "\n"
    (REPLAY_DIR / "fusion_v6_report.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print("[save] fusion_v6_report.txt / fusion_v6_weights.json")


if __name__ == "__main__":
    main()
