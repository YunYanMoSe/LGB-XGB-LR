"""LR 逐用户 macro 行加权 OOF 实验（wide_v2，时间序）。

LGB 的 macro(1/√行数) 相对 scale_pos 抬 gAUC_m (.8556 vs .8450)；LR 目前只 class_weight=
balanced（全局校准、每行等权）。若 LR+macro/macro_sp 行加权能再抬 gAUC_m（现 balanced
.8659），则 LR 组件可换加权直接进融合。

用法：venv\\Scripts\\python.exe scripts\\40_lr_macro_oof.py
产物：打印对照 + outputs/candidate/replay/lr_macro_oof.txt
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
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

REPLAY_DIR = PROJECT_ROOT / "outputs" / "candidate" / "replay"
FEATS = cand.FEATURES_CAND + V2_FEATS
MONTHS = [f"2010-{m:02d}-01" for m in range(6, 11)]


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


def lr_oof(weight: str):
    from sklearn.linear_model import LogisticRegression as _LR
    wide = {m: pd.read_csv(REPLAY_DIR / f"wide_v2_{m[:7]}.csv", dtype={"item_id": str})
            for m in MONTHS}
    parts = []
    for ti, tgt in enumerate(sorted(wide)):
        tr_m = sorted(wide)[:ti]
        if not tr_m:
            continue
        tr = pd.concat([wide[m] for m in tr_m], ignore_index=True)
        te = wide[tgt]
        nrow_u = tr.groupby("user_id").size()
        macro = 1.0 / np.sqrt(nrow_u.reindex(tr["user_id"]).to_numpy(dtype="float64"))
        if weight == "macro":
            sw = macro.astype("float64")
        elif weight == "macro_sp":
            spw = float((tr["label"] == 0).sum() / max(1, int(tr["label"].sum())))
            sw = (macro * np.where(tr["label"] == 1, spw, 1.0)).astype("float64")
        else:  # balanced-equivalent（class_weight balanced 在 LR 内部，此分支不用）
            sw = None
        lr = make_pipeline(StandardScaler(), _LR(
            max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED))
        kw = {"logisticregression__sample_weight": sw} if weight != "balanced" else {}
        lr.fit(tr[FEATS], tr["label"], **kw)
        part = te[["user_id", "item_id", "label", "month"]].copy()
        part["score"] = lr.predict_proba(te[FEATS])[:, 1].astype(np.float64)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    lines = []
    for weight in ("balanced", "macro", "macro_sp"):
        pf = lr_oof(weight)
        wg, mg = gauc_vec(pf, pf["score"].to_numpy())
        auc = roc_auc_score(pf["label"], pf["score"])
        ln = f"LR[{weight}] AUC={auc:.4f} gAUC_w={wg:.4f} gAUC_m={mg:.4f}"
        lines.append(ln)
        print(ln, flush=True)
    (REPLAY_DIR / "lr_macro_oof.txt").write_text(
        "# 对照：LR[balanced] v2 = AUC .8812 gAUC_w .8521 gAUC_m .8659\n" + "\n".join(lines) + "\n",
        encoding="utf-8")
    print("[save] lr_macro_oof.txt")


if __name__ == "__main__":
    main()
