"""训练窗口/时间衰减 OOF 实验（wide_v2，时间序，LGB-macro 与 LR-macro_sp）。

打分点在 11-01（贴 10 月标签），回放训练把更早月份等权，越近可能越相关。测两种：
    * 窗口截断：目标月 t 只用 [t-k .. t-1] 最近 k 个月（k∈{all,3,2}）；
    * 指数衰减：全部前月，行权重额外 × exp(-α·gap)，gap=该训练月离目标月的月序差。
行权重基准：LGB=macro(1/√行数)；LR=macro×spw(等价 macro_sp，class_weight 默认)。
只看 gAUC_w/gAUC_m（已证=真榜序）。

用法：venv\\Scripts\\python.exe scripts\\41_window_oof.py
产物：outputs/candidate/replay/window_oof.txt
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


def load_wide():
    return {m: pd.read_csv(REPLAY_DIR / f"wide_v2_{m[:7]}.csv", dtype={"item_id": str})
            for m in MONTHS}


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


def oof(model_kind: str, window: int | None, alpha: float):
    wide = load_wide()
    mo = sorted(wide)
    parts = []
    for ti, tgt in enumerate(mo):
        if ti == 0:
            continue
        tr_m = mo[:ti][-window:] if window else mo[:ti]
        tr = pd.concat([wide[m] for m in tr_m], ignore_index=True)
        te = wide[tgt]
        nrow_u = tr.groupby("user_id").size()
        macro = 1.0 / np.sqrt(nrow_u.reindex(tr["user_id"]).to_numpy(dtype="float64"))

        # 逐训练月行数 & 指数衰减系数（gap = ti - j）
        sizes = tr.groupby("month", sort=True).size()
        sizes = sizes.reindex(tr_m).to_numpy(dtype="int64")
        if alpha and len(tr_m) > 1:
            decay = np.exp(-alpha * np.arange(len(tr_m) - 1, -1, -1, dtype="float64"))  # 最近月=1
            extra = np.repeat(decay, sizes)
        else:
            extra = np.ones(len(tr), dtype="float64")

        if model_kind == "lgb_macro":
            from lightgbm import LGBMClassifier
            sw = (macro * extra).astype("float64")
            clf = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                                 min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=1.0, random_state=SEED, verbosity=-1,
                                 deterministic=True, n_jobs=1)
            clf.fit(tr[FEATS], tr["label"], sample_weight=sw)
            s = clf.predict_proba(te[FEATS])[:, 1].astype("float64")
        else:  # lr_macro_sp
            spw = float((tr["label"] == 0).sum() / max(1, int(tr["label"].sum())))
            sw = (macro * np.where(tr["label"] == 1, spw, 1.0) * extra).astype("float64")
            lr = make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=2000, C=1.0, random_state=SEED))
            lr.fit(tr[FEATS], tr["label"], logisticregression__sample_weight=sw)
            s = lr.predict_proba(te[FEATS])[:, 1].astype("float64")
        part = te[["user_id", "item_id", "label", "month"]].copy()
        part["score"] = s
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    lines = []
    for kind in ("lgb_macro", "lr"):
        for tag, w, a in [("all/α=0", None, 0.0), ("last3", 3, 0.0), ("last2", 2, 0.0),
                          ("α=0.3", None, 0.3), ("α=0.8", None, 0.8)]:
            pf = oof(kind, w, a)
            wg, mg = gauc_vec(pf, pf["score"].to_numpy())
            lines.append(f"{kind:9s} {tag:8s}  gAUC_w={wg:.4f}  gAUC_m={mg:.4f}")
            print(lines[-1], flush=True)
    (REPLAY_DIR / "window_oof.txt").write_text(
        "# 窗口/衰减 OOF（wide_v2；基准：lgb_macro gAUC_m .8556；lr(macro_sp) .8670）\n"
        + "\n".join(lines) + "\n", encoding="utf-8")
    print("[save] window_oof.txt")


if __name__ == "__main__":
    main()
