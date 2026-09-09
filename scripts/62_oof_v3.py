"""v3 新特征筛查 OOF：wide_v3(44+5) vs wide_v2(44) 同折同配方。

训练池 Jun..Oct、验证 Jul..Oct（与 v6 OOF 逐位相同 187,730 行 / 8,293 正例），三分量
lgb_macro_eq / xgb_macro_eq / lr_eq_l2_sp 配方不变，唯一变化 = 特征 44 → 49。
判据：
    * 融合 gAUC_m 超 v6 的 .8721（且 >.0003 余量）——旧口径，防伪正；
    * 融合本地复合分超 v6 的 .58429（且 >~.0015 余量）——新口径，直接对齐老师公式；
    任一超余量 → 值得深挖（单特征消融 + 上板）；两空 → 特征方向此批证伪。

用法：venv\\Scripts\\python.exe scripts\\62_oof_v3.py
产物：outputs/candidate/replay/v3_report.txt
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
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SEED
from src import candidate as cand
from src.candidate_v2 import V2_FEATS
from src.candidate_v3 import V3_FEATS

REPLAY = PROJECT_ROOT / "outputs" / "candidate" / "replay"
BASE_FEATS = cand.FEATURES_CAND
FEATS44 = list(BASE_FEATS) + list(V2_FEATS)
FEATS49 = FEATS44 + list(V3_FEATS)
MONTHS = [f"2010-{m:02d}-01" for m in range(6, 11)]
TGT = MONTHS[1:]  # Jul..Oct
W = np.array([0.1, 0.3, 0.6])  # v6 融合权


def load_wide(prefix: str, feats: list[str]):
    wide = {}
    for m in MONTHS:
        df = pd.read_csv(REPLAY / f"{prefix}_{m[:7]}.csv", dtype={"item_id": str})
        assert set(feats) <= set(df.columns), (prefix, m, set(feats) - set(df.columns))
        wide[m] = df
    return wide


def eq_group_sw(tr, with_sp):
    n_g = tr.groupby(["user_id", "month"], sort=True).size()
    keys = pd.MultiIndex.from_arrays([tr["user_id"], tr["month"].astype(str)])
    w = 1.0 / n_g.reindex(keys).to_numpy(dtype="float64")
    if not with_sp:
        return w.astype("float64")
    spw = float((tr["label"] == 0).sum() / max(1, int(tr["label"].sum())))
    return (w * np.where(tr["label"] == 1, spw, 1.0)).astype("float64")


def gen(wide, kind: str, feats):
    parts = []
    for ti, tgt in enumerate(TGT):
        tr_m = MONTHS[:ti + 1]
        if kind == "lr_eq_l2_sp" and len(tr_m) > 2:
            tr_m = tr_m[-2:]          # v6 配方：LR 每折只吃最近 2 个历史月
        tr = pd.concat([wide[m] for m in tr_m], ignore_index=True)
        te = wide[tgt]
        sw = eq_group_sw(tr, with_sp=(kind == "lr_eq_l2_sp"))
        t0 = time.time()
        if kind == "lgb_macro_eq":
            from lightgbm import LGBMClassifier
            clf = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                                 min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=1.0, random_state=SEED, verbosity=-1,
                                 deterministic=True, n_jobs=1)
            clf.fit(tr[feats], tr["label"], sample_weight=sw)
            s = clf.predict_proba(te[feats])[:, 1].astype("float64")
        elif kind == "xgb_macro_eq":
            from xgboost import XGBClassifier
            clf = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6,
                                subsample=0.8, colsample_bytree=0.8,
                                tree_method="hist", n_jobs=1, random_state=SEED,
                                eval_metric="auc", verbosity=0)
            clf.fit(tr[feats], tr["label"], sample_weight=sw)
            s = clf.predict_proba(te[feats])[:, 1].astype("float64")
        else:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            lr = make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=2000, C=1.0, random_state=SEED))
            lr.fit(tr[feats], tr["label"], logisticregression__sample_weight=sw)
            s = lr.predict_proba(te[feats])[:, 1].astype("float64")
        part = te[["user_id", "item_id", "label", "month"]].copy()
        part["score"] = s
        parts.append(part)
        print(f"[{kind}] 折 {tgt[:7]} 训 {len(tr):,} 行/{len(tr_m)}月（{time.time()-t0:.0f}s）",
              flush=True)
    return pd.concat(parts, ignore_index=True)


def eval_block(base, cols):
    g, _ = pd.factorize(base["user_id"].astype(str) + "_" + base["month"].astype(str))
    lbl = base["label"].to_numpy(dtype="int64")
    n_g = np.bincount(g); p_g = np.bincount(g, weights=lbl.astype("float64"))
    rmat = np.column_stack([rankdata(cols[k]) / len(base) for k in cols])
    res = {}
    for j, k in enumerate(cols):
        res[k] = {"gAUC_m": _gauc(g, lbl, n_g, p_g, rmat[:, j])}
    fus = rmat @ W
    res["fusion(.1,.3,.6)"] = {"gAUC_m": _gauc(g, lbl, n_g, p_g, fus)}
    res["fusion(.1,.3,.6)"]["comp_m"] = _comp(g, lbl, n_g, p_g, fus)
    for j, k in enumerate(cols):
        res[k]["comp_m"] = _comp(g, lbl, n_g, p_g, rmat[:, j])
    return res


def _gauc(g, lbl, n_g, p_g, s):
    sub = pd.DataFrame({"g": g, "s": s, "lbl": lbl})
    sub["ra"] = sub.groupby("g")["s"].rank(method="average")
    sp = np.bincount(g, weights=(sub["ra"].to_numpy() * lbl.astype("float64")))
    nn = n_g - p_g
    with np.errstate(divide="ignore", invalid="ignore"):
        auc = np.where((p_g > 0) & (nn > 0),
                       (sp - p_g * (p_g + 1) / 2) / (p_g * nn), np.nan)
    ok = ~np.isnan(auc)
    return float(auc[ok].mean())


def _comp(g, lbl, n_g, p_g, s):
    sub = pd.DataFrame({"g": g, "s": s, "lbl": lbl})
    sub["rd"] = sub.groupby("g")["s"].rank(method="first", ascending=False)
    top = sub["rd"].to_numpy() <= 10
    hit = top & (lbl == 1)
    hits = np.bincount(g, weights=hit.astype("float64"))
    dcg = np.bincount(g, weights=np.where(
        hit, 1.0 / np.log2(sub["rd"].to_numpy() + 1.0), 0.0))
    denom = np.log2(np.arange(1, 11, dtype="float64") + 1.0)
    idcg = np.array([denom[:int(min(v, 10))].sum() for v in p_g])
    with np.errstate(divide="ignore", invalid="ignore"):
        ndcg = np.where(p_g > 0, dcg / np.where(idcg > 0, idcg, np.nan), np.nan)
        rec = np.where(p_g > 0, hits / p_g, np.nan)
    ok = ~np.isnan(ndcg)
    return 0.4 * float(_gauc(g, lbl, n_g, p_g, s)) + 0.4 * float(ndcg[ok].mean()) \
        + 0.2 * float(rec[ok].mean())


def main() -> None:
    t0 = time.time()
    wide44 = load_wide("wide_v2", FEATS44)
    wide49 = load_wide("wide_v3", FEATS49)
    base = pd.concat([wide44[m][["user_id", "item_id", "label", "month"]]
                      for m in TGT], ignore_index=True)
    g, _ = pd.factorize(base["user_id"].astype(str) + "_" + base["month"].astype(str))
    lbl = base["label"].to_numpy(dtype="int64")
    n_g = np.bincount(g); p_g = np.bincount(g, weights=lbl.astype("float64"))
    print(f"[init] 验证 {len(base):,} 行 / 正例 {int(lbl.sum()):,}", flush=True)

    lines = ["# v3 新特征筛查（44 vs 49 特征，同折同配方；基准 v6 fusion gAUC_m .8721 / "
             "comp_m .58429）", ""]
    rows = []
    for pre, feats, tag in [("wide_v2", FEATS44, "44(v6基线)"), ("wide_v3", FEATS49, "49(+5v3)")]:
        wide = wide44 if pre == "wide_v2" else wide49
        cols = {}
        for k in ["lgb_macro_eq", "xgb_macro_eq", "lr_eq_l2_sp"]:
            pf = gen(wide, k, feats)
            cols[k] = pf["score"].to_numpy(dtype="float64")
        ev = eval_block(base, cols)
        for name, m in ev.items():
            lines.append(f"[{tag}] {name:16s} gAUC_m={m['gAUC_m']:.4f} "
                         f"comp_m={m['comp_m']:.5f}")
        lines.append("")
        fus_gauc = ev["fusion(.1,.3,.6)"]["gAUC_m"]
        fus_comp = ev["fusion(.1,.3,.6)"]["comp_m"]
        rows.append((tag, fus_gauc, fus_comp))
    d_g = rows[1][1] - rows[0][1]
    d_c = rows[1][2] - rows[0][2]
    lines.append(f"== 49 vs 44：融合 gAUC_m {d_g:+.4f}（基线 .8721）、"
                 f"comp_m {d_c:+.5f}（基线 .58429）==")
    if d_g > 0.0003 or d_c > 0.0015:
        lines.append("== 判据：超余量 → 值得单特征消融 + 上板 ==")
    else:
        lines.append("== 判据：两口径均空 → 该批特征证伪，收在 v6 ==")
    txt = "\n".join(lines) + "\n"
    (REPLAY / "v3_report.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"[done] 总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
