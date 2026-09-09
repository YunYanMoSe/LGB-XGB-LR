"""H≥2 视界赌注 —— 本地健康检查（不排名门禁，真伪交给真榜）。

假设：老师测试窗可能 >1 个月 ⇒ v6 的下月模型（"t 月买"标签）高估急切复购、低估稳健兴趣。
赌注：同一三分量重训在 H2 标签（m 月 ∪ (m+1) 月买）上 → serve。本脚本只验模型健康：
干净 no-overlap 折叠（训练窗不与目标窗口共享月份）：
    fold A：训 Jun..Jul 的 H2 行（Jun∪Jul, Jul∪Aug）→ 打 Sep 行分，GT=Sep∪Oct
    fold B：训 Jun 的 H2 行（Jun∪Jul）→ 打 Aug 行分，GT=Aug∪Sep
健康门禁：H2 窗口 gAUC_m 明显 >.8 且不崩（正例变多指标天然偏高，不可与 v6 的 1 月口径比）。

用法：venv\\Scripts\\python.exe scripts\\67_oof_h2.py
产物：replay/h2_oof_report.txt
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

REPLAY = PROJECT_ROOT / "outputs" / "candidate" / "replay"
LEADER_CLEAN = PROJECT_ROOT / "data" / "processed" / "leader_clean.csv"
USER = "CustomerID"; ITEM = "StockCode"; TIME = "InvoiceDate"
H2_MONTHS = [f"2010-{m:02d}-01" for m in range(6, 10)]  # Jun..Sep 可贴完整 H2


def monthly_purchase_sets(clean):
    """{month_str('2010-08'): frozenset[(u,i)str]} 当月真实购买对。"""
    df = clean[["CustomerID", "StockCode", "InvoiceDate"]].copy()
    df["m"] = df[TIME].dt.strftime("%Y-%m")
    return {m: set(zip(g["CustomerID"].to_numpy(), g["StockCode"].astype(str).to_numpy()))
            for m, g in df.groupby("m")}


def load_h2(month, buy_sets):
    w = pd.read_csv(REPLAY / f"wide_v2_{month[:7]}.csv", dtype={"item_id": str})
    nxt = f"2010-{int(month[5:7]) + 1:02d}"
    s = buy_sets.get(nxt, set())
    u = w["user_id"].to_numpy(dtype="int64")
    i = w["item_id"].astype(str).to_numpy()
    bought_next = np.fromiter(((a, b) in s for a, b in zip(u, i)),
                              dtype=bool, count=len(w))
    w = w.copy()
    w["label"] = (w["label"].to_numpy(dtype="int64") | bought_next.astype("int64"))
    return w


def eq_group_sw(tr, with_sp):
    n_g = tr.groupby(["user_id", "month"], sort=True).size()
    keys = pd.MultiIndex.from_arrays([tr["user_id"], tr["month"].astype(str)])
    w = 1.0 / n_g.reindex(keys).to_numpy(dtype="float64")
    if not with_sp:
        return w.astype("float64")
    spw = float((tr["label"] == 0).sum() / max(1, int(tr["label"].sum())))
    return (w * np.where(tr["label"] == 1, spw, 1.0)).astype("float64")


def gen(tr_rows, tgt, feats, kind):
    sw = eq_group_sw(tr_rows, with_sp=(kind == "lr_eq_l2_sp"))
    t0 = time.time()
    if kind == "lgb_macro_eq":
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                             min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=1.0, random_state=SEED, verbosity=-1,
                             deterministic=True, n_jobs=1)
        clf.fit(tr_rows[feats], tr_rows["label"], sample_weight=sw)
        s = clf.predict_proba(tgt[feats])[:, 1].astype("float64")
    elif kind == "xgb_macro_eq":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8,
                            tree_method="hist", n_jobs=1, random_state=SEED,
                            eval_metric="auc", verbosity=0)
        clf.fit(tr_rows[feats], tr_rows["label"], sample_weight=sw)
        s = clf.predict_proba(tgt[feats])[:, 1].astype("float64")
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        lr = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, C=1.0, random_state=SEED))
        lr.fit(tr_rows[feats], tr_rows["label"], logisticregression__sample_weight=sw)
        s = lr.predict_proba(tgt[feats])[:, 1].astype("float64")
    print(f"[{kind}] 训 {len(tr_rows):,} 行 → 折 {tgt['month'].iloc[0][:7]}"
          f"（{time.time()-t0:.0f}s）", flush=True)
    return s


def metrics(df, s):
    sub = df[["user_id", "month"]].copy()
    sub["s"] = np.asarray(s, dtype="float64")
    sub["lbl"] = df["label"].to_numpy(dtype="int64")
    g, _ = pd.factorize(sub["user_id"].astype(str) + "_" + sub["month"].astype(str))
    sub["g"] = g
    lbl = sub["lbl"].to_numpy()
    n_g = np.bincount(g); p_g = np.bincount(g, weights=lbl.astype("float64"))
    sub["ra"] = sub.groupby("g")["s"].rank(method="average")
    sp = np.bincount(g, weights=(sub["ra"].to_numpy() * lbl.astype("float64")))
    nn = n_g - p_g
    with np.errstate(divide="ignore", invalid="ignore"):
        auc = np.where((p_g > 0) & (nn > 0),
                       (sp - p_g * (p_g + 1) / 2) / (p_g * nn), np.nan)
    ok = ~np.isnan(auc)
    return float(auc[ok].mean())


def main() -> None:
    t_all = time.time()
    clean = cand.load_clean_train(LEADER_CLEAN)
    clean[TIME] = pd.to_datetime(clean[TIME])
    clean[USER] = pd.to_numeric(clean[USER], errors="coerce").astype("int64")
    clean[ITEM] = clean[ITEM].astype(str)
    buys = monthly_purchase_sets(clean)
    wide = {m: load_h2(m, buys) for m in H2_MONTHS}
    from src.candidate_v2 import V2_FEATS
    feats = list(cand.FEATURES_CAND) + list(V2_FEATS)

    lines = [f"# H2 健康检查（Jun..Sep 行贴 m∪(m+1) 标签；训窗不叠目标窗）", ""]
    folds = [
        ("A: 训 Jun..Jul → 打 Sep(GT=Sep∪Oct)", ["2010-06-01", "2010-07-01"],
         "2010-09-01", ["2010-08-01", "2010-09-01"]),  # LR 近2月 = Aug..Sep? 下面处理
        ("B: 训 Jun → 打 Aug(GT=Aug∪Sep)", ["2010-06-01"],
         "2010-08-01", ["2010-06-01"]),
    ]
    # LR 近 2 月由调用的训练月子集控制（H2 里近 2 个标注月）
    results = []
    for name, tr_ms, te_m, _ in folds:
        tr_rows = pd.concat([wide[m] for m in tr_ms], ignore_index=True)
        te = wide[te_m]
        cols = {}
        for k in ["lgb_macro_eq", "xgb_macro_eq"]:
            cols[k] = gen(tr_rows, te, feats, k)
        # LR 近 2 个标注月（不足则全量）
        lr_ms = tr_ms[-2:] if len(tr_ms) > 2 else tr_ms
        cols["lr_eq_l2_sp"] = gen(pd.concat([wide[m] for m in lr_ms], ignore_index=True),
                                  te, feats, "lr_eq_l2_sp")
        r = rankdata(cols["lgb_macro_eq"]) / len(te) * 0.1 \
            + rankdata(cols["xgb_macro_eq"]) / len(te) * 0.3 \
            + rankdata(cols["lr_eq_l2_sp"]) / len(te) * 0.6
        g = metrics(te, r)
        results.append((name, g))
        lines.append(f"{name}：H2 融合 gAUC_m={g:.4f}  "
                     f"(目标正例率 {te['label'].mean():.3f})")
    lines.append("")
    lines.append("== 健康门禁：两折均明显 >.8 且不崩 → 可 serve；仍差 → 记失败日志 ==")
    txt = "\n".join(lines) + "\n"
    (REPLAY / "h2_oof_report.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"[done] {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
