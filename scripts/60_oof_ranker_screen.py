"""rubric 对齐模型筛查：组内 NDCG 导向的 listwise ranker，在 v6 同款 OOF 上比复合分。

老师公式 60% 权重压在 NDCG@10 + Recall@10（每用户 top-10 列表）。我们整个回放链训的都是
pointwise logloss 模型；本脚本训 LGBMRanker（lambdarank，组=(user,month)，直接优化组内排序）
并测它的本地复合分能否超 v6 融合(.58429 macro) 一个真余量（>~.002，远超四 anchor 的 .0007
混淆区）。附带免费算高 LR 权融合（LR∈{.7,.8,.9,1.0}）的本地复合，看 LR 轴是否为纯板盲。

验证行 = wide_v2 Jul..Oct（与 v6 OOF 逐位相同，187,730 行 / 8,293 正例）。
产：outputs/candidate/replay/ranker_report.txt
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SEED
from src import candidate as cand
from src.candidate_v2 import V2_FEATS

REPLAY = PROJECT_ROOT / "outputs" / "candidate" / "replay"
FEATS = cand.FEATURES_CAND + V2_FEATS
MONTHS = [f"2010-{m:02d}-01" for m in range(6, 11)]
TGT = MONTHS[1:]  # Jul..Oct


def load_wide():
    return {m: pd.read_csv(REPLAY / f"wide_v2_{m[:7]}.csv", dtype={"item_id": str})
            for m in MONTHS}


def group_meta(df):
    g, _ = pd.factorize(df["user_id"].astype(str) + "_" + df["month"].astype(str))
    lbl = df["label"].to_numpy(dtype="int64")
    return g, np.bincount(g), np.bincount(g, weights=lbl.astype("float64")), lbl


def group_metrics(s, g, n_g, p_g, lbl):
    sub = pd.DataFrame({"g": g, "s": np.asarray(s, dtype="float64"), "lbl": lbl})
    sub["ra"] = sub.groupby("g")["s"].rank(method="average")
    sp = np.bincount(g, weights=(sub["ra"].to_numpy() * lbl.astype("float64")))
    nn = n_g - p_g
    with np.errstate(divide="ignore", invalid="ignore"):
        auc_g = np.where((p_g > 0) & (nn > 0),
                         (sp - p_g * (p_g + 1.0) / 2.0) / (p_g * nn), np.nan)
    sub["rd"] = sub.groupby("g")["s"].rank(method="first", ascending=False)
    top = sub["rd"].to_numpy() <= 10.0
    hit = top & (lbl == 1)
    hits_g = np.bincount(g, weights=hit.astype("float64"))
    dcg_g = np.bincount(g, weights=np.where(
        hit, 1.0 / np.log2(sub["rd"].to_numpy() + 1.0), 0.0))
    denom = np.log2(np.arange(1, 11, dtype="float64") + 1.0)
    idcg = np.array([denom[:int(min(v, 10))].sum() for v in p_g])
    with np.errstate(divide="ignore", invalid="ignore"):
        ndcg_g = np.where(p_g > 0, dcg_g / np.where(idcg > 0, idcg, np.nan), np.nan)
        rec_g = np.where(p_g > 0, hits_g / p_g, np.nan)
    return auc_g, ndcg_g, rec_g


def agg(v, n_g, weighted=False):
    ok = ~np.isnan(v)
    if not ok.any():
        return float("nan")
    return float((v[ok] * n_g[ok]).sum() / n_g[ok].sum()) if weighted else float(v[ok].mean())


def composite(s, g, n_g, p_g, lbl, macro=True):
    a, d, r = group_metrics(s, g, n_g, p_g, lbl)
    kw = dict(weighted=not macro)
    return (0.4 * agg(a, n_g, **kw) + 0.4 * agg(d, n_g, **kw)
            + 0.2 * agg(r, n_g, **kw))


def main() -> None:
    t0 = time.time()
    wide = load_wide()
    parts = [wide[m][["user_id", "item_id", "label", "month"]] for m in TGT]
    base = pd.concat(parts, ignore_index=True)
    g, n_g, p_g, lbl = group_meta(base)
    print(f"[init] 验证 Jul..Oct {len(base):,} 行 / 正例 {int(lbl.sum()):,}；"
          f"组数 {len(n_g):,}", flush=True)

    # ---------- LGBMRanker OOF（lambdarank，组=user-month） ----------
    pf_list = []
    for ti, tgt in enumerate(TGT):
        tr_m = MONTHS[:ti + 1]  # 与 v6 eq OOF 同：目标 Jul 训 Jun、…、Oct 训 Jun..Sep
        tr = pd.concat([wide[m] for m in tr_m], ignore_index=True)
        te = wide[tgt]
        # 组 = (user,month)，必须按组连续排序
        tr = tr.sort_values(["user_id", "month"], kind="mergesort").reset_index(drop=True)
        grp_idx, grp_sizes = pd.factorize(
            tr["user_id"].astype(str) + "_" + tr["month"].astype(str))
        group_sizes = np.bincount(grp_idx).astype("int32")

        from lightgbm import LGBMRanker
        m = LGBMRanker(objective="lambdarank", n_estimators=1000, learning_rate=0.05,
                       num_leaves=31, min_child_samples=20, subsample=0.8,
                       colsample_bytree=0.8, random_state=SEED, verbosity=-1,
                       deterministic=True, n_jobs=1)
        st = time.time()
        m.fit(tr[FEATS], tr["label"], group=group_sizes)
        te = te.sort_values(["user_id", "month"], kind="mergesort").reset_index(drop=True)
        s = m.predict(te[FEATS]).astype("float64")
        # 还原行序到 te 原序（与 base 对齐）
        te["_s"] = s
        te = te.sort_index()
        pf = te[["user_id", "item_id", "label", "month"]].copy()
        pf["score"] = te["_s"].to_numpy()
        pf_list.append(pf)
        print(f"[ranker] 折 {tgt[:7]} 训练 {len(tr):,} 行 / {len(tr_m)} 月"
              f"（{time.time()-st:.0f}s）", flush=True)
    oof = pd.concat(pf_list, ignore_index=True)
    oof = oof.sort_values(["user_id", "month", "item_id"], kind="mergesort").reset_index(drop=True)
    base = base.sort_values(["user_id", "month", "item_id"], kind="mergesort").reset_index(drop=True)
    assert (oof["user_id"].to_numpy() == base["user_id"].to_numpy()).all()
    assert (oof["item_id"].astype(str).to_numpy() == base["item_id"].astype(str).to_numpy()).all()
    g2, n_g2, p_g2, lbl2 = group_meta(base)
    s_rk = oof["score"].to_numpy(dtype="float64")

    lines = ["# rubrric-aligned 筛查（v6 融合 macro comp = .58429 为基准）", ""]
    cm = composite(s_rk, g2, n_g2, p_g2, lbl2)
    lines.append(f"LGBMRanker(lambdarank, 组=user-month) Jun..Oct:")
    lines.append(f"   comp_m={cm:.5f}  (v6 融合 .58429，差 {cm-.58429:+.5f})")
    lines.append("")

    # ---------- 免费：高 LR 权融合（用已存 oof_eqg 分量） ----------
    eqg = pd.read_csv(REPLAY / "oof_eqg_preds.csv", dtype={"item_id": str})
    eqg = eqg.sort_values(["user_id", "month", "item_id"], kind="mergesort").reset_index(drop=True)
    g3, n_g3, p_g3, lbl3 = group_meta(eqg)
    from scipy.stats import rankdata
    cols = {k: rankdata(eqg[k].to_numpy(dtype="float64")) / len(eqg)
            for k in ["lgb_macro_eq", "xgb_macro_eq", "lr_eq_l2_sp"]}
    lines.append("高 LR 权融合（oof_eqg 分量全局秩；LR 轴 = 纯板盲判断）:")
    for (a, b, c) in [(0.1, 0.3, 0.6), (0.05, 0.2, 0.75), (0.0, 0.2, 0.8),
                      (0.0, 0.1, 0.9), (0.0, 0.0, 1.0)]:
        s = a * cols["lgb_macro_eq"] + b * cols["xgb_macro_eq"] + c * cols["lr_eq_l2_sp"]
        lines.append(f"   w=({a},{b},{c})  comp_m={composite(s, g3, n_g3, p_g3, lbl3):.5f}")

    txt = "\n".join(lines) + "\n"
    (REPLAY / "ranker_report.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"[done] 总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
