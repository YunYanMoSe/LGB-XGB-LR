"""按老师公布的精确评分函数重建本地复合分，锁定聚合口径并重搜融合权。

综合分 = 0.40×GAUC + 0.40×NDCG@10 + 0.20×Recall@10（组长 2026-09-09 告知）。
本脚本在 Jul..Oct OOF（同一 187,730 行 / 8,293 正例）上：

  A. 用已存 OOF 分量逐位重建四个「已上板」融合的复合分，测 8 种聚合口径
     (GAUC/NDCG/Recall × 等权macro / 曝光加权gAUC_w) 哪一种复现真榜序：
        v6=.7782 > v8-ext=.7776 > v7-pooled=.7774 ≈ v7-tune=.7774
     ⇒ 锁定老师聚合口径。
  B. 单分量复合分：看 NDCG@10/Recall@10 是否偏爱与 gAUC_m 不同的分量（tree vs LR）。
  C. 在锁定口径下对三分量做融合权网格重搜 → composite-argmax 权 vs v6 (.1,.3,.6)。

组 = (user,month)。per-group 指标对分数单调变换不变 ⇒ 融合用全局百分位秩（与提交同管线）。

用法：venv\\Scripts\\python.exe scripts\\59_composite_probe.py
产物：outputs/candidate/replay/composite_report.txt
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools
import json
import sys
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPLAY = PROJECT_ROOT / "outputs" / "candidate" / "replay"
OUT = REPLAY / "composite_report.txt"


def group_meta(df: pd.DataFrame):
    """每组 (user,month) 的 g 编码、n、p；返回 (g, n_by_g, p_by_g)。"""
    g, _ = pd.factorize(df["user_id"].astype(str) + "_" + df["month"].astype(str))
    lbl = df["label"].to_numpy(dtype="int64")
    n_g = np.bincount(g)
    p_g = np.bincount(g, weights=lbl.astype("float64"))
    return g, n_g, p_g, lbl


def group_metrics(s: np.ndarray, g: np.ndarray, n_g: np.ndarray, p_g: np.ndarray,
                  lbl: np.ndarray):
    """向量化逐组指标。返回按组下标索引的 auc/ndcg10/recall10（无定义处 = nan）。"""
    sub = pd.DataFrame({"g": g, "s": np.asarray(s, dtype="float64"), "lbl": lbl})
    # 组内升序平均秩 → AUC
    sub["ra"] = sub.groupby("g")["s"].rank(method="average")
    sp = np.bincount(g, weights=(sub["ra"].to_numpy() * lbl.astype("float64")))
    nn = n_g - p_g
    with np.errstate(divide="ignore", invalid="ignore"):
        auc_g = np.where((p_g > 0) & (nn > 0),
                         (sp - p_g * (p_g + 1.0) / 2.0) / (p_g * nn), np.nan)
    # 组内降序 first 秩 → top10
    sub["rd"] = sub.groupby("g")["s"].rank(method="first", ascending=False)
    top = (sub["rd"].to_numpy() <= 10.0)
    hit = (top & (lbl == 1))
    hits_g = np.bincount(g, weights=hit.astype("float64"))
    dcg_g = np.bincount(g, weights=np.where(
        hit, 1.0 / np.log2(sub["rd"].to_numpy() + 1.0), 0.0))
    # IDCG@10 = sum_{i=1..min(p,10)} 1/log2(i+1)（p 取到 10 封顶，只 11 个取值，循环无害）
    denom = np.log2(np.arange(1, 11, dtype="float64") + 1.0)
    idcg = np.array([denom[:int(min(v, 10))].sum() for v in p_g])
    with np.errstate(divide="ignore", invalid="ignore"):
        ndcg_g = np.where(p_g > 0, dcg_g / np.where(idcg > 0, idcg, np.nan), np.nan)
        rec_g = np.where(p_g > 0, hits_g / p_g, np.nan)
    return auc_g, ndcg_g, rec_g


def agg(metric_g: np.ndarray, n_g: np.ndarray, *, weighted: bool):
    """metric_g 按组，nan=无定义。weighted=按组曝光量 n 加权；否则等权 macro。"""
    ok = ~np.isnan(metric_g)
    if not ok.any():
        return float("nan")
    if weighted:
        return float((metric_g[ok] * n_g[ok]).sum() / n_g[ok].sum())
    return float(metric_g[ok].mean())


def composite_for(s: np.ndarray, g, n_g, p_g, lbl, *, gauc_w, ndcg_w, rec_w):
    a, d, r, *_ = group_metrics(s, g, n_g, p_g, lbl)
    GAUC = agg(a, n_g, weighted=gauc_w)
    NDCG = agg(d, n_g, weighted=ndcg_w)
    REC = agg(r, n_g, weighted=rec_w)
    return 0.4 * GAUC + 0.4 * NDCG + 0.2 * REC, GAUC, NDCG, REC


def load(fn: str):
    return pd.read_csv(REPLAY / fn, dtype={"item_id": str})


def main() -> None:
    t0 = time.time()
    eqg = load("oof_eqg_preds.csv")     # v6 分量（Jun..Oct 池）
    eqt = load("oof_eqtune_preds.csv")  # 调参分量（xb_d5/lr_C3）
    ext = load("oof_ext_preds.csv")     # v8-ext 分量（Feb..Oct 池）
    base = eqg[["user_id", "item_id", "label", "month"]]
    g, n_g, p_g, lbl = group_meta(base)
    assert len(eqt) == len(ext) == len(base)

    # ---- A. 四个已上板融合 = 分量全局百分位秩 × 公布权 ----
    def pct(a: np.ndarray) -> np.ndarray:
        from scipy.stats import rankdata
        return rankdata(a) / len(a)

    # 各文件分量 → 全局百分位秩矩阵（与提交管线一致；组内单调，per-group 指标不变）
    def comps(df, names):
        return np.column_stack([pct(df[n].to_numpy(dtype="float64")) for n in names])

    anchors = {
        # name: (component-matrix, weights)  →  fused percentile rank
        "v6-fuse  (.1,.3,.6) eqg": (
            comps(eqg, ["lgb_macro_eq", "xgb_macro_eq", "lr_eq_l2_sp"]),
            np.array([0.1, 0.3, 0.6])),
        "v7-pooled(.075,.55,.375) eqg": (
            comps(eqg, ["lgb_macro_eq", "xgb_macro_eq", "lr_eq_l2_sp"]),
            np.array([0.075, 0.55, 0.375])),
        "v7-tune  (.4,.6) eqt": (
            comps(eqt, ["xgb_xb_d5", "lr_w2_a1p0_C3p0"]),
            np.array([0.4, 0.6])),
        "v8-ext   (.15,.225,.625) ext": (
            comps(ext, ["lgb_macro_eq", "xgb_macro_eq", "lr_eq_l2_sp"]),
            np.array([0.15, 0.225, 0.625])),
    }
    fused = {name: (cm @ w) for name, (cm, w) in anchors.items()}

    varis = [(False, False, False), (False, False, True), (False, True, False),
             (False, True, True), (True, False, False), (True, False, True),
             (True, True, False), (True, True, True)]
    vnames = {vv: f"GAUC{'w' if vv[0] else 'm'}|NDCG{'w' if vv[1] else 'm'}"
                     f"|Rec{'w' if vv[2] else 'm'}" for vv in varis}

    lines = ["# 复合分探针（综合分 = 0.4 GAUC + 0.4 NDCG@10 + 0.2 Recall@10）",
             "# 真榜序：v6 .7782 > v8-ext .7776 > v7-pooled .7774 ≈ v7-tune .7774", ""]
    # 8 口径 × 4 anchor 复合分
    scores = {}
    for vv in varis:
        row = []
        for name, s in fused.items():
            c, ga, nd, rc = composite_for(s, g, n_g, p_g, lbl, gauc_w=vv[0],
                                           ndcg_w=vv[1], rec_w=vv[2])
            scores[(name, vv)] = (c, ga, nd, rc)
            row.append((name.split()[0], c))
        order = sorted(row, key=lambda x: -x[1])
        rep = "OK " if order[0][0] == "v6-fuse" else "xx "
        lines.append(f"[{rep}] {vnames[vv]:24s} " +
                     "  ".join(f"{nm}={v:.5f}" for nm, v in row))
    lines.append("")
    # 拆分看三子分：选一个基准口径（全等权 macro）展开
    lines.append("== 子分拆解（全等权 macro 口径）：composite/GAUC/NDCG10/Recall10 ==")
    for name in fused:
        c, ga, nd, rc = scores[(name, (False, False, False))]
        lines.append(f"{name:26s} {c:.5f} | {ga:.5f} {nd:.5f} {rc:.5f}")
    lines.append("")

    # ---- B. 单分量复合（全等权 macro + 全曝光加权两口径展示）----
    lines.append("== 单分量复合（等权macro 与 曝光加权两口径）==")
    singles = {}
    for df, nm in [(eqg, "eqg Jun池"), (eqt, "eqt Jun池"), (ext, "ext Feb池")]:
        for c in df.columns:
            if c in ("user_id", "item_id", "label", "month") or df[c].dtype == object:
                continue
            s = df[c].to_numpy(dtype="float64")
            ca, ga, nd, rc = composite_for(s, g, n_g, p_g, lbl, gauc_w=False,
                                           ndcg_w=False, rec_w=False)
            cb, gb, nb, rb = composite_for(s, g, n_g, p_g, lbl, gauc_w=True,
                                           ndcg_w=True, rec_w=True)
            singles[(nm, c)] = (ca, ga, nd, rc, cb)
            lines.append(f"{nm:10s} {c:22s} comp_m={ca:.5f} (GAUC {ga:.4f} "
                         f"NDCG {nd:.4f} Rec {rc:.4f}) | comp_w={cb:.5f}")
    lines.append("")

    # ---- C. 融合权网格重搜（锁定口径待选，先做两个最可能口径）----
    lines.append("== 融合权网格重搜（step .05 起；对 eqg Jun..Oct 三分量）==")
    grid_res = {}
    for vv in [(False, False, False), (True, True, True)]:
        cm = comps(eqg, ["lgb_macro_eq", "xgb_macro_eq", "lr_eq_l2_sp"])
        step = 0.05
        n_ = int(round(1.0 / step)) + 1
        rows = []
        for a, b in itertools.product(range(n_), repeat=2):
            if a + b > n_ - 1:
                continue
            w = np.array([a, b, n_ - 1 - a - b], dtype="float64") * step
            s = cm @ w
            c, *_ = composite_for(s, g, n_g, p_g, lbl, gauc_w=vv[0],
                                  ndcg_w=vv[1], rec_w=vv[2])
            rows.append((tuple(round(float(x), 3) for x in w), c))
        rows.sort(key=lambda x: -x[1])
        grid_res[vv] = rows
        lines.append(f"-- 口径 {vnames[vv]}：top5 --")
        for w, c in rows[:5]:
            lines.append(f"   w={w}  comp={c:.5f}")
        # v6 基准在网格里的名次
        for i, (w, c) in enumerate(rows):
            if abs(w[0] - 0.1) < 1e-9 and abs(w[1] - 0.3) < 1e-9:
                lines.append(f"   [v6 (0.1,0.3,0.6) 在第 {i+1} 名 comp={c:.5f}]")
                break
    lines.append("")

    # 存一个压缩 json 便于后续脚本读
    json.dump({"anchors": {nm: list(map(float, scores[(nm, (False, False, False))]))
                           for nm in fused},
               "grid_top_macro": grid_res[(False, False, False)][:10],
               "grid_top_w": grid_res[(True, True, True)][:10]},
              open(REPLAY / "composite_weights.json", "w"), indent=2)

    txt = "\n".join(lines) + "\n"
    OUT.write_text(txt, encoding="utf-8")
    print(txt)
    print(f"[done] 总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
