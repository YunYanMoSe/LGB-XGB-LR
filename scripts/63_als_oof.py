"""ALS 隐因子第4分量 —— OOF（Jul..Oct）+ 诊断。跳出 44 特征/三分量框架的结构赌注。

归纳偏置 = 纯 user×item 隐亲和：在「< cut 的购买史」上做 implicit-feedback ALS
(confidence c=1+α·r, r=去重购买日数封顶)，候选对 (u,i) 打分 = <u,i> 点积。
树/LR 吃的是候选行 44 维表格式特征（含近窗脉冲），ALS 吃的是用户/商品在全体购买
矩阵里的全局亲和结构 —— 两路信息基本去相关。

用法与本脚本定位：
* 本脚本只做 3 件事：逐月(Jul..Oct)用 H<t 史拟合 ALS → 打候选行分；对齐 oof_eqg_preds
  校验；出诊断（gAUC_m/comp_m/AUC + 与三分量的 Spearman 相关 + 逐月 gAUC 稳定性）。
* 门禁 = 防退化（相关不≈1、gAUC_m 不比 LR 差太远、无大规模并列崩顶），**不是**排名门禁
  ——跨归纳偏置的代理排名不可信（v4-FM 教训），真伪交给免费真榜。
* 单配置不调参：k=48, α=30, λ=5, 事件日数封顶 5, 迭代 18, init seed42, 单线程 BLAS。

用法：venv\\Scripts\\python.exe scripts\\63_als_oof.py
产物：outputs/candidate/replay/oof_als_preds.csv + als_oof_report.txt
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
from scipy import sparse
from scipy.stats import rankdata, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SEED
from src import candidate as cand

REPLAY = PROJECT_ROOT / "outputs" / "candidate" / "replay"
LEADER_CLEAN = PROJECT_ROOT / "data" / "processed" / "leader_clean.csv"
MONTHS = [f"2010-{m:02d}-01" for m in range(6, 11)]
TGT = MONTHS[1:]  # Jul..Oct

# ALS 单配置
K = 48
ALPHA = 30.0
REG = 5.0
R_CAP = 5
ITERS = 18
USER = "CustomerID"
ITEM = "StockCode"
TIME = "InvoiceDate"


def fit_als(clean, cut, user_vocab, item_vocab):
    """implicit ALS on 购买史<cut；返回 {u:行向量}, {i:列向量}(np.ndarray k)。"""
    fp = clean[clean[TIME] < cut]
    ev = fp[[USER, ITEM]].copy()
    ev[USER] = pd.to_numeric(ev[USER], errors="coerce").astype("int64")
    ev[ITEM] = ev[ITEM].astype(str)
    ev = ev.drop_duplicates()                      # (u,i,行) 去重 → 一用户一商品一行=1 次
    r = ev.groupby([USER, ITEM]).size().clip(upper=R_CAP)   # 去重购买行数，封顶
    r = r.astype("float64")
    ui = pd.MultiIndex.from_arrays([r.index.get_level_values(0),
                                    r.index.get_level_values(1).astype(str)])
    row = ui.get_level_values(0).to_numpy()
    col = ui.get_level_values(1).to_numpy()
    # 排序 vocab → 确定性索引
    u_map = {u: j for j, u in enumerate(user_vocab)}
    i_map = {i: j for j, i in enumerate(item_vocab)}
    ridx = np.fromiter((u_map[u] for u in row), dtype=np.int64, count=len(row))
    cidx = np.fromiter((i_map[i] for i in col), dtype=np.int64, count=len(col))
    conf = 1.0 + ALPHA * r.to_numpy()
    R = sparse.coo_matrix((conf, (ridx, cidx)),
                          shape=(len(user_vocab), len(item_vocab))).tocsr()
    n_u, n_i = R.shape
    rng = np.random.default_rng(SEED)
    U = (rng.standard_normal((n_u, K)) * 0.01).astype("float64")
    V = (rng.standard_normal((n_i, K)) * 0.01).astype("float64")
    regI = REG * np.eye(K, dtype="float64")
    for _it in range(ITERS):
        # ---- 更新 U（按行用户；RHS = R@V 可整体算，A 逐行）----
        rhs_u = R @ V                                   # n_u × k（Σ c_ui v_i）
        for a in range(n_u):
            s, e = R.indptr[a], R.indptr[a + 1]
            if s == e:
                U[a] = 0.0
                continue
            vsel = V[R.indices[s:e]]
            csel = R.data[s:e, None]
            A = (vsel * csel).T @ vsel + regI
            U[a] = np.linalg.solve(A, rhs_u[a])
        # ---- 更新 V（按列商品；转置视角：R.T 行=商品）----
        Rt = R.T.tocsr()
        rhs_i = Rt @ U                                   # n_i × k
        for b in range(n_i):
            s, e = Rt.indptr[b], Rt.indptr[b + 1]
            if s == e:
                V[b] = 0.0
                continue
            usel = U[Rt.indices[s:e]]
            csel = Rt.data[s:e, None]
            A = (usel * csel).T @ usel + regI
            V[b] = np.linalg.solve(A, rhs_i[b])
    return U, V


def score_pairs(U, V, user_vocab, item_vocab, users, items):
    """users/items(np 数组) → 每行点积；vocab 外(OOV)→0。"""
    u_pos = np.searchsorted(user_vocab, users)         # vocab 有序 → searchsorted
    i_pos = np.searchsorted(item_vocab, items)
    n = len(users)
    s = np.zeros(n, dtype="float64")
    ok = (u_pos < len(user_vocab)) & (i_pos < len(item_vocab))
    if ok.any():
        uu = u_pos[ok]; ii = i_pos[ok]
        hit = (user_vocab[uu] == users[ok]) & (item_vocab[ii] == items[ok])
        uu = uu[hit]; ii = ii[hit]
        if len(uu):
            s[ok] = np.einsum("ij,ij->i", U[uu], V[ii])
    return s


def gauc_comp(base, s):
    sub = base[["user_id", "month"]].copy()
    sub["s"] = np.asarray(s, dtype="float64")
    sub["lbl"] = base["label"].to_numpy(dtype="int64")
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
    gauc = float(auc[ok].mean())
    # composite macro
    sub["rd"] = sub.groupby("g")["s"].rank(method="first", ascending=False)
    top = sub["rd"].to_numpy() <= 10
    hit = top & (lbl == 1)
    hits = np.bincount(g, weights=hit.astype("float64"))
    dcg = np.bincount(g, weights=np.where(hit, 1.0 / np.log2(sub["rd"].to_numpy() + 1.0), 0.0))
    denom = np.log2(np.arange(1, 11, dtype="float64") + 1.0)
    idcg = np.array([denom[:int(min(v, 10))].sum() for v in p_g])
    with np.errstate(divide="ignore", invalid="ignore"):
        ndcg = np.where(p_g > 0, dcg / np.where(idcg > 0, idcg, np.nan), np.nan)
        rec = np.where(p_g > 0, hits / p_g, np.nan)
    ndcg = ndcg[~np.isnan(ndcg)]; rec = rec[~np.isnan(rec)]
    comp = 0.4 * gauc + 0.4 * float(ndcg.mean()) + 0.2 * float(rec.mean())
    return gauc, comp


def main() -> None:
    t_all = time.time()
    clean = cand.load_clean_train(LEADER_CLEAN)   # 列名即 InvoiceDate/CustomerID/StockCode…
    clean[TIME] = pd.to_datetime(clean[TIME])
    clean[USER] = pd.to_numeric(clean[USER], errors="coerce").astype("int64")
    clean[ITEM] = clean[ITEM].astype(str)

    oof = pd.read_csv(REPLAY / "oof_eqg_preds.csv", dtype={"item_id": str})
    parts = []
    for cut_s in TGT:
        cut = pd.Timestamp(cut_s)
        w = pd.read_csv(REPLAY / f"wide_v2_{cut_s[:7]}.csv", dtype={"item_id": str})
        users = w["user_id"].to_numpy(dtype="int64")
        items = w["item_id"].astype(str).to_numpy(dtype=object)
        fp = clean[clean[TIME] < cut]
        user_vocab = np.sort(pd.unique(fp[USER].astype("int64")))
        item_vocab = np.sort(pd.unique(fp[ITEM].astype(str)))
        t0 = time.time()
        U, V = fit_als(clean, cut, user_vocab, item_vocab)
        s = score_pairs(U, V, user_vocab, item_vocab, users, items)
        part = w[["user_id", "item_id", "label", "month"]].copy()
        part["als_score"] = s
        parts.append(part)
        print(f"[{cut_s[:7]}] ALS 拟合+打分完成（{time.time()-t0:.0f}s, "
              f"正例 {int(w['label'].sum()):,}）", flush=True)
    pf = pd.concat(parts, ignore_index=True)
    base = oof[["user_id", "item_id", "label", "month"]]
    assert (pf["user_id"].astype(str) == base["user_id"].astype(str)).all() and \
           (pf["item_id"].astype(str) == base["item_id"].astype(str)).all(), "未对齐 oof_eqg"
    oof2 = base.copy()
    oof2["als_macro_eq"] = pf["als_score"].to_numpy()
    oof2.to_csv(REPLAY / "oof_als_preds.csv", index=False, encoding="utf-8")

    s = oof2["als_macro_eq"].to_numpy()
    gauc, comp = gauc_comp(base, s)
    lines = [f"# ALS 隐因子 OOF（k={K}, α={ALPHA}, λ={REG}, 事件日≤{R_CAP}, "
             f"{ITERS} iter；Jul..Oct {len(base):,} 行 / 正例 {int(base['label'].sum()):,}）",
             "", f"单 ALS：gAUC_m={gauc:.4f}  comp_m={comp:.5f}"]
    for c in ["lgb_macro_eq", "xgb_macro_eq", "lr_eq_l2_sp"]:
        rho = spearmanr(oof2["als_macro_eq"], oof[c]).statistic
        lines.append(f"  Spearman(ALS, {c}) = {rho:.3f}")
    lines.append("")
    # 逐月稳定性
    per = []
    for m in TGT:
        msk = base["month"].astype(str) == m[:7]
        gg, cc = gauc_comp(base[msk], s[msk])
        per.append(f"{m[:7]}: gAUC_m={gg:.4f} comp_m={cc:.5f}")
    lines.append("逐月：  " + " | ".join(per))
    # 4 路融合（全局百分位秩）信息性读数
    rmat = np.column_stack([rankdata(oof[c]) / len(base) for c in
                            ["lgb_macro_eq", "xgb_macro_eq", "lr_eq_l2_sp"]]
                           + [rankdata(s) / len(base)])
    for wgt in [(0.05, 0.15, 0.50, 0.30), (0.05, 0.14, 0.46, 0.35), (0.05, 0.10, 0.35, 0.50)]:
        fus = rmat @ np.array(wgt)
        gg, cc = gauc_comp(base, fus)
        lines.append(f"4路融合 w={wgt}：gAUC_m={gg:.4f} comp_m={cc:.5f}")
    lines.append("")
    lines.append("== 仅退化检查：corr 不≈1、gAUC 不太低、无崩顶 → 决定 serve ==")
    txt = "\n".join(lines) + "\n"
    (REPLAY / "als_oof_report.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"[done] 总耗时 {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
