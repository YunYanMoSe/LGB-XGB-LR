"""ALS v2 构造筛查 —— 只为让第4分量摆脱退化（不是挑代理冠军）。

v1（scripts/63）教训：全历史等权 ALS = 终身亲和，无时机信号 → 单 gAUC_m .5755，
融合 25% 占比就把 v6 从 .8721 砸到 .8464 ⇒ 太弱，带不动自身权重。
v2 改两处（都对齐既有洞见，非代理过拟合）：
  * 近因衰减 conf：事件权重 = exp(-days_ago/τ)，τ=60 —— collab 版 LR「近2月」；
  * 打分 + γ·log1p(商品近期衰减流行度) —— 补回 ALS 缺的 item bias，救 top-10 召回。
退化门禁：单 ALS gAUC_m 明显抬升（~.7+）且 25% 融合不把 gAUC 砸破 .868 才 serve；
仍弱 ⇒ 记入失败日志不烧板（弱分量 5-10% 权重对榜是噪声，无信息量）。

用法：venv\\Scripts\\python.exe scripts\\64_als_v2_oof.py
产物：replay/als_v2_oof_report.txt
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
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SEED
from src import candidate as cand

REPLAY = PROJECT_ROOT / "outputs" / "candidate" / "replay"
LEADER_CLEAN = PROJECT_ROOT / "data" / "processed" / "leader_clean.csv"
TGT = [f"2010-{m:02d}-01" for m in range(7, 11)]

K = 32; ALPHA = 20.0; REG = 5.0; ITERS = 18
TAU = 60.0
GAMMA = 1.0
REF = pd.Timestamp("2009-11-30")
USER = "CustomerID"; ITEM = "StockCode"; TIME = "InvoiceDate"


def fit_v2(clean, cut, tau=TAU, gamma=GAMMA):
    """decay conf ALS；返回 U,V,item_pop(衰减流行度), users, items(排序vocab)。"""
    fp = clean[clean[TIME] < cut]
    fp = fp[[USER, ITEM, TIME]].copy()
    fp[USER] = pd.to_numeric(fp[USER], errors="coerce").astype("int64")
    fp[ITEM] = fp[ITEM].astype(str)
    fp["day"] = (fp[TIME] - REF).dt.days.astype("int64")
    cut_day = int((cut - REF).days)
    fp = fp.drop_duplicates([USER, ITEM, "day"])
    fp["w"] = np.exp(-(cut_day - fp["day"].to_numpy()) / tau).astype("float64")
    g = fp.groupby([USER, ITEM], sort=False)["w"].sum()
    users = np.sort(pd.unique(fp[USER].to_numpy()))
    items = np.sort(pd.unique(fp[ITEM].to_numpy()))
    u_map = {u: j for j, u in enumerate(users)}
    i_map = {i: j for j, i in enumerate(items)}
    rows = g.index.get_level_values(0).to_numpy()
    cols = g.index.get_level_values(1).astype(str).to_numpy()
    ridx = np.fromiter((u_map[u] for u in rows), dtype=np.int64, count=len(g))
    cidx = np.fromiter((i_map[i] for i in cols), dtype=np.int64, count=len(g))
    conf = (1.0 + ALPHA * g.to_numpy()).astype("float64")
    R = sparse.coo_matrix((conf, (ridx, cidx)),
                          shape=(len(users), len(items))).tocsr()
    item_pop = np.zeros(len(items))
    np.add.at(item_pop, cidx, g.to_numpy())
    n_u, n_i = R.shape
    rng = np.random.default_rng(SEED)
    U = (rng.standard_normal((n_u, K)) * 0.01).astype("float64")
    V = (rng.standard_normal((n_i, K)) * 0.01).astype("float64")
    regI = REG * np.eye(K, dtype="float64")
    for _ in range(ITERS):
        rhs_u = R @ V
        for a in range(n_u):
            s0, e0 = R.indptr[a], R.indptr[a + 1]
            if s0 == e0:
                U[a] = 0.0; continue
            vsel = V[R.indices[s0:e0]]; csel = R.data[s0:e0, None]
            U[a] = np.linalg.solve((vsel * csel).T @ vsel + regI, rhs_u[a])
        Rt = R.T.tocsr()
        rhs_i = Rt @ U
        for b in range(n_i):
            s0, e0 = Rt.indptr[b], Rt.indptr[b + 1]
            if s0 == e0:
                V[b] = 0.0; continue
            usel = U[Rt.indices[s0:e0]]; csel = Rt.data[s0:e0, None]
            V[b] = np.linalg.solve((usel * csel).T @ usel + regI, rhs_i[b])
    return U, V, item_pop, users, items


def score(U, V, item_pop, users, items, users_all, items_all, gamma=GAMMA):
    cu = np.searchsorted(users, users_all)
    ci = np.searchsorted(items, items_all)
    n = len(users_all)
    s = np.zeros(n, dtype="float64")
    ok = (cu < len(users)) & (ci < len(items))
    if ok.any():
        good = (users[cu[ok]] == users_all[ok]) & (items[ci[ok]] == items_all[ok])
        uu = cu[ok][good]; ii = ci[ok][good]
        if len(uu):
            s[ok] = np.einsum("ij,ij->i", U[uu], V[ii]) + gamma * np.log1p(item_pop[ii])
    return s


def metrics(base, s):
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
    okm = ~np.isnan(ndcg)
    comp = 0.4 * gauc + 0.4 * float(ndcg[okm].mean()) + 0.2 * float(rec[okm].mean())
    return gauc, comp


def main() -> None:
    t_all = time.time()
    clean = cand.load_clean_train(LEADER_CLEAN)
    clean[TIME] = pd.to_datetime(clean[TIME])
    clean[USER] = pd.to_numeric(clean[USER], errors="coerce").astype("int64")
    clean[ITEM] = clean[ITEM].astype(str)
    oof = pd.read_csv(REPLAY / "oof_eqg_preds.csv", dtype={"item_id": str})
    base = oof[["user_id", "item_id", "label", "month"]]

    parts = []
    for cut_s in TGT:
        cut = pd.Timestamp(cut_s)
        w = pd.read_csv(REPLAY / f"wide_v2_{cut_s[:7]}.csv", dtype={"item_id": str})
        users_all = w["user_id"].to_numpy(dtype="int64")
        items_all = w["item_id"].astype(str).to_numpy(dtype=object)
        U, V, item_pop, users, items = fit_v2(clean, cut)
        s = score(U, V, item_pop, users, items, users_all, items_all)
        p = w[["user_id", "item_id", "label", "month"]].copy()
        p["als2"] = s
        parts.append(p)
        print(f"[{cut_s[:7]}] v2 ALS 完成（{time.time()-t_all:.0f}s）", flush=True)
    pf = pd.concat(parts, ignore_index=True)
    assert (pf["user_id"].astype(str).to_numpy() == base["user_id"].astype(str).to_numpy()).all() \
        and (pf["item_id"].astype(str).to_numpy() == base["item_id"].astype(str).to_numpy()).all()
    s = pf["als2"].to_numpy()
    gauc, comp = metrics(base, s)
    lines = [f"# ALS v2（decay τ={TAU}, α={ALPHA}, k={K}, λ={REG}, itembias γ={GAMMA}）",
             f"单 ALS v2：gAUC_m={gauc:.4f}  comp_m={comp:.5f}", ""]
    rmat3 = np.column_stack([rankdata(oof[c]) / len(base) for c in
                             ["lgb_macro_eq", "xgb_macro_eq", "lr_eq_l2_sp"]])
    r4 = np.column_stack([rmat3, rankdata(s) / len(base)])
    for wgt in [(0.05, 0.15, 0.50, 0.30), (0.05, 0.14, 0.46, 0.35)]:
        gg, cc = metrics(base, r4 @ np.array(wgt))
        lines.append(f"4路融合 w={wgt}：gAUC_m={gg:.4f} comp_m={cc:.5f}  （v6 基线 .8721/.58429）")
    txt = "\n".join(lines) + "\n"
    (REPLAY / "als_v2_oof_report.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"[done] {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
