"""候选集打分 v9（ALS 第4分量）：v6 三分量 + ALS 隐因子，全局百分位秩融合，cut=11-01。

结构赌注（跳出 44 特征/三分量框架）：ALS 吃「<cut 购买史」里的全局 user×item 隐亲和，
与树/LR 从 44 维表格式特征里榨出的近窗/画像信号去相关。做法守 v4-FM 教训：
* 不单点 —— ALS 永远作第4分量全局秩融合（LR 侧重，板方向）；
* 强正则防 FM 式极端集中/并列崩顶（k=48, α=30, λ=5, 事件日≤5, 18 iter, seed42）；
* 三分量直接用 v6 serve 已存 pkl（44 特征，cut 11-01，models/candidate_cand-align-v6-fuse_*）；
* 自检：以 v6 权重 (.1,.3,.6) 重融合应逐位复现已存 v6 提交 ⇒ 复用模型正确。

两发 ALS 占比（剂量-反应对，榜保留最佳、免费）：
  * v9-als25：lgb .06 / xgb .16 / lr .53 / als .25
  * v9-als35：lgb .05 / xgb .14 / lr .46 / als .35

用法：venv\\Scripts\\python.exe scripts\\65_build_submission_v9_als.py
产物：outputs/candidate/sample_submission_cand-align-v9-als{25,35}.csv
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SEED, set_seed
from src import candidate as cand
from src.candidate_v2 import v2_features, V2_FEATS

set_seed(SEED)

REPLAY = PROJECT_ROOT / "outputs" / "candidate" / "replay"
LEADER_CLEAN = PROJECT_ROOT / "data" / "processed" / "leader_clean.csv"
CAND_INPUT = PROJECT_ROOT / "train_data" / "sample_candidates.csv"
SC_CUT = pd.Timestamp("2010-11-01")
SC_VEC = PROJECT_ROOT / "outputs" / "candidate" / "v9_vecs_20101101_skip.npz"
BASE_FEATS = cand.FEATURES_CAND
FEATS = list(BASE_FEATS) + list(V2_FEATS)
V6 = PROJECT_ROOT / "outputs" / "candidate" / "sample_submission_cand-align-v6-fuse.csv"
MODELS = {k: PROJECT_ROOT / "models" / f"candidate_cand-align-v6-fuse_{k}.pkl"
          for k in ["lgb", "xgb", "lr"]}
W_V6 = {"lgb": 0.1, "xgb": 0.3, "lr": 0.6}

# ALS（与 scripts/63 同配置）
K = 48; ALPHA = 30.0; REG = 5.0; R_CAP = 5; ITERS = 18
USER = "CustomerID"; ITEM = "StockCode"; TIME = "InvoiceDate"


def fit_als_score(clean, cut, cand_users, cand_items):
    """fit ALS on 史<cut 全体 → 打候选对点积；OOV(用户/商品不在史)→0。返回 score(n)。"""
    fp = clean[clean[TIME] < cut]
    fp = fp[[USER, ITEM]].copy()
    fp[USER] = pd.to_numeric(fp[USER], errors="coerce").astype("int64")
    fp[ITEM] = fp[ITEM].astype(str)
    users = np.sort(pd.unique(pd.concat([fp[USER], pd.Series(cand_users)], ignore_index=True)))
    items = np.sort(pd.unique(pd.concat([fp[ITEM], pd.Series(cand_items)], ignore_index=True)))
    u_map = {u: j for j, u in enumerate(users)}
    i_map = {i: j for j, i in enumerate(items)}
    ev = fp.drop_duplicates()
    r = ev.groupby([USER, ITEM]).size().clip(upper=R_CAP).astype("float64")
    ridx = np.fromiter((u_map[u] for u in r.index.get_level_values(0)),
                       dtype=np.int64, count=len(r))
    cidx = np.fromiter((i_map[i] for i in r.index.get_level_values(1).astype(str)),
                       dtype=np.int64, count=len(r))
    conf = 1.0 + ALPHA * r.to_numpy()
    R = sparse.coo_matrix((conf, (ridx, cidx)),
                          shape=(len(users), len(items))).tocsr()
    n_u, n_i = R.shape
    rng = np.random.default_rng(SEED)
    U = (rng.standard_normal((n_u, K)) * 0.01).astype("float64")
    V = (rng.standard_normal((n_i, K)) * 0.01).astype("float64")
    regI = REG * np.eye(K, dtype="float64")
    for _it in range(ITERS):
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
    cu = np.searchsorted(users, cand_users)
    ci = np.searchsorted(items, cand_items)
    ok = (cu < n_u) & (ci < n_i)
    score = np.zeros(len(cand_users), dtype="float64")
    if ok.any():
        good = (users[cu[ok]] == cand_users[ok]) & (items[ci[ok]] == cand_items[ok])
        uu = cu[ok][good]; ii = ci[ok][good]
        if len(uu):
            score[ok] = np.einsum("ij,ij->i", U[uu], V[ii])
    return score


def main() -> None:
    t_all = time.time()
    clean = cand.load_clean_train(LEADER_CLEAN)
    clean[TIME] = pd.to_datetime(clean[TIME])
    clean[USER] = pd.to_numeric(clean[USER], errors="coerce").astype("int64")
    clean[ITEM] = clean[ITEM].astype(str)

    # ---- 候选 44 特征（与 v6 完全同管线）----
    raw = pd.read_csv(CAND_INPUT, dtype={c: str for c in ["user_id", "item_id"]})
    df = raw[["user_id", "item_id"]].copy()
    df["user_id"] = pd.to_numeric(df["user_id"]).astype("int64")
    df["item_id"] = df["item_id"].astype(str)
    bundle = cand.build_bundle(clean, cut=SC_CUT, vec_cache=SC_VEC)
    Xb = cand.features_for_pairs(bundle, df["user_id"].to_numpy(), df["item_id"].to_numpy())
    Xv = v2_features(clean, SC_CUT, df)
    Xs = pd.concat([Xb.reset_index(drop=True), Xv.reset_index(drop=True)], axis=1)
    assert list(Xs.columns) == FEATS and Xs.isna().sum().sum() == 0

    s = {}
    for k in ["lgb", "xgb", "lr"]:
        m = joblib.load(str(MODELS[k]))
        s[k] = m.predict_proba(Xs)[:, 1].astype("float64")
        print(f"[{k}] 复用 v6 serve 模型打分完成", flush=True)

    # 自检：v6 权重应复现已存 v6 提交
    N = len(df)
    rmat3 = np.column_stack([rankdata(s[k]) / N for k in ["lgb", "xgb", "lr"]])
    fus_v6 = rmat3 @ np.array([W_V6["lgb"], W_V6["xgb"], W_V6["lr"]])
    if V6.exists():
        v6 = pd.read_csv(V6)
        assert len(v6) == N
        merged = pd.merge(v6.rename(columns={"score": "v6"}), df, on=["user_id", "item_id"])
        rho = np.corrcoef(merged["v6"].to_numpy(), fus_v6)[0, 1]
        print(f"[自检] 与已存 v6 提交相关 {rho:.6f}（>0.9999 即复用模型正确）", flush=True)

    # ---- ALS serve ----
    t0 = time.time()
    s_als = fit_als_score(clean, SC_CUT, df["user_id"].to_numpy(),
                          df["item_id"].astype(str).to_numpy())
    print(f"[als] ALS serve（cut 11-01, 史全量）拟合+打分完成（{time.time()-t0:.0f}s）；"
          f"非零 {int((s_als != 0).sum()):,}/{N}", flush=True)

    r4 = np.column_stack([rmat3, rankdata(s_als) / N])
    configs = {
        "cand-align-v9-als25": (0.06, 0.16, 0.53, 0.25),
        "cand-align-v9-als35": (0.05, 0.14, 0.46, 0.35),
    }
    for name, w in configs.items():
        fused = r4 @ np.array(w)
        out = pd.DataFrame({"user_id": df["user_id"], "item_id": df["item_id"],
                            "score": fused})
        assert len(out) == N and out.isna().sum().sum() == 0
        p = PROJECT_ROOT / "outputs" / "candidate" / f"sample_submission_{name}.csv"
        out.to_csv(p, index=False, encoding="utf-8")
        meta = {"name": name, "desc": "v6 三分量 + ALS 隐因子第4分量全局秩融合 "
                "(k48/α30/λ5/ev≤5/18it seed42, 史<cut)",
                "weights": dict(zip(["lgb", "xgb", "lr", "als"], w)),
                "score_cut": str(SC_CUT.date()), "feature_order": FEATS,
                "als_train": "history < 2010-11-01 (leader_clean 全量)"}
        (PROJECT_ROOT / "models" / f"candidate_meta_{name}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{name}] → {p.name}（w={w}，总耗时 {time.time()-t_all:.0f}s）", flush=True)
    print(f"[done] v9 两发完成，总耗时 {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
