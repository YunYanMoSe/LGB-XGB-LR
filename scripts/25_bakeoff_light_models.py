"""候选集打分任务 ⑦：轻量模型组烧写（LR / XGBoost / numpy FM / 组合集成）。

背景：老师点名的模型族分成轻量组（LR、XGBoost、FM）与深度组（DeepFM、双塔、DIN）。
本脚本只跑轻量组，在同一份协议 B 数据上（9 月训练 → 10 月从未当训练标签的测试池）
与当前最佳配置 c1+macro_user_wt（= v3 采用的 LGB 1000 树）正面比，并做 rank 集成。

产出只有一张对比表（新文件 outputs/candidate/bakeoff_light_models.txt），
供决定"是否值得据此出 v4 提交"。不写模型、不改任何既有产物。

用法：venv\\Scripts\\python.exe scripts\\25_bakeoff_light_models.py  （约 5-10 分钟）
"""
from __future__ import annotations

# 锁单线程数值库（跨进程可复现）
import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import math
import sys
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from config.settings import SEED, set_seed
from src import candidate as cand

set_seed(SEED)

TR_CUT = pd.Timestamp("2010-09-01")   # 训练：特征<09-01、标签=9 月
TE_CUT = pd.Timestamp("2010-10-01")   # 测试：特征<10-01、标签=10 月（从未当训练标签）
VEC_TR = PROJECT_ROOT / "outputs" / "candidate" / "vectors_20100901_skip.npz"
KS = (5, 10, 20)
OUT_TXT = PROJECT_ROOT / "outputs" / "candidate" / "bakeoff_light_models.txt"

# ---- v3 采用的锚点配置：c1+macro_user_wt（33 特征）本地参考线 ----
ANCHOR_LINE = ("anchor c1+macro_user_wt (33feat, 来自 tune_v3_33feat.txt)", "0.244567", "0.2164",
               "0.4565", "0.9434", "0.8964")

INT_KEYS = ("n_estimators", "num_leaves", "min_child_samples", "max_depth",
            "random_state", "verbosity", "n_jobs")
FLOAT_KEYS = ("learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda")


def build_test_pool(clean: pd.DataFrame, cut: pd.Timestamp, n_neg: int = cand.EVAL_NEG):
    """正=未来窗实购；负=窗内未购（含特征期已购未复购）均匀抽 n_neg。返回 (u,c,label)。"""
    n_orders = clean[clean[cand._TIME] < cut].groupby(cand._USER)[cand._INVOICE].nunique()
    cand_users = sorted(int(u) for u, n in n_orders.items() if int(n) >= cand.MIN_FP_ORDERS)
    lp = clean[clean[cand._TIME] >= cut]
    lp_items: dict[int, set[str]] = {}
    for u, gg in lp.groupby(cand._USER):
        lp_items[int(u)] = set(gg[cand._ITEM].astype(str))
    all_items = np.asarray(sorted(set(clean[cand._ITEM].astype(str).unique())), dtype=object)
    rng = np.random.default_rng(SEED)
    users, codes, labels = [], [], []
    for u in cand_users:
        pos = lp_items.get(u, set())
        if not pos:
            continue
        pool = np.asarray(sorted(set(all_items) - pos), dtype=object)
        k = min(n_neg, len(pool))
        neg = pool[rng.choice(len(pool), size=k, replace=False)] if k else np.asarray([], dtype=object)
        users.append(np.full(len(pos) + len(neg), u, dtype="int64"))
        codes.append(np.concatenate([np.asarray(sorted(pos), dtype=object), neg]))
        labels.append(np.concatenate([np.ones(len(pos), dtype="int64"),
                                      np.zeros(len(neg), dtype="int64")]))
    return (np.concatenate(users), np.concatenate(codes), np.concatenate(labels))


def _ndcg(ranked_codes, gt: set, k: int) -> float:
    dcg = idcg = 0.0
    for i, code in enumerate(ranked_codes[:k]):
        rel = 1.0 if code in gt else 0.0
        dcg += rel / math.log2(i + 2)
    for i in range(min(k, len(gt))):
        idcg += 1.0 / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


def metric_block(ev: pd.DataFrame, ks=KS) -> dict:
    out: dict = {}
    out["pair_auc"] = round(float(roc_auc_score(ev["label"], ev["score"])), 4)
    n_u = 0
    hits = {k: 0.0 for k in ks}; rec = {k: 0.0 for k in ks}
    jac = {k: 0.0 for k in ks}; ndc = {k: 0.0 for k in ks}
    for _u, g in ev.groupby("user_id", sort=False):
        g = g.sort_values("score", ascending=False)
        gt = set(g.loc[g["label"] == 1, "stock_code"])
        if not gt:
            continue
        n_u += 1
        codes = list(g["stock_code"])
        for k in ks:
            top = codes[:k]
            inter = set(top) & gt
            hits[k] += 1.0 if inter else 0.0
            rec[k] += len(inter) / len(gt)
            jac[k] += len(inter) / len(set(top) | gt)
            ndc[k] += _ndcg(codes, gt, k)
    m = max(1, n_u)
    out["n_users_pos"] = int(n_u)
    for k in ks:
        out[f"hit@{k}"] = round(hits[k] / m, 4)
        out[f"rec@{k}"] = round(rec[k] / m, 4)
        out[f"jac@{k}"] = round(jac[k] / m, 4)
        out[f"ndcg@{k}"] = round(ndc[k] / m, 4)
    return out


def _lgb_params(over: dict) -> dict:
    p = {"n_estimators": 300, "learning_rate": 0.1, "num_leaves": 31,
         "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
         "random_state": SEED, "verbosity": -1, "deterministic": True, "n_jobs": 1}
    p.update(over)
    for k in INT_KEYS:
        if k in p:
            p[k] = int(p[k])
    for k in FLOAT_KEYS:
        if k in p:
            p[k] = float(p[k])
    return p


def _sigmoid(z):
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _logloss(p, y, eff):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -float(np.sum(eff * (y * np.log(p) + (1 - y) * np.log(1 - p))))


class NumpyFM:
    """二阶因子分解机：logloss + L2，全量批量 Adam（确定性）。

    输入：Xs = 33 稠密特征（建议已标准化）；uidx/iidx = user/item 词表下标（0=冷 id/OOV，
    嵌入恒为 0、不训练，靠稠密特征兜底）；y 0/1；eff 每样本权重（= macro 加权 × scale_pos）。
    预测：w0 + Xs·w + bu[u] + bi[i] + 0.5( (Σ_j v_j x_j)·(Σ_j v_j x_j) - Σ_j x_j²‖v_j‖² )
    """

    def __init__(self, k=16, lr=0.1, steps=1000, wd=1e-5, seed=SEED, verbose=200):
        self.k = int(k); self.lr = float(lr); self.steps = int(steps)
        self.wd = float(wd); self.seed = int(seed); self.verbose = int(verbose)

    # -- 参数 --  (self.Vd/Eu/Ei/w/bu/bi/b0)
    def _adam(self, p, g, states, name):
        m, v, t = states
        m[name] = self.b1 * m.get(name, 0.0) + (1 - self.b1) * g
        v[name] = self.b2 * v.get(name, 0.0) + (1 - self.b2) * (g * g)
        mh = m[name] / (1 - self.b1 ** t)
        vh = v[name] / (1 - self.b2 ** t)
        p[name] -= self.lr * mh / (np.sqrt(vh) + 1e-8)

    def fit(self, Xs, uidx, iidx, y, eff):
        n, nD = Xs.shape
        Nu = int(uidx.max()); Ni = int(iidx.max())
        k = self.k
        rng = np.random.default_rng(self.seed)
        Vd = rng.normal(0, 0.05, (nD, k)).astype(np.float64)
        Eu = np.zeros((Nu + 1, k), dtype=np.float64)
        Eu[1:] = rng.normal(0, 0.05, (Nu, k))
        Ei = np.zeros((Ni + 1, k), dtype=np.float64)
        Ei[1:] = rng.normal(0, 0.05, (Ni, k))
        w = np.zeros(nD, dtype=np.float64)
        bu = np.zeros(Nu + 1, dtype=np.float64)
        bi = np.zeros(Ni + 1, dtype=np.float64)
        prior = float(np.clip(y.mean(), 0.01, 0.99))
        b0 = float(math.log(prior / (1 - prior)))
        P = {"Vd": Vd, "Eu": Eu, "Ei": Ei, "w": w, "bu": bu, "bi": bi, "b0": b0}
        self.b1, self.b2 = 0.9, 0.999
        m_st = {}; v_st = {}; t = 0
        states = (m_st, v_st, t)
        Xs2 = Xs * Xs
        t0 = time.time()
        for step in range(1, self.steps + 1):
            Vd, Eu, Ei, w, bu, bi = P["Vd"], P["Eu"], P["Ei"], P["w"], P["bu"], P["bi"]
            # 前向
            A = Xs @ Vd + Eu[uidx] + Ei[iidx]                      # (n,k)
            nd = (Vd * Vd).sum(axis=1)                              # (33,)
            Ssq = (Xs2 @ nd) + np.einsum("ij,ij->i", Eu[uidx], Eu[uidx]) \
                + np.einsum("ij,ij->i", Ei[iidx], Ei[iidx])
            logit = (P["b0"] + Xs @ w + bu[uidx] + bi[iidx]
                     + 0.5 * ((A * A).sum(axis=1) - Ssq))
            p = _sigmoid(logit)
            if step == 1 or step % self.verbose == 0 or step == self.steps:
                print(f"          [FM] step {step:5d}/{self.steps}  train_logloss="
                      f"{_logloss(p, y, eff):.5f}  ({time.time()-t0:.0f}s)", flush=True)
            # 梯度（dL/dlogit = eff*(p-y)）
            g = eff * (p - y)
            gA = g[:, None] * A                                     # (n,k)
            grads = {
                "b0": g.sum(),
                "w": Xs.T @ g,
                "bu": np.bincount(uidx, weights=g, minlength=Nu + 1),
                "bi": np.bincount(iidx, weights=g, minlength=Ni + 1),
                "Vd": (Xs.T @ gA) - Vd * (g[:, None] * Xs2).sum(axis=0)[:, None],
            }
            # Eu/Ei：两个来源（Σ_gA 的 A 项 − 自身×Σ_g 的范数项）
            Eu1 = np.zeros((Nu + 1, k))
            Ei1 = np.zeros((Ni + 1, k))
            for f in range(k):
                Eu1[:, f] = np.bincount(uidx, weights=gA[:, f], minlength=Nu + 1)
                Ei1[:, f] = np.bincount(iidx, weights=gA[:, f], minlength=Ni + 1)
            grads["Eu"] = Eu1 - Eu * np.bincount(uidx, weights=g, minlength=Nu + 1)[:, None]
            grads["Ei"] = Ei1 - Ei * np.bincount(iidx, weights=g, minlength=Ni + 1)[:, None]
            # L2（只罚嵌入/线性权，不罚偏置；0 号冷 id 行为 0 梯度自动保住）
            for nm in ("Vd", "Eu", "Ei", "w"):
                grads[nm] = grads[nm] + self.wd * P[nm]
            # Adam 更新
            t += 1
            for nm in P:
                self._adam(P, grads[nm], (m_st, v_st, t), nm)
        self.P = P
        return self

    def predict(self, Xs, uidx, iidx):
        P = self.P; k = self.k
        Vd, Eu, Ei = P["Vd"], P["Eu"], P["Ei"]
        A = Xs @ Vd + Eu[uidx] + Ei[iidx]
        nd = (Vd * Vd).sum(axis=1)
        Ssq = (Xs * Xs) @ nd + np.einsum("ij,ij->i", Eu[uidx], Eu[uidx]) \
            + np.einsum("ij,ij->i", Ei[iidx], Ei[iidx])
        logit = (P["b0"] + Xs @ P["w"] + P["bu"][uidx] + P["bi"][iidx]
                 + 0.5 * ((A * A).sum(axis=1) - Ssq))
        return _sigmoid(logit)


def make_id_maps(train_codes, train_users):
    """词表只从训练行建（排序保证确定性）；0 号留给测试冷 id=OOV。"""
    ucodes = np.unique(train_codes)
    uusers = np.unique(train_users)
    code2i = {str(c): i + 1 for i, c in enumerate(ucodes)}
    user2i = {int(u): i + 1 for i, u in enumerate(uusers)}
    return user2i, code2i


def main() -> None:
    t0 = time.time()
    clean = cand.load_clean_train()
    feats = cand.FEATURES_CAND

    print("[1/4] 构建 9 月训练集（向量走缓存）…", flush=True)
    bundle_tr = cand.build_bundle(clean, cut=TR_CUT, vec_cache=VEC_TR)
    ds = cand.build_pair_dataset(clean, bundle_tr, cut=TR_CUT)
    st = ds.attrs["stats"]
    X = ds[feats].to_numpy(dtype=np.float64)
    y = ds["label"].to_numpy()
    users_tr = ds["user_id"].to_numpy(dtype="int64")
    codes_tr = ds["stock_code"].astype(str).to_numpy()
    print(f"      训练样本：正 {st['pos_n']:,} / 负 {st['neg_n']:,}（有正用户 {st['n_users_with_pos']:,}）",
          flush=True)
    nrow_u = ds.groupby("user_id").size()
    w_macro = (1.0 / np.sqrt(nrow_u.reindex(ds["user_id"]).to_numpy(dtype="float64"))).astype("float64")

    print("[2/4] 构建 10 月测试池特征…", flush=True)
    bundle_te = cand.build_bundle(clean, cut=TE_CUT, vec_cache=cand.VEC_CACHE)
    u, c, l = build_test_pool(clean, TE_CUT)
    X_ev = cand.features_for_pairs(bundle_te, u, c)
    ev_base = pd.DataFrame({"user_id": u, "stock_code": c, "label": l})
    print(f"      测试池 {len(u):,} 行", flush=True)

    # FM 输入：标准化 + id 映射
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float64)
    Xs_ev = scaler.transform(X_ev.to_numpy(dtype=np.float64)).astype(np.float64)
    user2i, code2i = make_id_maps(codes_tr, users_tr)
    uidx_tr = np.asarray([user2i[int(x)] for x in users_tr], dtype="int64")
    iidx_tr = np.asarray([code2i[str(x)] for x in codes_tr], dtype="int64")
    uidx_te = np.asarray([user2i.get(int(x), 0) for x in u], dtype="int64")
    iidx_te = np.asarray([code2i.get(str(x), 0) for x in c], dtype="int64")

    print("[3/4] 训练轻量模型…", flush=True)
    rows: list[tuple] = []
    evals: dict[str, pd.DataFrame] = {}

    def add(name, kind, ev) -> dict:
        b = metric_block(ev)
        evals[name] = ev
        rows.append((name, kind, b))
        print(f"      {name:18s} " + "  ".join(f"rec@{k}={b[f'rec@{k}']:.3f}" for k in KS)
              + f"   ndcg@10={b['ndcg@10']:.3f}   hit@10={b['hit@10']:.3f}   pairAUC={b['pair_auc']:.4f}",
              flush=True)
        return b

    spw = float((y == 0).sum() / max(1.0, y.sum()))

    # A. 锚点：LGB c1+macro（应与 tune_v3_33feat 一致 → 校验管道同源）
    t = time.time()
    clf = LGBMClassifier(**_lgb_params({"n_estimators": 1000, "learning_rate": 0.05,
                                        "num_leaves": 63, "min_child_samples": 30,
                                        "scale_pos_weight": spw}))
    clf.fit(X, y, sample_weight=w_macro)
    ev = ev_base.copy(); ev["score"] = clf.predict_proba(X_ev)[:, 1]
    add("lgb_c1_macro", "lgb", ev)
    print(f"          [LGB] 1000 树完成，{time.time()-t:.0f}s", flush=True)

    # B. XGBoost（macro 加权，形状与 c1 相近）
    t = time.time()
    import xgboost as xgb
    xc = xgb.XGBClassifier(n_estimators=600, learning_rate=0.07, max_depth=6,
                           min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
                           reg_lambda=1.0, reg_alpha=0.0, scale_pos_weight=spw,
                           tree_method="hist", random_state=SEED, n_jobs=1,
                           eval_metric="logloss", verbosity=0)
    xc.fit(X, y, sample_weight=w_macro)
    ev = ev_base.copy(); ev["score"] = xc.predict_proba(X_ev)[:, 1]
    add("xgb_macro", "xgb", ev)
    print(f"          [XGB] 700 树完成，{time.time()-t:.0f}s", flush=True)

    # C. LR（内部已标准化，与 scripts/23 同一基线）
    t = time.time()
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0,
                                                            random_state=SEED))
    lr.fit(X, y)
    ev = ev_base.copy(); ev["score"] = lr.predict_proba(X_ev)[:, 1]
    add("lr_baseline", "lr", ev)
    print(f"          [LR] 完成，{time.time()-t:.0f}s", flush=True)

    # D. numpy FM（macro 加权 × scale_pos）
    t = time.time()
    eff = w_macro * np.where(y == 1, spw, 1.0)
    fm = NumpyFM(k=16, lr=0.1, steps=800, wd=1e-5, seed=SEED).fit(Xs, uidx_tr, iidx_tr, y, eff)
    ev = ev_base.copy(); ev["score"] = fm.predict(Xs_ev, uidx_te, iidx_te)
    add("fm_macro", "fm", ev)
    print(f"          [FM] 完成，{time.time()-t:.0f}s", flush=True)

    print("[4/4] rank 集成与汇总…", flush=True)

    # 基模型原始分数落盘（免重训即可做任意后处理融合/对比）
    base_names = ["lgb_c1_macro", "xgb_macro", "lr_baseline", "fm_macro"]
    SAVE_NPZ = PROJECT_ROOT / "outputs" / "candidate" / "bakeoff_light_scores.npz"
    np.savez(SAVE_NPZ, user_id=u, stock_code=c, label=l,
             **{nm: evals[nm]["score"].to_numpy() for nm in base_names})
    print(f"      基模型分数已存 {SAVE_NPZ}", flush=True)

    def rank_blend(names, out_name, weights=None):
        parts = []
        wt = np.ones(len(names)) if weights is None else np.asarray(weights, dtype="float64")
        wt = wt / wt.sum()
        for nm in names:
            df = evals[nm][["user_id", "stock_code", "label"]].copy()
            df["sc"] = evals[nm].groupby("user_id")["score"].rank(pct=True, method="average")
            parts.append(df.set_index(["user_id", "stock_code", "label"])["sc"] * wt[len(parts)])
        bl = pd.concat(parts, axis=1).sum(axis=1).rename("score").reset_index()
        ev = bl[["user_id", "stock_code", "label", "score"]].copy()
        add(out_name, "ens", ev)

    rank_blend(["lgb_c1_macro", "xgb_macro"], "ens(lgb+xgb)")
    rank_blend(["lgb_c1_macro", "fm_macro"], "ens(lgb+fm)")
    rank_blend(["lgb_c1_macro", "xgb_macro", "fm_macro", "lr_baseline"], "ens(all4)")
    rank_blend(["fm_macro", "lgb_c1_macro"], "fm0.7+lgb0.3", weights=[0.7, 0.3])
    rank_blend(["fm_macro", "lgb_c1_macro"], "fm0.85+lgb0.15", weights=[0.85, 0.15])

    # ---- 汇总表 ----
    df = pd.DataFrame([
        {"config": n, "kind": k,
         **{f"rec@{kk}": b[f"rec@{kk}"] for kk in KS},
         **{f"ndcg@{kk}": b[f"ndcg@{kk}"] for kk in KS},
         **{f"hit@{kk}": b[f"hit@{kk}"] for kk in KS},
         "jac@10": b["jac@10"], "pair_auc": b["pair_auc"], "n_users": b["n_users_pos"]}
        for n, k, b in rows])
    df["avg_rec"] = df[[f"rec@{kk}" for kk in KS]].mean(axis=1)
    df["avg_ndcg"] = df[[f"ndcg@{kk}" for kk in KS]].mean(axis=1)
    df = df.sort_values("avg_rec", ascending=False).reset_index(drop=True)

    cols = ["config", "avg_rec", "avg_ndcg", "rec@5", "rec@10", "rec@20",
            "ndcg@5", "ndcg@10", "ndcg@20", "hit@5", "hit@10", "hit@20",
            "jac@10", "pair_auc", "n_users"]
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write("# 轻量模型组烧写（协议 B：9 月训练→10 月打分）2026-09-08\n")
        fh.write("# 老师点名模型族：LR / XGBoost / FM / DeepFM / 双塔 / DIN；本表只含轻量组。\n")
        fh.write("# 锚点(来自 33 特征调参表 c1+macro_user_wt)  avg_rec=%s rec@10=%s rec@20=%s"
                 " hit@10=%s pairAUC=%s\n\n" % tuple(ANCHOR_LINE[1:]))
        df[cols].to_string(fh, index=False)
    print("\n结果已写 " + str(OUT_TXT))
    print(df[cols].round(4).to_string(index=False))
    print(f"总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
