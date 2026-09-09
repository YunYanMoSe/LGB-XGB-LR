"""候选集打分任务 ⑧：FM 优势的跨窗复验（8 月训练 → 9 月打分）。

scripts/25 在 Sep→Oct 协议 B 上发现 numpy FM 大幅领先 LGB c1+macro（rec@10 0.295 vs 0.216、
ndcg@10 0.696 vs 0.328）。在烧掉一次老师提交之前，先在**整月前移的同构窗口**
（特征<08-01 训练/标签=8 月 → 特征<09-01 打分/标签=9 月）复验：
若 FM 相对 LGB 的巨大优势在此窗依旧成立 ⇒ 不是 9→10 单月的偶然，才考虑据此出 v4。

只比两个候选：lgb_c1_macro（= v3 采用配置）vs fm_macro（= 25 的 NumpyFM 同参数）。
25 的 NumpyFM/metric/build_test_pool 经 importlib 复用，保证与 25 同源。

用法：venv\\Scripts\\python.exe scripts\\26_backtest_aug.py  （约 4-8 分钟，8 月向量缓存首次要建）
产物：outputs/candidate/backtest_aug_results.txt
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import importlib.util
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
from lightgbm import LGBMClassifier

from config.settings import SEED, set_seed
from src import candidate as cand

# 加载 25 为模块（不执行其 main），复用同一份模型/评测实现
_spec = importlib.util.spec_from_file_location(
    "b25", PROJECT_ROOT / "scripts" / "25_bakeoff_light_models.py")
b25 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(b25)

set_seed(SEED)

# ---- 前移一月的同构窗口 ----
TR_CUT = pd.Timestamp("2010-08-01")   # 训练：特征<08-01、标签=8 月
TE_CUT = pd.Timestamp("2010-09-01")   # 测试：特征<09-01、标签=9 月（从未当训练标签）
VEC_TR = PROJECT_ROOT / "outputs" / "candidate" / "vectors_20100801_skip.npz"
VEC_TE = b25.VEC_TR                     # = vectors_20100901_skip.npz（9 月特征窗，25 已建）
KS = (5, 10, 20)
OUT_TXT = PROJECT_ROOT / "outputs" / "candidate" / "backtest_aug_results.txt"

ROW_HEAD = ("config", "avg_rec", "avg_ndcg", "rec@5", "rec@10", "rec@20", "ndcg@10",
            "hit@5", "hit@10", "hit@20", "jac@10", "pair_auc", "n_users")


def main() -> None:
    t0 = time.time()
    clean = cand.load_clean_train()
    feats = cand.FEATURES_CAND

    print("[1/4] 构建 8 月训练集（向量首次可能要建）…", flush=True)
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

    print("[2/4] 构建 9 月测试池特征…", flush=True)
    bundle_te = cand.build_bundle(clean, cut=TE_CUT, vec_cache=VEC_TE)
    u, c, l = b25.build_test_pool(clean, TE_CUT)
    X_ev = cand.features_for_pairs(bundle_te, u, c)
    ev_base = pd.DataFrame({"user_id": u, "stock_code": c, "label": l})
    print(f"      测试池 {len(u):,} 行", flush=True)

    # FM 输入
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float64)
    Xs_ev = scaler.transform(X_ev.to_numpy(dtype=np.float64)).astype(np.float64)
    user2i, code2i = b25.make_id_maps(codes_tr, users_tr)
    uidx_tr = np.asarray([user2i[int(x)] for x in users_tr], dtype="int64")
    iidx_tr = np.asarray([code2i[str(x)] for x in codes_tr], dtype="int64")
    uidx_te = np.asarray([user2i.get(int(x), 0) for x in u], dtype="int64")
    iidx_te = np.asarray([code2i.get(str(x), 0) for x in c], dtype="int64")

    print("[3/4] 训练 LGB 锚点 + FM…", flush=True)
    spw = float((y == 0).sum() / max(1.0, y.sum()))
    rows: list[tuple] = []

    def run_and_log(name, score):
        ev = ev_base.copy(); ev["score"] = score
        b = b25.metric_block(ev)
        rows.append((name, b))
        print(f"      {name:14s} " + "  ".join(f"rec@{k}={b[f'rec@{k}']:.3f}" for k in KS)
              + f"   ndcg@10={b['ndcg@10']:.3f}   hit@10={b['hit@10']:.3f}   "
              f"jac@10={b['jac@10']:.3f}   pairAUC={b['pair_auc']:.4f}", flush=True)

    t = time.time()
    clf = LGBMClassifier(**b25._lgb_params({"n_estimators": 1000, "learning_rate": 0.05,
                                            "num_leaves": 63, "min_child_samples": 30,
                                            "scale_pos_weight": spw}))
    clf.fit(X, y, sample_weight=w_macro)
    run_and_log("lgb_c1_macro", clf.predict_proba(X_ev)[:, 1])
    print(f"          [LGB] 完成，{time.time()-t:.0f}s", flush=True)

    t = time.time()
    eff = w_macro * np.where(y == 1, spw, 1.0)
    fm = b25.NumpyFM(k=16, lr=0.1, steps=800, wd=1e-5, seed=SEED).fit(Xs, uidx_tr, iidx_tr, y, eff)
    run_and_log("fm_macro", fm.predict(Xs_ev, uidx_te, iidx_te))
    print(f"          [FM] 完成，{time.time()-t:.0f}s", flush=True)

    # 汇总（表里直接与 Sep→Oct 参照行并列）
    print("[4/4] 汇总…", flush=True)
    ref = ("# Sep→Oct 参照(25):  lgb rec@10=0.216 ndcg@10=0.328 hit@10=0.943 | "
           "fm rec@10=0.295 ndcg@10=0.696 hit@10=0.961 rec@20=0.418\n")
    lines = ["# 跨窗复验（8 月训练→9 月打分，同构协议 B）2026-09-08\n", ref, "\n"]
    head = "%-14s %8s %8s %6s %6s %6s %8s %6s %6s %6s %6s %6s %6s" % ROW_HEAD
    lines.append(head + "\n")
    for name, b in rows:
        r = dict(avg_rec=np.mean([b[f"rec@{k}"] for k in KS]),
                 avg_ndcg=np.mean([b[f"ndcg@{k}"] for k in KS]), **b)
        lines.append("%-14s %8.6f %8.6f %6.4f %6.4f %6.4f %8.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6d"
                     % (name, r["avg_rec"], r["avg_ndcg"], r["rec@5"], r["rec@10"], r["rec@20"],
                        r["ndcg@10"], r["hit@5"], r["hit@10"], r["hit@20"], r["jac@10"],
                        r["pair_auc"], b["n_users_pos"]))
        lines.append("\n")
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.writelines(lines)
    print("结果已写 " + str(OUT_TXT))
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
