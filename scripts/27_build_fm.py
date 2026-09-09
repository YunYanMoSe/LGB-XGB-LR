"""候选集打分任务 ⑨：正式 v4 提交 = numpy FM（在烧写 25 + 跨窗复验 26 后上线）。

背景：老师点名的模型族烧写发现 LightGBM 只吃到 33 维画像特征，学不到
"用户在标签月买过 → 未来月复购"的 pair 亲和（训练正样本在 fp 里 owned=0，树看不到）。
FM（user×item 低秩 + 稠密特征二阶交互）直接建模这对亲和，在 Sep→Oct 与 Aug→Sep 两个
同构窗口上全面碾压 LGB c1+macro（rec@10 ~2.5×、ndcg@10 ~2.4×、hit@10 +0.02~0.18），
且全部集成（含 FM 主导加权）都输给 FM 单点 ⇒ v4 就用 FM 单点。

本脚本复刻 scripts/24 的正式生产协议（可复现、不覆盖任何旧产物）：
    * 训练切窗 2010-10-01（历史<10-01、标签=10 月整月），全部用户（有正样本者）；
    * macro 用户加权 1/sqrt(行数) × scale_pos（= 烧写验证过的 FM 配方）；
    * 打分切窗 2010-11-01（用尽整份 train.csv），冷 id→0 号 OOV 嵌入只靠稠密特征；
    * 权重与 id 映射落盘 models/candidate_cand-fm-v1.npz，meta 另存 json。

用法：venv\\Scripts\\python.exe scripts\\27_build_fm.py  （约 3 分钟）
产物：outputs/candidate/sample_submission_cand-fm-v1.csv （提交文件，58,205 行）
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import importlib.util
import json
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
from sklearn.preprocessing import StandardScaler

from config.settings import SEED, set_seed
from src import candidate as cand

# 复用 25 里验证过的 NumpyFM / make_id_maps（main 有 __main__ 守卫，import 不执行）
_spec = importlib.util.spec_from_file_location(
    "b25", PROJECT_ROOT / "scripts" / "25_bakeoff_light_models.py")
b25 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(b25)

set_seed(SEED)

TR_CUT = cand.CUT_DEFAULT                      # 2010-10-01：特征<10-01、标签=10 月
SC_CUT = pd.Timestamp("2010-11-01")            # 打分切窗：用尽整份 train.csv
VEC_SC = PROJECT_ROOT / "outputs" / "candidate" / "vectors_20101101_skip.npz"
CAND_INPUT = PROJECT_ROOT / "train_data" / "sample_candidates.csv"
NAME = "cand-fm-v1"
FM_K, FM_LR, FM_STEPS, FM_WD = 16, 0.1, 800, 1e-5


def main() -> None:
    t0 = time.time()
    feats = cand.FEATURES_CAND
    clean = cand.load_clean_train()

    # ---- 1. 训练数据（cut=10-01，全量）----
    print("[1/3] 构建训练集（cut=10-01，全部用户）…", flush=True)
    bundle_tr = cand.build_bundle(clean, cut=TR_CUT, vec_cache=cand.VEC_CACHE)
    ds = cand.build_pair_dataset(clean, bundle_tr, cut=TR_CUT)
    st = ds.attrs["stats"]
    X = ds[feats].to_numpy(dtype=np.float64)
    y = ds["label"].to_numpy()
    users_tr = ds["user_id"].to_numpy(dtype="int64")
    codes_tr = ds["stock_code"].astype(str).to_numpy()
    spw = float((y == 0).sum() / max(1, int(y.sum())))
    print(f"      训练样本：正 {st['pos_n']:,} / 负 {st['neg_n']:,}（用户 {st['n_users_total']:,}，"
          f"含正 {st['n_users_with_pos']:,}）", flush=True)

    nrow_u = ds.groupby("user_id").size()
    w_macro = (1.0 / np.sqrt(nrow_u.reindex(ds["user_id"]).to_numpy(dtype="float64"))).astype("float64")
    eff = w_macro * np.where(y == 1, spw, 1.0)

    # ---- 2. 训练 FM ----
    print("[2/3] FM 训练（k=%d, lr=%.1f, steps=%d, macro×scale_pos）…" % (FM_K, FM_LR, FM_STEPS),
          flush=True)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float64)
    user2i, code2i = b25.make_id_maps(codes_tr, users_tr)
    uidx_tr = np.asarray([user2i[int(x)] for x in users_tr], dtype="int64")
    iidx_tr = np.asarray([code2i[str(x)] for x in codes_tr], dtype="int64")
    fm = b25.NumpyFM(k=FM_K, lr=FM_LR, steps=FM_STEPS, wd=FM_WD, seed=SEED,
                     verbose=800).fit(Xs, uidx_tr, iidx_tr, y, eff)
    print(f"      FM 训练完成（{time.time()-t0:.0f}s）", flush=True)

    # ---- 3. 打分：候选集（切窗 11-01）----
    print("[3/3] 候选集打分（cut=2010-11-01，向量缓存复用）…", flush=True)
    bundle_sc = cand.build_bundle(clean, cut=SC_CUT, vec_cache=VEC_SC)
    raw = pd.read_csv(CAND_INPUT, dtype={c: str for c in pd.read_csv(CAND_INPUT, nrows=0).columns})
    lower = {c: str(c).lower() for c in raw.columns}
    uc = next(c for c in raw.columns if lower[c] in {"user_id", "customerid"})
    ic = next(c for c in raw.columns if lower[c] in {"item_id", "stockcode", "stock_code"})
    df = raw[[uc, ic]].copy()
    df.columns = ["user_id", "item_id"]
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("int64")
    df = df.dropna(subset=["user_id", "item_id"]).reset_index(drop=True)
    users = df["user_id"].to_numpy(dtype="int64")
    codes = df["item_id"].astype(str).to_numpy(dtype=object)

    Xsf = cand.features_for_pairs(bundle_sc, users, codes)
    assert list(Xsf.columns) == feats and Xsf.isna().sum().sum() == 0
    Xs_ev = scaler.transform(Xsf.to_numpy(dtype=np.float64)).astype(np.float64)
    uidx_te = np.asarray([user2i.get(int(x), 0) for x in users], dtype="int64")
    iidx_te = np.asarray([code2i.get(str(x), 0) for x in codes], dtype="int64")
    score = fm.predict(Xs_ev, uidx_te, iidx_te).astype(np.float64)

    # 口径纯排序 ⇒ 输出全精度 float（不 round 6 位）。FM 概率在 sigmoid 饱和处大批趋近 0/1，
    # round 6 位会把 1.9 万行压成 0.0、8.8 千行压成 1.0 → 每用户顶部队列大片并列（均值 6.4/9），
    # 等值块内顺序任意会钝化 FM 的顶置排序。全精度保留 float64 区分度，几乎无并列。
    out = pd.DataFrame({"user_id": df["user_id"], "item_id": df["item_id"],
                        "score": score})
    assert len(out) == len(df) and out.isna().sum().sum() == 0
    out_path = PROJECT_ROOT / "outputs" / "candidate" / f"sample_submission_{NAME}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")

    # ---- 4. 权重落盘 + meta ----
    P = fm.P
    wpath = PROJECT_ROOT / "models" / f"candidate_{NAME}.npz"
    np.savez(wpath, Vd=P["Vd"], Eu=P["Eu"], Ei=P["Ei"], w=P["w"], bu=P["bu"], bi=P["bi"],
             b0=np.asarray(P["b0"]),
             k=np.asarray(FM_K),
             scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
             n_feat=np.asarray(len(feats)))
    meta = {
        "name": NAME, "model_family": "numpy-FM (2nd-order, user/item id low-rank + 33 dense)",
        "train_cut": str(TR_CUT.date()), "score_cut": str(SC_CUT.date()),
        "fm": {"k": FM_K, "lr": FM_LR, "steps": FM_STEPS, "wd": FM_WD, "seed": SEED},
        "weight": "macro x scale_pos (eff = macro * (spw if pos else 1))",
        "feature_order": feats, "n_feature_cols": len(feats),
        "n_users_train": int(st["n_users_total"]), "n_users_with_pos": int(st["n_users_with_pos"]),
        "n_embed_users": int(P["Eu"].shape[0] - 1), "n_embed_items": int(P["Ei"].shape[0] - 1),
        "train_stats": st,
        "backtest_ref": "scripts/25 Sep->Oct: fm avg_rec 0.307 vs lgb 0.245; "
                        "scripts/26 Aug->Sep: fm rec@10 0.259 vs lgb 0.103",
        "weight_file": wpath.name,
    }
    meta_path = PROJECT_ROOT / "models" / f"candidate_meta_{NAME}.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    owned = sum(1 for u, c in zip(users, codes)
                if int(u) in bundle_sc["owned_fp"] and str(c) in bundle_sc["owned_fp"][int(u)])
    print(f"[版本] {NAME} → {out_path}")
    print(f"[覆盖] 输出 {len(out):,} 行；fp 历史已购对 {owned:,}（~{owned / len(out):.1%}）")
    print(f"[分数] min={score.min():.4f} mean={score.mean():.4f} max={score.max():.4f}")
    print(f"[权重] {wpath.name}；[meta] {meta_path.name}；总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
