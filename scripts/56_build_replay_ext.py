"""候选集打分：回放前伸——补建 2010-02..05 四个更早标签月的 wide_v2 回放行。

目的（方向 2 实验）：把 serve 训练池从 Jun..Oct 5 个月扩到 Feb..Oct 9 个月，给树族更多
训练量 / 让每折训练历史更贴近真实 serve 的深历史。OOF 验证折叠仍是 Jul..Oct（可比 .8721），
只加训练行，不动已有 June..Oct 产物。

方法与 组长 v9 一致：标签月 t ∈ {2010-02..2010-05}，历史 H<t = 交易 < t 月初；
C_t = 固定候选集中「用户∈H<t 用户 且 商品∈H<t 目录」的子集；特征只用 H<t（33 base，
candidate.features_for_pairs）；标签 = (u,i) 是否在 t 月被真实购买；再追尾 v2 的 11 列
（src.candidate_v2.v2_features，同样只用 < t 历史）→ wide_v2 44 列。

全部新文件（wide_v2_2010-02..05.csv + 逐月 item2vec 缓存 v9_vecs_20100{2..5}01_skip.npz），
绝不覆盖现存 June..Oct 回放/缓存/其他旧产物。

用法：venv\\Scripts\\python.exe scripts\\56_build_replay_ext.py
产物：outputs/candidate/replay/wide_v2_2010-{02..05}.csv
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

from config.settings import COLUMNS
from src import candidate as cand
from src.candidate_v2 import v2_features, V2_FEATS

RAW = PROJECT_ROOT / "data" / "raw" / "train.csv"
CAND_CSV = PROJECT_ROOT / "train_data" / "sample_candidates.csv"
LEADER_CLEAN = PROJECT_ROOT / "data" / "processed" / "leader_clean.csv"
REPLAY_DIR = PROJECT_ROOT / "outputs" / "candidate" / "replay"

USER = COLUMNS["user"]
ITEM = COLUMNS["item"]
TIME = COLUMNS["time"]

MONTHS = [pd.Timestamp(f"2010-{m:02d}-01") for m in range(2, 6)]  # 前伸 2010-02..05
FEATS = cand.FEATURES_CAND  # 33 列 base


def vec_cache_for(cut: pd.Timestamp) -> Path:
    return PROJECT_ROOT / "outputs" / "candidate" / f"v9_vecs_{cut:%Y%m%d}_skip.npz"


def main() -> None:
    t0 = time.time()
    if not LEADER_CLEAN.exists():
        raise SystemExit("缺 data/processed/leader_clean.csv，先跑 scripts/29_build_replay_wide.py")
    clean = cand.load_clean_train(LEADER_CLEAN)
    candf = pd.read_csv(CAND_CSV, dtype={c: str for c in ["user_id", "item_id"]})
    candf.columns = ["user_id", "item_id"]
    candf["user_id"] = pd.to_numeric(candf["user_id"]).astype("int64")
    candf["item_id"] = candf["item_id"].astype(str)

    stats = []
    for cut in MONTHS:
        t_m = time.time()
        nxt = cut + pd.offsets.MonthBegin(1)
        fp = clean[clean[TIME] < cut]
        lp = clean[(clean[TIME] >= cut) & (clean[TIME] < nxt)]
        fp_users = set(int(x) for x in fp[USER].unique())
        fp_items = set(str(x) for x in fp[ITEM].unique())

        sub = candf[candf["user_id"].isin(fp_users) & candf["item_id"].isin(fp_items)]
        users = sub["user_id"].to_numpy(dtype="int64")
        codes = sub["item_id"].astype(str).to_numpy(dtype=object)
        print(f"[{cut.month:02d}] {cut.date()}：C_t={len(sub):,} 行"
              f"（fp 用户 {len(fp_users):,} / 目录 {len(fp_items):,}）", flush=True)
        if len(sub) == 0:
            print(f"[{cut.month:02d}] 跳过：C_t 为空", flush=True)
            continue

        lp_bought = {(int(u), str(c)) for u, c in zip(lp[USER], lp[ITEM])}
        fp_bought = {(int(u), str(c)) for u, c in zip(fp[USER], fp[ITEM])}

        vc = vec_cache_for(cut)
        bundle = cand.build_bundle(clean, cut=cut, vec_cache=vc)
        X = cand.features_for_pairs(bundle, users, codes)
        assert list(X.columns) == FEATS and X.isna().sum().sum() == 0

        labels = np.asarray([1 if (int(u), str(c)) in lp_bought else 0
                             for u, c in zip(users, codes)], dtype="int64")
        newc = np.asarray([1 if (int(u), str(c)) in lp_bought
                           and (int(u), str(c)) not in fp_bought else 0
                           for u, c in zip(users, codes)], dtype="int64")

        wide = pd.DataFrame({"user_id": users, "item_id": codes})
        wide = pd.concat([wide, X.reset_index(drop=True)], axis=1)
        wide["label"] = labels
        wide["new_combo"] = newc
        wide["month"] = str(cut.date())

        rows = wide[["user_id", "item_id"]].copy()
        extra = v2_features(clean, cut, rows)
        v2 = pd.concat([wide.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
        fn = REPLAY_DIR / f"wide_v2_{cut:%Y-%m}.csv"
        v2.to_csv(fn, index=False, encoding="utf-8")
        stats.append({"month": str(cut.date()), "samples": len(v2), "pos": int(labels.sum())})
        print(f"[{cut.month:02d}] 写 {fn.name}：{len(v2):,} 行 / 正例 {labels.sum():,}"
              f"（{labels.mean():.4%}）耗时 {time.time()-t_m:.0f}s", flush=True)
        del bundle, v2, wide, X

    tot_s = sum(s["samples"] for s in stats)
    tot_p = sum(s["pos"] for s in stats)
    print(f"[done] 前伸 {len(stats)} 月，共 {tot_s:,} 行 / {tot_p:,} 正例"
          f"（{tot_p/max(1, tot_s):.4%}）；总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
