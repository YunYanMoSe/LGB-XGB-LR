"""候选集打分任务 ⑩（v9 对标线）：候选对齐时序回放 —— 建 5 截点逐月回放宽表。

方法（组长黄俊霖 v9，表 I/IV/VI）：
    * 用【组长清洗口径】重建数据（332,848 → 327,827：删取消/退货、非正价、非法量、
      缺用户/缺主键、无效日期、整行重复；【保留服务码】）。注意：旧 clean_train.csv 是
      326,712 行（scripts/18 多删了服务码），不能复用 → 新写 leader_clean.csv；
    * 对标签月 t ∈ {2010-06..2010-10}：历史 H<t = 交易日期 < t 月初；
      C_t = 固定候选集中「用户 ∈ H<t 的用户 且 商品 ∈ H<t 的目录」的子集（当时可计算）；
      样本 = C_t 每一行在截点 t 上的 33 维打分特征（candidate.features_for_pairs，仅用 H<t）；
      标签 = (u,i) 是否在 t 月（当月）被真实购买；new_combo = 标签=1 且 (u,i) ∉ H<t 已购；
    * 宽表列：user_id / item_id(str) / 33 特征 / label / new_combo / month。
    * 最终模型 = 全部回放样本重训 → 用 cut=11-01（完整历史）对 58,205 候选行打分（本脚本不产提交）。

关键约束：
    * 逐月 item2vec 向量缓存独立路径（vectors 只能由该月 H<t 训练，防泄漏）；
    * 每个 (cut, 数据切片) 的 bundle 与特征只用 < cut 的历史；
    * 全部产物新文件，不覆盖任何旧产物。

用法：venv\\Scripts\\python.exe scripts\\29_build_replay_wide.py [--test]   （--test 只建 2010-06 验证）
产物：
    data/processed/leader_clean.csv                    组长口径清洗（327,827 行）
    outputs/candidate/replay/wide_2010-{06..10}.csv    逐月回放宽表（含 label/new_combo/month）
    outputs/candidate/replay/replay_meta.json
"""
from __future__ import annotations

import argparse
import json
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

RAW = PROJECT_ROOT / "data" / "raw" / "train.csv"
CAND_CSV = PROJECT_ROOT / "train_data" / "sample_candidates.csv"
LEADER_CLEAN = PROJECT_ROOT / "data" / "processed" / "leader_clean.csv"
REPLAY_DIR = PROJECT_ROOT / "outputs" / "candidate" / "replay"

USER = COLUMNS["user"]
ITEM = COLUMNS["item"]
TIME = COLUMNS["time"]

MONTHS = [pd.Timestamp(f"2010-{m:02d}-01") for m in range(6, 11)]  # 5 个标签月
FEATS = cand.FEATURES_CAND  # 33 列


def clean_leaders_way(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df = df.dropna(subset=[USER])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C", na=False)]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df[df[ITEM].astype(str).str.strip() != ""]
    df = df[pd.to_datetime(df[TIME], errors="coerce").notna()]
    df = df.drop_duplicates().reset_index(drop=True)
    df[TIME] = pd.to_datetime(df[TIME])
    df[USER] = pd.to_numeric(df[USER], errors="coerce").astype("int64")
    df[ITEM] = df[ITEM].astype(str)
    return df


def vec_cache_for(cut: pd.Timestamp) -> Path:
    return PROJECT_ROOT / "outputs" / "candidate" / f"v9_vecs_{cut:%Y%m%d}_skip.npz"


def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    # ---- 1. 组长口径 clean（缓存到磁盘，避免每次重洗）----
    if LEADER_CLEAN.exists():
        clean = cand.load_clean_train(LEADER_CLEAN)
        print(f"[1/4] leader_clean 已存在：{len(clean):,} 行", flush=True)
    else:
        raw = pd.read_csv(RAW)
        clean = clean_leaders_way(raw)
        LEADER_CLEAN.parent.mkdir(parents=True, exist_ok=True)
        clean.to_csv(LEADER_CLEAN, index=False, encoding="utf-8")
        print(f"[1/4] 组长口径清洗：raw={len(raw):,} → clean={len(clean):,}", flush=True)

    # ---- 2. 候选集（字符串 item）----
    candf = pd.read_csv(CAND_CSV, dtype={c: str for c in ["user_id", "item_id"]})
    candf.columns = ["user_id", "item_id"]
    candf["user_id"] = pd.to_numeric(candf["user_id"]).astype("int64")
    candf["item_id"] = candf["item_id"].astype(str)
    print(f"[2/4] 候选 {len(candf):,} 行 / {candf['user_id'].nunique():,} 用户 / "
          f"{candf['item_id'].nunique():,} 商品", flush=True)

    # ---- 3. 逐月回放 ----
    months = MONTHS[:1] if args.test else MONTHS
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    stats_all = []
    for i, cut in enumerate(months):
        t_m = time.time()
        nxt = cut + pd.offsets.MonthBegin(1)
        fp = clean[clean[TIME] < cut]
        lp = clean[(clean[TIME] >= cut) & (clean[TIME] < nxt)]
        fp_users = set(int(x) for x in fp[USER].unique())
        fp_items = set(str(x) for x in fp[ITEM].unique())

        # C_t：候选行中用户、商品在截点 t 均已知
        sub = candf[candf["user_id"].isin(fp_users) & candf["item_id"].isin(fp_items)]
        users = sub["user_id"].to_numpy(dtype="int64")
        codes = sub["item_id"].astype(str).to_numpy(dtype=object)
        print(f"[3/4] {cut.date()}：C_t={len(sub):,} 行（fp 用户 {len(fp_users):,} / 目录 {len(fp_items):,}）",
              flush=True)

        # 当月真实购买集合
        lp_bought: set[tuple[int, str]] = set()
        for u, i in zip(lp[USER], lp[ITEM]):
            lp_bought.add((int(u), str(i)))
        fp_bought: set[tuple[int, str]] = set()
        for u, i in zip(fp[USER], fp[ITEM]):
            fp_bought.add((int(u), str(i)))

        # 逐月 bundle（含当月专属 item2vec 向量缓存）
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

        fn = REPLAY_DIR / f"wide_{cut:%Y-%m}.csv"
        wide.to_csv(fn, index=False, encoding="utf-8")
        stats_all.append({"month": str(cut.date()), "samples": len(wide),
                          "pos": int(labels.sum()), "new": int(newc.sum()),
                          "file": fn.name})
        print(f"       wide={len(wide):,}  pos={labels.sum():,} ({labels.mean():.4%})  "
              f"new={newc.sum():,}  耗时 {time.time()-t_m:.0f}s", flush=True)
        del bundle, wide, X

    # ---- 4. meta ----
    tot_s = sum(s["samples"] for s in stats_all)
    tot_p = sum(s["pos"] for s in stats_all)
    meta = {"method": "candidate-aligned temporal replay (leader v9)",
            "months": stats_all,
            "total_samples": tot_s, "total_pos": tot_p,
            "pos_rate": tot_p / max(1, tot_s),
            "features": FEATS, "n_features": len(FEATS),
            "clean_note": "leader cleaning (keep service codes), clean=%d" % len(clean),
            "label": "real purchase in label month (含复购)", }
    (REPLAY_DIR / "replay_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[4/4] 共 {tot_s:,} 样本 / {tot_p:,} 正例（{tot_p/max(1,tot_s):.4%}）；"
          f"meta → replay_meta.json；总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
