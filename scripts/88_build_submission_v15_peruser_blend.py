"""88：v15×v6 融合的「逐用户组内百分位」版（新归一化轴，未上板）。

背景：老师口径 = 逐 (user_id, snapshot_month) 组内排序后对用户等权 macro（probe B 钉死）。
v13 全轴 6 发（97/85/70/65/60/75）烧的都是**全局百分位**融合
  score = w·rank_global(v15) + (1−w)·rank_global(v6)
全局百分位把「同用户内 a/b 的位次差」混进了「跨用户被其它候选隔开的距离」——用老师从不读取的
跨用户信息做组内决策。本脚本烧**逐用户组内百分位**（rank_within_user / 该用户候选数）融合，
是同一配方在这个指标下的「干净」版本：
  score_u(c) = w·pct_u(v15,c) + (1−w)·pct_u(v6,c),  pct_u = 该候选在本用户内的百分位

与已烧全局版的关系：同 w=0.65 时 top-10 差 269 席（58,205 行 / 736 用户）——是不同的排序，
不是旧轴重揉。EV≥0（榜保留最佳）：可能破全局峰 0.7863，最差回落 ~.786 邻域，无下沉风险。

CSV 真伪纪律：纯秩运算、确定性、由组长 Nov 提交与 v6 两源文件逐行对齐 58,205 行重算。
产物：cand-align-v15pu-blend{65,70} 提交 + meta
用法：venv\\Scripts\\python.exe scripts\\88_build_submission_v15_peruser_blend.py
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
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
LEADER_SUB = ROOT / "data" / "submission_traincsv_v15_activity_moe_blend.csv"
V6_SUB = ROOT / "outputs" / "candidate" / "sample_submission_cand-align-v6-fuse.csv"
CAND = ROOT / "train_data" / "sample_candidates.csv"
NEW_W = [0.65, 0.70]


def pct_within_user(df: pd.DataFrame, col: str) -> np.ndarray:
    """组内百分位：rankdata(average)/该组行数。等权组归一，不跨用户比较。"""
    out = df.groupby("user_id", sort=False)[col].transform(
        lambda s: rankdata(s.to_numpy(float), method="average") / len(s))
    return out.to_numpy(float)


def main() -> None:
    t0 = time.time()
    cand = pd.read_csv(CAND, dtype={"user_id": str, "item_id": str})
    v15 = pd.read_csv(LEADER_SUB)
    v6 = pd.read_csv(V6_SUB)
    for df in (v15, v6):
        df["user_id"] = df["user_id"].astype(str)
        df["item_id"] = df["item_id"].astype(str)
    for nm, df in (("cand", cand), ("v15", v15), ("v6", v6)):
        assert df[["user_id", "item_id"]].equals(cand[["user_id", "item_id"]]), f"{nm} 对齐失败"
    n_user = cand.groupby("user_id").size()
    print(f"[aligned] {len(cand)} rows / {n_user.size} users; "
          f"cand/user median {n_user.median():.0f} (min {n_user.min()}, max {n_user.max()})")

    r15 = pct_within_user(v15, "score")      # 0..1 per user
    r6 = pct_within_user(v6, "score")
    for w in NEW_W:
        score = w * r15 + (1.0 - w) * r6
        name = f"cand-align-v15pu-blend{int(round(w * 100)):02d}"
        p = ROOT / "outputs" / "candidate" / f"sample_submission_{name}.csv"
        pd.DataFrame({"user_id": cand["user_id"], "item_id": cand["item_id"], "score": score}
                     ).to_csv(p, index=False, encoding="utf-8")
        # 与已烧全局版（同 w）的 top-10 差异，只诊断不落盘
        rk_pu = pd.DataFrame({"user_id": cand["user_id"], "item_id": cand["item_id"], "score": score})
        rk_pu["rk"] = rk_pu.groupby("user_id")["score"].rank(method="first", ascending=False)
        pu10 = set(rk_pu[rk_pu["rk"] <= 10].set_index(["user_id", "item_id"]).index)
        gbl = ROOT / "outputs" / "candidate" / f"sample_submission_cand-align-v13-v15blend{int(round(w*100)):02d}.csv"
        if gbl.exists():
            gd = pd.read_csv(gbl, dtype={"user_id": str, "item_id": str})
            gd["rk"] = gd.groupby("user_id")["score"].rank(method="first", ascending=False)
            g10 = set(gd[gd["rk"] <= 10].set_index(["user_id", "item_id"]).index)
            print(f"w={w}: peruser-top10 vs global-top10 seat overlap {len(pu10 & g10)} / "
                  f"{len(g10)}  (differ {len(g10) - len(pu10 & g10)})")
        meta = {"name": name,
                "desc": f"{w}×v15(组长 submission) 组内百分位 + {1.0-w:.2f}×v6 组内百分位 逐用户融合 "
                        f"(归一化口径=逐用户，对应用户 macro 指标)",
                "v15_board": 0.7845, "v6_board": 0.7782, "w_v15": w,
                "norm": "per-user percentile (group=(user_id))", "global_analog_score": 0.7863,
                "family": "cand-align-v15pu-blend",
                "note": "新归一化轴：老师口径逐用户 macro ⇒ 组内百分位为度量一致融合；全局版 6 发已钉峰 .7863"}
        (ROOT / "models" / f"candidate_meta_{name}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[wrote] sample_submission_{name}.csv ({len(cand)} rows) + meta")
    print(f"\n[done {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
