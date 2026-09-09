"""候选集打分 v2 特征：在 v1 回放宽表（33 基础特征）基础上，加法式追加 11 个 v2 特征。

只新增文件（wide_v2_*.csv），不覆盖 v1 产物。行宇宙/标签与 v1 逐行一致：
    * 行 = 每月 C_t（当月可计算候选行），标签 = 当月真实购买；
    * v1 基础 33 列照抄回放宽表 wide_2010-MM.csv（脚本 29 已建并复核）；
    * v2 列 = src/candidate_v2.v2_features(clean, cut, 当月行) —— 只 < cut 历史 + 当月候选宇宙。

用法：venv\\Scripts\\python.exe scripts\\32_build_replay_v2.py [--test]
产物：outputs/candidate/replay/wide_v2_2010-{06..10}.csv（= 原列 + V2_FEATS）
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

from src import candidate as cand
from src.candidate_v2 import v2_features, V2_FEATS

REPLAY_DIR = PROJECT_ROOT / "outputs" / "candidate" / "replay"
LEADER_CLEAN = PROJECT_ROOT / "data" / "processed" / "leader_clean.csv"
MONTHS = [f"2010-{m:02d}-01" for m in range(6, 11)]


def main() -> None:
    t0 = time.time()
    if not REPLAY_DIR.joinpath("wide_2010-10.csv").exists():
        raise SystemExit("先跑 scripts/29_build_replay_wide.py 建 v1 回放宽表")
    clean = cand.load_clean_train(LEADER_CLEAN)
    months = MONTHS[:1] if "--test" in sys.argv else MONTHS
    for m in months:
        cut = pd.Timestamp(m)
        base = pd.read_csv(REPLAY_DIR / f"wide_{m[:7]}.csv", dtype={"item_id": str})
        # base 列序：user_id, item_id, 33 特征, label, new_combo, month —— 保持原样 + v2 追尾
        rows = base[["user_id", "item_id"]].copy()
        extra = v2_features(clean, cut, rows)
        v2 = pd.concat([base.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
        fn = REPLAY_DIR / f"wide_v2_{m[:7]}.csv"
        v2.to_csv(fn, index=False, encoding="utf-8")
        print(f"[v2] {m[:7]}：{len(v2):,} 行（base {len(base.columns)} 列 + v2 {len(V2_FEATS)} 列）→ {fn.name}",
              flush=True)
    print(f"[done] {len(months)} 月；总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
