"""把 v3 的 5 个新特征追加到已建的回放 wide 行 → 新文件 wide_v3_*.csv（不覆盖 wide_v2）。

对每个训练月（Jun..Oct），读 replay/wide_v2_YYYY-MM.csv 的行（user_id,item_id），在该月
cut（月首）上用 src/candidate_v3.v3_features 算 5 列（只用 < cut 历史，与 base/v2 同口径）
→ 原列 + 5 列写成 wide_v3_YYYY-MM.csv。v3 纯附加、逐位确定。

用法：venv\\Scripts\\python.exe scripts\\61_build_wide_v3.py
产物：outputs/candidate/replay/wide_v3_2010-{06..10}.csv
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

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import candidate as cand
from src.candidate_v3 import v3_features, V3_FEATS

REPLAY = PROJECT_ROOT / "outputs" / "candidate" / "replay"
LEADER_CLEAN = PROJECT_ROOT / "data" / "processed" / "leader_clean.csv"
MONTHS = [f"2010-{m:02d}-01" for m in range(6, 11)]


def main() -> None:
    t0 = time.time()
    clean = cand.load_clean_train(LEADER_CLEAN)
    for cut_s in MONTHS:
        cut = pd.Timestamp(cut_s)
        fn = f"wide_v2_{cut_s[:7]}.csv"
        w = pd.read_csv(REPLAY / fn, dtype={"item_id": str})
        rows = w[["user_id", "item_id"]].copy()
        extra = v3_features(clean, cut, rows)
        assert list(extra.columns) == V3_FEATS and extra.isna().sum().sum() == 0
        w3 = pd.concat([w.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
        assert len(w3) == len(w) and w3.isna().sum().sum() == 0
        out_fn = f"wide_v3_{cut_s[:7]}.csv"
        w3.to_csv(REPLAY / out_fn, index=False, encoding="utf-8")
        print(f"[{cut_s[:7]}] {out_fn}：{len(w3):,} 行 / {len(w.columns)+len(V3_FEATS)} 列"
              f"（{time.time()-t0:.0f}s）", flush=True)
    print(f"[done] v3 宽表完成，总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
