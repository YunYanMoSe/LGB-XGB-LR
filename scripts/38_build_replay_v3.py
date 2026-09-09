"""候选集打分 v3 特征：在 wide_v2（44 特征）基础上再叠 3 个 item/pair 级 v3 特征。

行宇宙/标签与 v1/v2 逐行一致；wide_v3 = wide_v2 全列 + V3_FEATS（不覆盖任何旧产物）。

用法：venv\\Scripts\\python.exe scripts\\38_build_replay_v3.py [--test]
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
from src.candidate_v3 import extra_v3_features, V3_FEATS

REPLAY_DIR = PROJECT_ROOT / "outputs" / "candidate" / "replay"
LEADER_CLEAN = PROJECT_ROOT / "data" / "processed" / "leader_clean.csv"
MONTHS = [f"2010-{m:02d}-01" for m in range(6, 11)]


def main() -> None:
    t0 = time.time()
    if not REPLAY_DIR.joinpath("wide_v2_2010-10.csv").exists():
        raise SystemExit("先跑 scripts/32_build_replay_v2.py")
    clean = cand.load_clean_train(LEADER_CLEAN)
    months = MONTHS[:1] if "--test" in sys.argv else MONTHS
    for m in months:
        cut = pd.Timestamp(m)
        base = pd.read_csv(REPLAY_DIR / f"wide_v2_{m[:7]}.csv", dtype={"item_id": str})
        rows = base[["user_id", "item_id"]].copy()
        extra = extra_v3_features(clean, cut, rows)
        v3 = pd.concat([base.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
        fn = REPLAY_DIR / f"wide_v3_{m[:7]}.csv"
        v3.to_csv(fn, index=False, encoding="utf-8")
        print(f"[v3] {m[:7]}：{len(v3):,} 行（{len(base.columns)}+{len(V3_FEATS)} 列）→ {fn.name}",
              flush=True)
    print(f"[done] {len(months)} 月；总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
