"""候选集打分 v13 精扫（scripts/73+78 菜单的 60-85 峰区补点）：w_v15 ∈ {0.65, 0.75}。

烧板结果 97=.7848 / 85=.7850 / 70=.7862 / 60=.7858 ⇒ 峰在 60-85 之间、≈0.70。补 65/75 两档
精扫确认峰形；若 65 ≥ .7862 峰偏左、75 ≥ .7862 峰偏右，否则 70=.7862 即为最优。

CSV 真伪纪律：纯秩操作，由组长 Nov 提交与 v6 两源文件确定性重算，可复现到 1 ULP。
产物：cand-align-v13-v15blend{65,75} 提交 + meta
用法：venv\\Scripts\\python.exe scripts\\79_build_submission_v13_refine.py
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
NEW_W = [0.65, 0.75]


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

    N = len(cand)
    r15 = rankdata(v15["score"].to_numpy(float)) / N
    r6 = rankdata(v6["score"].to_numpy(float)) / N
    for w in NEW_W:
        score = w * r15 + (1.0 - w) * r6
        name = f"cand-align-v13-v15blend{int(round(w * 100)):02d}"
        p = ROOT / "outputs" / "candidate" / f"sample_submission_{name}.csv"
        pd.DataFrame({"user_id": cand["user_id"], "item_id": cand["item_id"], "score": score}
                     ).to_csv(p, index=False, encoding="utf-8")
        meta = {"name": name,
                "desc": f"{w}×v15(组长 submission) 秩 + {1.0-w:.2f}×v6 秩 全局百分位融合",
                "v15_board": 0.7845, "v6_board": 0.7782, "w_v15": w,
                "family": "cand-align-v13-v15blend", "note": "60-85 峰区精扫（70=.7862 峰）"}
        (ROOT / "models" / f"candidate_meta_{name}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[out] {name} → {p.name}  ({time.time()-t0:.0f}s)", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
