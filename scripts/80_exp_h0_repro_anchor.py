"""Exp-H0：用统一评分器复现 v15 anchor（blend_v15）OOF 指标。

通过条件：整体 composite 与 scripts/74 已有基准 blend_v15 = 0.57753 一致（浮点误差内），
逐月指标一并落盘，供后续路线对照。

产物：outputs/experiment_basket_hypergraph/
    h0_anchor_repro.txt   （逐月 + 整体指标，含与既有基准对比）
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.basket_experiment import io_data as io
from src.basket_experiment.metrics import composite_score, monthly_composite

# scripts/74 既有基准（同 146,530 行同口径）
BASELINE = {"blend_v15": {"composite": 0.57753, "GAUC": 0.8675,
                          "NDCG@10": 0.2630, "Recall@10": 0.6268}}


def main() -> None:
    t0 = time.time()
    io.OUT_DIR.mkdir(parents=True, exist_ok=True)
    oof = io.load_anchor_oof()
    assert len(oof) == 146530, f"anchor OOF 行数 {len(oof)} != 146530"
    assert oof["blend_v15"].notna().all()
    assert (oof["blend_v15"] > 0).all()

    lines = []
    lines.append("## Exp-H0 复现 anchor：data/candidate_aligned_oof_v15.csv 的 blend_v15")
    lines.append("口径：组=(user, snapshot_month) | composite=0.4*GAUC+0.4*NDCG@10+0.2*Recall@10（scripts/74 同款函数）")
    lines.append("")
    mt = monthly_composite(oof, "blend_v15")
    lines.append("### 逐月")
    lines.append(mt.to_string(index=False))
    ov = composite_score(oof, "blend_v15")
    lines.append("")
    lines.append("### 整体")
    lines.append(f"rows={len(oof):,}  composite={ov[0]:.5f}  GAUC={ov[1]:.4f}  "
                 f"NDCG@10={ov[2]:.4f}  Recall@10={ov[3]:.4f}")
    b = BASELINE["blend_v15"]
    ok = abs(ov[0] - b["composite"]) < 5e-4
    lines.append("")
    lines.append("### 与 scripts/74 既有基准对比")
    lines.append(f"  基准  composite={b['composite']:.5f}  GAUC={b['GAUC']:.4f}  "
                 f"NDCG@10={b['NDCG@10']:.4f}  Recall@10={b['Recall@10']:.4f}")
    lines.append(f"  复现  composite={ov[0]:.5f}  GAUC={ov[1]:.4f}  "
                 f"NDCG@10={ov[2]:.4f}  Recall@10={ov[3]:.4f}")
    lines.append(f"  Δcomposite={ov[0]-b['composite']:+.5f}  →  "
                 f"{'PASS' if ok else 'FAIL（需排查合并/排序/口径）'}")

    txt = "\n".join(lines) + "\n"
    out = io.OUT_DIR / "h0_anchor_repro.txt"
    out.write_text(txt, encoding="utf-8")
    print(txt)
    mt.to_csv(io.OUT_DIR / "monthly_metrics_anchor.csv", index=False)
    print(f"[done] {time.time()-t0:.0f}s → {out}")


if __name__ == "__main__":
    main()
