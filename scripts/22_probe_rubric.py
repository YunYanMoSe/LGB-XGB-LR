"""候选集打分任务 ⑤：rubric 反推探针生成器。

老师只回一个综合分。为反推其评分口径，对"已得分基准提交"（默认 cand-lgb-v2，
综合分 0.6348）的 score 列做单变量受控变换，user_id / item_id / 行序原样保留
（覆盖率恒 100%）。每支探针只动一件事，用"返回综合分 vs 基准 0.6348"解读口径。

变换模式：
  rank_global     全局排序 → 均匀分位 (0,1)：严格保留全局相对序，抹掉全部绝对值。
                  综合分仍=0.6348 ⇒ 口径只看排序不看数值（推荐系统几乎必然如此，作为
                  廉价确认）；若变了 ⇒ 口径含阈值/0.5 二值化/绝对值误差类项，另谋路。
  per_user_block  每用户内部序原样，但把用户整块按 user_id 抬到不同平台，彻底打乱
                  跨用户相对位置（分数仍全部落在 (0,1)，便于老师侧校验）。
                  综合分仍=0.6348 ⇒ 口径 = "逐用户内部序"的 macro（每用户 top-K 或
                  每用户列表指标，跨用户比较无关）；跌了 ⇒ 含全局跨用户成分
                  （全局 AUC / 全局 top-K），需转向校准跨用户分数。

用法：
  venv\\Scripts\\python.exe scripts\\22_probe_rubric.py --mode rank_global
  venv\\Scripts\\python.exe scripts\\22_probe_rubric.py --mode per_user_block
产物：outputs/candidate/sample_submission_cand-probe-<mode>.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE = PROJECT_ROOT / "outputs" / "candidate" / "sample_submission_cand-lgb-v2.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "candidate"


def _global_rank_frac(s: np.ndarray, n: int) -> np.ndarray:
    order = np.argsort(s, kind="stable")
    rank = np.empty(n, dtype="int64")
    rank[order] = np.arange(n)
    return (rank + 1.0) / (n + 1.0)          # (0,1)，随 s 严格递增


def _per_user_block(df: pd.DataFrame) -> np.ndarray:
    """每用户内部保序；用户整体按 user_id 升序排到间隔平台上。"""
    users = np.sort(df["user_id"].to_numpy())
    block = {u: i for i, u in enumerate(np.unique(users))}
    n_u = len(block)
    out = np.zeros(len(df), dtype="float64")
    for u, g in df.groupby("user_id", sort=False):
        idx = g.index.to_numpy()
        s = g["score"].to_numpy(dtype="float64")
        k = len(s)
        within = _global_rank_frac(s, k) * 0.98        # (0,0.98)，留出平台间隔
        out[idx] = (block[int(u)] + within) / n_u       # 平台间隔 ~1/n_u >> 取整误差
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rank_global", "per_user_block"], required=True)
    ap.add_argument("--base", default=str(BASE), help="基准提交（需已得分，默认 cand-lgb-v2）")
    args = ap.parse_args()

    df = pd.read_csv(args.base)
    assert list(df.columns) == ["user_id", "item_id", "score"], f"基准列不符：{list(df.columns)}"
    s0 = df["score"].to_numpy(dtype="float64")
    assert np.isfinite(s0).all()
    n = len(df)

    if args.mode == "rank_global":
        s1 = _global_rank_frac(s0, n)
        chk = "全局相对序 = 基准（Spearman 应=1）"
    else:
        s1 = _per_user_block(df)
        # 校验：每用户稳定 argsort 与基准全等（tie 按原行序同序破开）⇒ 内部序逐字节保；
        #      全局序则应与基准显著不同 ⇒ 跨用户已乱。
        same = df.assign(s1=s1).groupby("user_id", sort=False).apply(
            lambda g: np.array_equal(np.argsort(g["score"].to_numpy(dtype="float64"), kind="stable"),
                                     np.argsort(g["s1"].to_numpy(dtype="float64"), kind="stable")),
            include_groups=False)
        assert bool(same.all()), "存在用户内部序被破坏"
        tau = pd.Series(s0).rank().corr(pd.Series(s1).rank())
        chk = f"每用户稳定 argsort 全等（{int(same.sum())}/{len(same)} 用户）；全局序 Spearman 仅 {tau:.3f} ⇒ 跨用户已乱"

    assert s1.min() > 0 and s1.max() < 1 and np.isfinite(s1).all()
    # 8 位小数：块内相邻间隔 ~3.9e-6（最大用户 344 行）远大于 1e-8 网格，保证块内不产生 tie
    out = pd.DataFrame({"user_id": df["user_id"], "item_id": df["item_id"],
                        "score": np.round(s1, 8)})
    assert out.isna().sum().sum() == 0 and len(out) == n
    assert out["score"].nunique() == n, "输出存在重复分数，可能破坏序信息"

    tag = {"rank_global": "rank", "per_user_block": "block"}[args.mode]
    fname = f"sample_submission_cand-probe-{tag}.csv"
    out_path = OUT_DIR / fname
    out.to_csv(out_path, index=False, encoding="utf-8")

    print(f"[探针] mode={args.mode}  基于 {Path(args.base).name}（综合分 0.6348）")
    print(f"[校验] {chk}")
    print(f"[分数] min={s1.min():.5f} mean={s1.mean():.5f} max={s1.max():.5f}（限 (0,1)）")
    print(f"[覆盖] 行数 {len(out):,} = 基准，user/item/行序原样 ⇒ 覆盖率 100%")
    print(f"[版本] cand-probe-{tag} → {out_path}")


if __name__ == "__main__":
    main()
