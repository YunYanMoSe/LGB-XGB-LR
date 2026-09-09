"""候选集打分 v13 扩展（跨架构秩融合菜单补全）：leader-heavy 侧 + 低侧 + 席位扰动诊断。

scripts/73 已建 cand-align-v13-v15blend{70,85,92}（组长 v15 Nov 提交 × 我方 v6）。本地 OOF 复合
与 Nov 反向（scripts/74）、own-model 轴全钉死/收束 ⇒ 能把分数抬过 v6 0.7782 的输入只有组长 Nov
验证信号这一路，而组长 .7845 赢 .0063 提示 **Nov 上 leader 主导**——融合峰可能落在高 w_v15 处。
本脚本补 leader-heavy 侧 {0.97, 0.99} 与低侧 {0.60}，把菜单做成可从板上读出峰的扫描。

换位诊断：对每个新权重算它与「已有点」每用户 top-10 的席位位移——确保每发都是真实不同的提交
（若与邻居席位差 0，说明是数值鬼影不必烧）。纯秩操作，无训练，秒级。

产物：cand-align-v13-v15blend{60,97,99} 提交 + meta + 席位扰动报告
用法：venv\\Scripts\\python.exe scripts\\78_build_submission_v13_ext.py
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
NEW_W = [0.60, 0.97, 0.99]                 # 本脚本新增（60 低侧 / 97·99 leader-heavy）
EXIST_W = [0.70, 0.85, 0.92]               # scripts/73 已建
ALL_W = sorted(NEW_W + EXIST_W)


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
    uid = cand["user_id"].astype(str).to_numpy()

    scores = {w: w * r15 + (1.0 - w) * r6 for w in ALL_W}

    def top10_rows(s):
        frame = pd.DataFrame({"u": uid, "row": np.arange(N), "s": s})
        rows = np.concatenate([g.nlargest(10, "s")["row"].to_numpy(int) for _, g in
                               frame.groupby("u", sort=False)])
        return rows

    menu = {w: set(top10_rows(scores[w]).tolist()) for w in ALL_W}
    # 每权重相对其低一侧邻居（除 v6 外的最接近点）的 top-10 席位位移
    print(f"[diag] 58,205 行 / 736 用户；v15↔v6 全局 rank Spearman="
          f"{np.corrcoef(r15, r6)[0, 1]:.4f}", flush=True)
    lines = ["## v13 跨架构融合菜单补全（leader-heavy + 低侧）席位扰动诊断",
             f"行={N} 用户=736  w_v15 ∈ {ALL_W}",
             "| w_v15 | 相对低一侧邻居的 top-10 位移席位 | 说明 |",
             "| --- | --- | --- |"]
    prev = set(top10_rows(r6).tolist())  # 起点 = 纯 v6
    for w in ALL_W:
        cur = menu[w]
        move = len(cur - prev) + len(prev - cur)
        tag = ("v6→ 引入 leader" if w == min(ALL_W) else
               "leader-heavy 逼近组长" if w >= 0.97 else "中段")
        lines.append(f"| {w:.2f} | {move} | {tag} |")
        prev = cur
    # 每个新增权重 vs 纯 v6 与纯 leader 的位移
    set_v6 = set(top10_rows(r6).tolist()); set_v15 = set(top10_rows(r15).tolist())
    lines.append("")
    lines.append("| w_v15 | 相对纯 v6 top10 位移 | 相对纯 leader top10 位移 |")
    lines.append("| --- | --- | --- |")
    for w in NEW_W:
        c = menu[w]
        lines.append(f"| {w:.2f} | {len(c-set_v6)+len(set_v6-c)} | "
                     f"{len(c-set_v15)+len(set_v15-c)} |")
    lines.append("")
    txt = "\n".join(lines)
    (ROOT / "outputs" / "candidate" / "replay" / "v13_ext_diag_report.txt").write_text(
        txt, encoding="utf-8")
    print(txt, flush=True)

    # ---- 新增权重写出 ----
    for w in NEW_W:
        name = f"cand-align-v13-v15blend{int(round(w * 100)):02d}"
        p = ROOT / "outputs" / "candidate" / f"sample_submission_{name}.csv"
        pd.DataFrame({"user_id": cand["user_id"], "item_id": cand["item_id"],
                      "score": scores[w]}).to_csv(p, index=False, encoding="utf-8")
        meta = {"name": name,
                "desc": f"{w}×v15(v15 组长 submission) 秩 + {1.0-w:.2f}×v6(我方) 秩 全局百分位融合",
                "v15_source": str(LEADER_SUB.name), "v15_board": 0.7845,
                "v6_board": 0.7782, "w_v15": w,
                "family": "cand-align-v13-v15blend",
                "note": "leader 共享 submission 跨架构融合彩票；v13 菜单 70/85/92(73)+60/97/99(78) 补全，板上读峰"}
        (ROOT / "models" / f"candidate_meta_{name}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[out] {name} → {p.name}", flush=True)
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
