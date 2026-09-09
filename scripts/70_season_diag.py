"""季节性 12 月彩票 · 前提诊断（不做模型 OOF——12 月标签本地不存在、测不出）。

只回答三个**构造前提**，前提不成立就不烧：
  P1  单月体量轨迹：2009-12（数据首月= 唯一在手的 12 月）是不是异常旺季？还是仅冷启动？
  P2  商品级季节倾斜：有没有一批『12 月才爆发』的商品，且不是全体商品的首月伪影？
      （对照：若零售商整体体量逐月增长，则任何在 Dec09 显高的商品反而是更强的季节信号。）
  P3  候选覆盖：这批高倾斜商品有没有进入候选集/逐用户列表、有没有 (u,i) 是 2009-12 已购的
      「年周期复购」候选——覆盖不到 = 赌注无从生效。

产物：outputs/candidate/season_diag_20260909.txt
用法：venv\\Scripts\\python.exe scripts\\70_season_diag.py
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import candidate as cd

LEADER_CLEAN = PROJECT_ROOT / "data" / "processed" / "leader_clean.csv"
CAND_INPUT = PROJECT_ROOT / "train_data" / "sample_candidates.csv"
OUT = PROJECT_ROOT / "outputs" / "candidate" / "season_diag_20260909.txt"

U, I, T, Q, D = "CustomerID", "StockCode", "InvoiceDate", "Quantity", "Description"
DEC = "2009-12"
REST = [f"2010-{m:02d}" for m in range(1, 11)]          # Jan..Oct 2010（10 个月）
MIN_MONTHLY = 2.0                                        # 商品月均 ≥2 units 才评倾斜（防零星噪声）


def main() -> None:
    clean = cd.load_clean_train(LEADER_CLEAN)
    clean = clean[clean[Q] > 0]
    clean["m"] = clean[T].dt.strftime("%Y-%m")
    lines = []

    # ---- P1 月度体量轨迹 -------------------------------------------------
    g = clean.groupby("m", sort=True)
    rows = pd.DataFrame({
        "rows": g.size(),                # 订单行数作体量指标
        "units": g[Q].sum(),
        "users": g[U].nunique(),
        "items": g[I].nunique(),
    })
    months = ["2009-12"] + REST
    rows = rows.reindex(months)
    print("== P1 月度体量轨迹（rows/units/users/items）==")
    print(rows.to_string(), flush=True)
    lines.append("== P1 月度体量轨迹 ==")
    lines.append(rows.to_string())
    dec_rows = rows.loc[DEC, "rows"]; dec_units = rows.loc[DEC, "units"]
    rest_mean_rows = rows.loc[REST, "rows"].mean(); rest_mean_units = rows.loc[REST, "units"].mean()
    ratio_rows = dec_rows / rest_mean_rows; ratio_units = dec_units / rest_mean_units
    ln = (f"\nDec2009 rows/月均值 = {ratio_rows:.2f}×；Dec2009 units/月均值 = {ratio_units:.2f}×"
          f"（>1 = 12 月真旺；<1 且随后增长 = 更偏冷启动增长，那 Dec 显高的商品更可能真季节）")
    print(ln, flush=True); lines.append(ln)

    # ---- P2 商品级季节倾斜 -------------------------------------------------
    dec_rows_df = clean[clean["m"] == DEC]
    rest_rows_df = clean[clean["m"].isin(REST)]
    item_dec = dec_rows_df.groupby(I, sort=True)[Q].sum()
    item_rest = rest_rows_df.groupby(I, sort=True)[Q].sum()
    it = pd.DataFrame({"dec": item_dec, "rest": item_rest}).fillna(0.0)
    it["total"] = it["dec"] + it["rest"]
    it["rest_mo"] = it["rest"] / 10.0
    it["lift"] = it["dec"] / np.maximum(it["rest_mo"], 1e-9)   # Dec 月量是平时月量的几倍
    it = it.sort_values("lift", ascending=False)
    stable = it[it["rest_mo"] >= MIN_MONTHLY]
    desc = rest_rows_df.groupby(I, sort=True)[D].first()
    stable_desc = stable.join(desc)

    # 高倾斜商品在 Dec 量里的占比 vs 平时量里的占比 —— 季节商品是否真的『只在 12 月放量』
    for thr in (1.5, 2.0, 3.0):
        hi = it[it["lift"] >= thr]
        n_hi = int(len(hi))
        dec_share = hi["dec"].sum() / max(1e-9, it["dec"].sum())
        rest_share = hi["rest"].sum() / max(1e-9, it["rest"].sum())
        ln = (f"\nP2 lift(Dec/平时月) ≥ {thr}: {n_hi} 商品；占 Dec 总 units {dec_share:.1%}"
              f"，占平时总 units {rest_share:.1%}（占比差距越大 = 季节结构越真）")
        print(ln, flush=True); lines.append(ln)

    top = stable_desc.head(30)
    print("\n== P2 顶倾斜商品（rest_mo≥%.0f, 按 lift）==" % MIN_MONTHLY)
    print(top[["dec", "rest_mo", "lift", D]].to_string(), flush=True)
    lines.append("\n== P2 顶倾斜商品 ==")
    lines.append(top[["dec", "rest_mo", "lift", D]].to_string())

    # ---- P2b 内容健全性：高 lift 商品到底是不是『圣诞/节令』商品？-----
    import re
    SKW = r"CHRISTMAS|XMAS|SANTA|MERRY|FESTIVE|STOCKING|ANGEL|SNOW|TREE|WRAP|DECORAT|CARD|NOVELTY"
    desc_all = clean.groupby(I, sort=True)[D].first().fillna("")
    stable2 = stable_desc.copy()
    stable2["is_skw"] = stable2[D].fillna("").str.upper().str.contains(SKW, regex=True)
    print("\n== P2b lift 分桶里的『节令关键词』商品占比（stable 商品 rest_mo≥%.0f）==" % MIN_MONTHLY)
    lines.append("\n== P2b lift 分桶节令占比 ==")
    buckets = [(0.0, 0.5, "<0.5(跌/停售)"), (0.5, 1.5, "0.5-1.5(平时水平)"),
               (1.5, 3.0, "1.5-3"), (3.0, 1e18, "≥3")]
    for lo, hi, lab in buckets:
        m = (stable2["lift"] >= lo) & (stable2["lift"] < hi)
        if m.sum() == 0:
            continue
        fr = stable2.loc[m, "is_skw"].mean()
        ln = f"  lift {lab}: n={int(m.sum()):4d}, 节令关键词占比 {fr:6.1%}"
        print(ln, flush=True); lines.append(ln)

    # 明确的圣诞商品（描述含 CHRISTMAS/XMAS）月度剖面：是集中 12 月还是全年摊？
    xmas = desc_all[desc_all.str.upper().str.contains(r"CHRISTMAS|XMAS", regex=True)].index
    xmas = [c for c in xmas if c in it.index and it.loc[c, "total"] >= 50]
    if xmas:
        prof = clean[clean[I].isin(xmas)].groupby("m", sort=True)[Q].sum().reindex(months).fillna(0)
        prof_frac = prof / max(1e-9, prof.sum())
        xmas_dec_share = prof_frac.loc[DEC]
        ln = (f"\n真圣诞描述商品（≥50 units, n={len(xmas)}）月度 units 分布：12月占 {xmas_dec_share:.1%}"
              f"，峰值月 = {prof.idxmax()}（若 12 月不是明显峰值 → 批发商圣诞货全年铺货，节令重排无从谈起）")
        print(ln, flush=True); lines.append(ln)
        print("  " + prof.round(0).to_string().replace("\n", "\n  "), flush=True)
        lines.append("  " + prof.round(0).to_string())

    # ---- P3 候选覆盖 ------------------------------------------------------
    cand = pd.read_csv(CAND_INPUT, dtype={"item_id": str})
    cand.columns = ["u", "item"]
    cand["u"] = pd.to_numeric(cand["u"]).astype("int64")
    cand["item"] = cand["item"].astype(str)
    cand = cand.merge(it.reset_index().rename(columns={I: "item"}), how="left", on="item")
    cand["rest_mo"] = cand["rest_mo"].fillna(0.0)
    cand["lift"] = cand["lift"].fillna(0.0)
    hi_mask = cand["lift"] >= 2.0
    ln = (f"\nP3 候选对 58,205：item 出现在 clean 的 {cand['total'].notna().sum():,} 对"
          f"；item lift≥2 的 {int(hi_mask.sum()):,} 对 ({hi_mask.mean():.1%})；"
          f"item lift≥3 的 {int((cand['lift']>=3.0).sum()):,} 对")
    print(ln, flush=True); lines.append(ln)

    # 逐用户列表里高倾斜商品的占比
    per_u = cand.assign(hi=hi_mask).groupby("u", sort=True)["hi"].agg(["sum", "count"])
    frac_hi = per_u["sum"] / per_u["count"]
    ln = (f"逐用户候选列表含 lift≥2 商品的占比：均值 {frac_hi.mean():.2%}，"
          f"至少 1 个的用户 {int((frac_hi > 0).sum())}/{len(per_u)} 人")
    print(ln, flush=True); lines.append(ln)

    # 年周期复购候选：(u,i) 2009-12 买过 且 在候选集
    dec_pairs = set(zip(dec_rows_df[U].to_numpy(), dec_rows_df[I].astype(str).to_numpy()))
    c_arr = np.array([(a, b) in dec_pairs for a, b in zip(cand["u"].to_numpy(), cand["item"].to_numpy())],
                     dtype=bool)
    ln = (f"(u,i) 在 2009-12 买过（年周期复购候选）：{int(c_arr.sum())} 对；"
          f"涉及用户 {int(cand.loc[c_arr, 'u'].nunique())} 人")
    print(ln, flush=True); lines.append(ln)

    top_cand = cand[cand["lift"] >= 3.0].sort_values("lift", ascending=False)
    if len(top_cand):
        td = top_cand.drop_duplicates("item").head(25)[["item", "dec", "rest_mo", "lift"]].copy()
        td["desc"] = td["item"].map(desc_all)
        print("\n== P3 候选里 lift≥3 的商品样张 ==")
        print(td.to_string(), flush=True)
        lines.append("\n== P3 候选里 lift≥3 的商品样张 ==")
        lines.append(td.to_string())

    OUT.write_text("\n".join(map(str, lines)) + "\n", encoding="utf-8")
    print(f"\n[done] 摘要 → {OUT.name}", flush=True)


if __name__ == "__main__":
    main()
