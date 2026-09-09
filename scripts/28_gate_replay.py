"""候选集打分任务 ⑩（v9 对标线）：门禁预检 —— 复现组长黄俊霖《候选对齐时序回放》表 VI。

目标：在动手建特征前，先用最便宜的统计把「回放语义」钉死：
    * 用组长口径清洗 raw（去掉取消/退货、非正价、非法量、缺用户/主键、无效日期、整行重复；
      但【保留服务码行】——组长 clean=327,827 恰为 332,848-5,021，说明他未删服务码）；
    * 逐月 t∈{2010-06..2010-10} 重放固定候选：
        H<t = 交易日期 < t月1日；
        C_t = 候选行中「用户 ∈ H<t 且 商品 ∈ H<t 目录」的子集（用户/商品在 t 截点均已知）；
        标签 = (u, i) 在【t 月当月】真实购买（Pt）；
        新组合 = 标签=1 且 (u,i) 在 H<t 从未购买过。
    * 对比组长表 VI：
        月份       样本     正例   正例率   新组合
        2010-06  38,005   1,562  4.110%   432
        2010-07  41,200   1,625  3.944%   426
        2010-08  45,399   1,627  3.584%   454
        2010-09  48,097   2,313  4.809%   779
        2010-10  53,034   2,728  5.144%  1,044
        合计     225,735  9,855  4.366%  3,135

若逐月样本数与正例数与上表一致（或极小偏差），回放语义即被复现，
后续特征构建/OOF/提交才有意义。本脚本不产任何模型/不覆盖旧产物。

用法：venv\\Scripts\\python.exe scripts\\28_gate_replay.py
产物：outputs/candidate/replay_gate.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw" / "train.csv"
CAND = PROJECT_ROOT / "train_data" / "sample_candidates.csv"
OUT = PROJECT_ROOT / "outputs" / "candidate" / "replay_gate.txt"

USER, ITEM, TIME = "CustomerID", "StockCode", "InvoiceDate"


def clean_leaders_way(raw: pd.DataFrame) -> pd.DataFrame:
    """组长口径清洗：删取消/退货、非正价、非法量、缺用户/缺主键、无效日期、整行重复；保留服务码。"""
    df = raw.copy()
    n0 = len(df)
    df = df.dropna(subset=[USER])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C", na=False)]      # 取消/退货单
    df = df[df["Quantity"] > 0]                                              # 非法数量/退货
    df = df[df["UnitPrice"] > 0]                                             # 非正价
    df = df[df[ITEM].astype(str).str.strip() != ""]                          # 缺主键
    df = df[pd.to_datetime(df[TIME], errors="coerce").notna()]               # 无效日期
    df = df.drop_duplicates()                                                # 整行重复
    df = df.reset_index(drop=True)
    return df


def main() -> None:
    lines: list[str] = []
    raw = pd.read_csv(RAW)
    clean = clean_leaders_way(raw)
    clean[TIME] = pd.to_datetime(clean[TIME])
    clean[USER] = pd.to_numeric(clean[USER], errors="coerce").astype("int64")
    clean[ITEM] = clean[ITEM].astype(str)

    lines.append(f"raw={len(raw):,}  leader_clean={len(clean):,}  "
                 f"(报告 332,848→327,827; 差 {332848-327827:,})")

    cand = pd.read_csv(CAND, dtype={c: str for c in ["user_id", "item_id"]})
    cand.columns = ["user_id", "item_id"]
    cand["user_id"] = pd.to_numeric(cand["user_id"]).astype("int64")
    cand["item_id"] = cand["item_id"].astype(str)

    # 用户/商品目录（按整段 train；用于判定候选行是否"在 H<t 已知"）
    users_known_ever = set(int(x) for x in clean[USER].unique())
    items_known_ever = set(str(x) for x in clean[ITEM].unique())

    months = [pd.Timestamp(f"2010-{m:02d}-01") for m in range(6, 11)]
    rows = []
    for m in months:
        nxt = (m + pd.offsets.MonthBegin(1))
        fp = clean[clean[TIME] < m]
        lp = clean[(clean[TIME] >= m) & (clean[TIME] < nxt)]
        fp_users = set(int(x) for x in fp[USER].unique())
        fp_items = set(str(x) for x in fp[ITEM].unique())
        fp_bought: dict[tuple[int, str], bool] = set()
        for u, i in zip(fp[USER], fp[ITEM]):
            fp_bought.add((int(u), str(i)))
        lp_bought: set[tuple[int, str]] = set()
        for u, i in zip(lp[USER], lp[ITEM]):
            lp_bought.add((int(u), str(i)))

        # C_t：候选行中用户与商品在截点 t 均已知（用户、商品都出现在 fp）
        sub = cand[(cand["user_id"].isin(fp_users)) & (cand["item_id"].isin(fp_items))]
        n = len(sub)
        pos = 0
        newc = 0
        for u, i in zip(sub["user_id"], sub["item_id"]):
            key = (int(u), str(i))
            if key in lp_bought:
                pos += 1
                if key not in fp_bought:
                    newc += 1
        rows.append({"month": str(m.date()), "samples": n, "pos": pos,
                     "rate": pos / max(1, n), "new": newc})
        lines.append(f"{str(m.date())}  samples={n:,}  pos={pos:,}  "
                     f"rate={pos/max(1,n):.4%}  new_combo={newc:,}")

    tot_s = sum(r["samples"] for r in rows)
    tot_p = sum(r["pos"] for r in rows)
    lines.append(f"TOTAL  samples={tot_s:,}  pos={tot_p:,}  rate={tot_p/max(1,tot_s):.4%}  "
                 f"new={sum(r['new'] for r in rows):,}")

    # 用户/商品全谱核对
    lines.append(f"users_known_ever={len(users_known_ever):,} items_known_ever={len(items_known_ever):,} "
                 f"cand_users={cand['user_id'].nunique():,} cand_items={cand['item_id'].nunique():,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
