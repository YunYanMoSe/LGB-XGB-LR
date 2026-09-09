"""train.csv 独立清洗（候选集打分任务的原料准备）。

数据：data/raw/train.csv（utf-8，8 列：InvoiceNo,StockCode,Description,Quantity,
InvoiceDate,UnitPrice,CustomerID,Country）。

清洗口径 =「标准+更干净」：
    ① 删除完全重复行（字段全同）；
    ② 删除 StockCode ∈ 服务码黑名单的行（精确匹配，不用描述关键词删——防误伤真商品，
       如 GOTHIC CARRIAGE LANTERN；描述只做日志审计）。

产物（全部为新文件，不碰旧数据/旧答辩产物）：
    data/processed/clean_train.csv       8 列同序清洗表
    outputs/candidate/clean_train_report.txt   清洗报告（各步前后行数、删除码计数）

运行：venv\\Scripts\\python.exe scripts\\18_clean_train.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src import candidate as cand

RAW = PROJECT_ROOT / "data" / "raw" / "train.csv"
OUT = cand.CLEAN_TRAIN
REPORT = PROJECT_ROOT / "outputs" / "candidate" / "clean_train_report.txt"

COL8 = ["InvoiceNo", "StockCode", "Description", "Quantity",
        "InvoiceDate", "UnitPrice", "CustomerID", "Country"]


def main() -> None:
    t0 = time.time()
    df = pd.read_csv(RAW, encoding="utf-8")
    n0 = len(df)
    assert list(df.columns) == COL8, f"train.csv 列序与预期不符: {list(df.columns)}"

    # ① 完全重复行
    df = df.drop_duplicates().copy()
    n_dup = n0 - len(df)

    # ② 服务码黑名单（精确匹配）
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    before_service = len(df)
    removed_mask = df["StockCode"].isin(cand.SERVICE_CODES)
    df_clean = df[~removed_mask].copy()
    removed = df[removed_mask]
    n_svc = len(removed)
    svc_codes = sorted(removed["StockCode"].unique())

    # 描述关键词仅日志审计（不据此删除）
    audit = removed["Description"].fillna("").astype(str).str.contains(
        "bank charge|adjust|postage|test", case=False, regex=True).sum()

    # 类型收口：CustomerID int、StockCode str
    df_clean["CustomerID"] = pd.to_numeric(df_clean["CustomerID"], errors="coerce").astype("int64")
    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
    df_clean = df_clean[COL8]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUT, index=False, encoding="utf-8")

    lines = [
        "train.csv 清洗报告（候选集打分任务原料）",
        "=" * 56,
        f"数据源    : {RAW.name}（utf-8）",
        f"清洗前行数 : {n0:,}",
        f"① 完全重复行删除 : {n_dup:,}（字段全同）",
        f"② 服务码黑名单删除: {n_svc:,} 行 / {len(svc_codes)} 种码",
        f"   删除的服务码  : {', '.join(svc_codes)}",
        f"   其中描述含 audit 关键词行 : {audit:,}（仅审计，非删除依据）",
        f"清洗后行数 : {len(df_clean):,}",
        f"剩余字段   : {', '.join(df_clean.columns)}",
        f"时间范围   : {df_clean['InvoiceDate'].min():%Y-%m-%d} ~ {df_clean['InvoiceDate'].max():%Y-%m-%d}",
        f"剩余用户数 : {df_clean['CustomerID'].nunique():,}",
        f"剩余商品数 : {df_clean['StockCode'].nunique():,}",
        f"剩余订单数 : {df_clean['InvoiceNo'].nunique():,}",
        f"校验: {len(df_clean):,} = {n0:,} - {n_dup:,} - {n_svc:,} "
        f"→ {len(df_clean) == n0 - n_dup - n_svc}",
        f"校验: CustomerID 无缺失 = {not df_clean['CustomerID'].isna().any()}",
        f"校验: Quantity>0 = {(df_clean['Quantity'] > 0).all()}  UnitPrice>0 = {(df_clean['UnitPrice'] > 0).all()}",
        f"校验: 已删除服务码在 clean_train 残留 = {df_clean['StockCode'].isin(cand.SERVICE_CODES).any()}",
        f"输出     : {OUT}",
        f"耗时     : {time.time() - t0:.1f}s",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
