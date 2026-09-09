"""双人实验·组员路线 IO：anchor OOF、交易、Nov 候选/提交的加载（只读，绝不覆盖产物）。

月份口径：snapshot_month 形如 2010-08 → ym '2010-08'，cutoff = 该月 1 日 00:00，
特征只用交易时间 < cutoff 的行（时间安全）。leader_clean 为候选线唯一交易源。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("BASKET_ROOT", Path(__file__).resolve().parents[2]))
# ROOT = d:\0000\recommender_project（src/basket_experiment -> src -> 根）

ANCHOR_OOF = ROOT / "data" / "candidate_aligned_oof_v15.csv"
LEADER_CLEAN = ROOT / "data" / "processed" / "leader_clean.csv"
SAMPLE_CANDIDATES = ROOT / "train_data" / "sample_candidates.csv"
V15_NOV_SUB = ROOT / "data" / "submission_traincsv_v15_activity_moe_blend.csv"
OUT_DIR = ROOT / "outputs" / "experiment_basket_hypergraph"

# 统一协议月份：08 排错 / 09 选型 / 10 最近月确认
YM_ALL = ["2010-08", "2010-09", "2010-10"]
YM_TRAIN = ["2010-08", "2010-09"]
YM_CONFIRM = ["2010-10"]

_COL_USER = "CustomerID"
_COL_ITEM = "StockCode"
_COL_TIME = "InvoiceDate"
_COL_INVOICE = "InvoiceNo"
_COL_QTY = "Quantity"


def cutoff_of(ym: str) -> pd.Timestamp:
    return pd.Timestamp(ym + "-01")


def load_anchor_oof() -> pd.DataFrame:
    """anchor OOF（146,530 行，ym 覆盖 2010-08/09/10）。只读。"""
    df = pd.read_csv(ANCHOR_OOF)
    df["ym"] = pd.to_datetime(df["snapshot_month"]).dt.strftime("%Y-%m")
    df["user_id"] = df["user_id"].astype("int64")
    df["item_id"] = df["item_id"].astype(str)
    return df


def load_clean_transactions() -> pd.DataFrame:
    """候选线唯一交易源 leader_clean.csv。

    返回列：InvoiceNo, StockCode, InvoiceDate(datetime), CustomerID(int64, 去空)。
    不做进一步清洗（篮去重、服务码等由调用方在篮构建时处理）。
    """
    df = pd.read_csv(LEADER_CLEAN)
    df[_COL_TIME] = pd.to_datetime(df[_COL_TIME])
    df[_COL_USER] = pd.to_numeric(df[_COL_USER], errors="coerce")
    df = df[df[_COL_USER].notna()].copy()
    df[_COL_USER] = df[_COL_USER].astype("int64")
    df[_COL_ITEM] = df[_COL_ITEM].astype(str)
    return df


def load_nov_candidates() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE_CANDIDATES)
    df["user_id"] = df["user_id"].astype("int64")
    df["item_id"] = df["item_id"].astype(str)
    return df


def load_v15_nov_submission() -> pd.DataFrame:
    """组长 Nov 提交 (58,205×3)，作 Nov 候选行的 anchor_score。"""
    df = pd.read_csv(V15_NOV_SUB)
    df["user_id"] = df["user_id"].astype("int64")
    df["item_id"] = df["item_id"].astype(str)
    return df
