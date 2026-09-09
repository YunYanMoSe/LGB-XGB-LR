"""全局配置：路径、随机种子、常用超参。

学生可改文件名/超参，但请保持 SEED=42，便于组间对照。
"""
from __future__ import annotations

from pathlib import Path

# 项目根目录（无论从哪启动脚本，都能定位 data/ models/）
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEED = 42

PATHS = {
    "raw_transactions": PROJECT_ROOT / "data" / "raw" / "transactions.csv",
    "raw_products": PROJECT_ROOT / "data" / "raw" / "products.csv",
    "clean_transactions": PROJECT_ROOT / "data" / "processed" / "clean_transactions.csv",
    "user_sequences": PROJECT_ROOT / "data" / "processed" / "user_sequences.csv",
    "recall_set": PROJECT_ROOT / "data" / "processed" / "recall_set.csv",
    "models_dir": PROJECT_ROOT / "models",
    "vector_index_dir": PROJECT_ROOT / "models" / "vector_index",
    "item2vec_model": PROJECT_ROOT / "models" / "item2vec.model",
    "outputs_dir": PROJECT_ROOT / "outputs",
}

# 原始数据文件编码（transactions 含 £ 等字符，实测为 Latin-1；products 为带 BOM 的 UTF-8）
ENCODINGS = {
    "transactions": "latin-1",
    "products": "utf-8-sig",
}

# 字段约定（若你改了原始列名，只改这里）
COLUMNS = {
    "invoice": "InvoiceNo",
    "item": "StockCode",
    "user": "CustomerID",
    "time": "InvoiceDate",
    "qty": "Quantity",
    "price": "UnitPrice",
    "country": "Country",
    "desc": "Description",
}

HYPERPARAMS = {
    "item2vec": {
        # 阶段五实际训练使用（scripts/06_train_item2vec.py）。CBOW 与 Skip-gram
        # 除 sg 不同外共用同一组超参；sg=0 为 CBOW、sg=1 为 Skip-gram。
        "vector_size": 128,
        "window": 8,
        "min_count": 5,
        "negative": 5,
        "sample": 1e-3,
        "epochs": 12,
        "workers": 4,
        "seed": 42,
        "sg": 1,  # 说明性默认值：实际脚本里两种方法分别用 sg=0 / sg=1 各训一个
    },
    "recall": {
        "final_k": 50,
        "quotas": {"popular": 10, "similar": 20, "user_vec": 20},
    },
    # 5 路召回融合（scripts/04_build_recall.py）。配额总和要求=final_k。
    # 路序即融合优先级；neighbor_k = 第 2/4/5 路「每个历史/最近商品取 Top-K」。
    "recall5": {
        "final_k": 50,
        "hot_top_n": 100,  # 「热门占比」口径：全局销量 Top-N
        "route_order": ["popular", "similar", "user_vec", "itemcf", "recent"],
        "quotas": {"popular": 10, "similar": 10, "user_vec": 15, "itemcf": 10, "recent": 5},
        "neighbor_k": 5,
        # 对照实验：把热门配额 10→20（从 user_vec 借 10，总配额仍为 50）
        "alt_note": "调大热门配额：popular 10→20（自 user_vec 借 10，总量仍 50）",
        "alt_quotas": {"popular": 20, "similar": 10, "user_vec": 5, "itemcf": 10, "recent": 5},
    },
    "rank": {
        "topn": 10,
    },
}


def set_seed(seed: int = SEED) -> None:
    """各脚本开头调用，尽量保证可复现。"""
    import os
    import random

    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass
