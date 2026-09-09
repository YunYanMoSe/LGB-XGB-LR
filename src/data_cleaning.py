"""数据清洗与原始数据探查模块。"""

import pandas as pd
from pathlib import Path
from config.settings import PATHS, COLUMNS, ENCODINGS


def load_transactions() -> pd.DataFrame:
    """
    从 data/raw 读取原始交易 CSV，并做基础类型转换。
    
    关键转换：
        - InvoiceNo（发票号）：转为字符串（处理 C 开头取消单）
        - StockCode（商品ID）：转为字符串（处理 85123A 等字母商品码）
    
    Returns:
        pd.DataFrame: 原始交易表格
    """
    df = pd.read_csv(PATHS["raw_transactions"], encoding=ENCODINGS["transactions"])

    # 发票号与商品ID强制转为字符串
    df[COLUMNS["invoice"]] = df[COLUMNS["invoice"]].astype(str)
    df[COLUMNS["item"]] = df[COLUMNS["item"]].astype(str)

    return df


def profile_data(df: pd.DataFrame) -> dict:
    """
    对原始交易数据进行统计探查。
    
    指标：
        - 总交互行数
        - 唯一用户数
        - 唯一商品数
        - CustomerID 缺失比例
        - Top10 商品交互次数占比（按行数计）
    
    Args:
        df: 原始交易 DataFrame
        
    Returns:
        dict: 包含各项统计指标的字典
    """
    user_col = COLUMNS["user"]
    item_col = COLUMNS["item"]
    
    total_rows = len(df)
    unique_users = df[user_col].nunique(dropna=False)  # 包含 NaN 作为一类
    unique_items = df[item_col].nunique()
    missing_ratio = df[user_col].isna().mean()
    
    # Top10 热门商品（按交互行数统计）
    top_items_counts = df[item_col].value_counts().head(10)
    top10_share = top_items_counts.sum() / total_rows
    
    return {
        "total_rows": total_rows,
        "unique_users": unique_users,
        "unique_items": unique_items,
        "missing_customer_ratio": missing_ratio,
        "top10_items": top_items_counts.to_dict(),
        "top10_share": top10_share,
    }


def save_profile_report(report: dict, output_path: Path) -> None:
    """
    将探查报告保存为文本文件。
    
    Args:
        report: profile_data 返回的字典
        output_path: 输出文件路径（如 outputs/raw_data_profile.txt）
    """
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("         原始交易数据体检报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"【总体规模】\n")
        f.write(f"  总交互行数 (Rows)         : {report['total_rows']:,}\n")
        f.write(f"  唯一用户数 (Users)         : {report['unique_users']:,}\n")
        f.write(f"  唯一商品数 (Items)         : {report['unique_items']:,}\n")
        f.write(f"  CustomerID 缺失比例        : {report['missing_customer_ratio']:.4%}\n\n")
        
        f.write(f"【交互集中度】\n")
        f.write(f"  Top10 商品交互占比         : {report['top10_share']:.4%}\n\n")
        
        f.write(f"【Top10 热门商品】（按交互行数）\n")
        for item, cnt in report['top10_items'].items():
            f.write(f"  {item:12s} : {cnt:>8,} 次\n")
        
        f.write("\n" + "=" * 60 + "\n")


def load_products() -> pd.DataFrame:
    """
    读取商品主数据表，并打印基本信息（用于阶段一验证）。

    Returns:
        pd.DataFrame: 商品主数据
    """
    df = pd.read_csv(PATHS["raw_products"], encoding=ENCODINGS["products"])
    # 商品ID 同样强制转字符串，保证与交易表 StockCode 可正常关联
    df[COLUMNS["item"]] = df[COLUMNS["item"]].astype(str)
    print("\n[商品主数据] 加载成功！")
    print(f"  形状: {df.shape}")
    print(f"  列名: {list(df.columns)}")
    print("\n前 5 行预览:")
    print(df.head())
    return df


# ---------------------------------------------------------------------------
# 三级清洗漏斗
# ---------------------------------------------------------------------------

def clean_stage1_drop_missing_customer(df: pd.DataFrame) -> pd.DataFrame:
    """一级清洗：删除 CustomerID 缺失的行（无法归属到用户的交互无推荐价值）。"""
    return df[df[COLUMNS["user"]].notna()].copy()


def clean_stage2_drop_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    """
    二级清洗：删除取消/退货记录。

    规则：InvoiceNo 以 C 开头（取消单），或 Quantity <= 0（退货/负数量）。
    二者满足其一即整行删除。
    """
    invoice = df[COLUMNS["invoice"]].astype(str).str.strip().str.upper()
    is_cancel = invoice.str.startswith("C")
    is_bad_qty = df[COLUMNS["qty"]] <= 0
    return df[~(is_cancel | is_bad_qty)].copy()


def clean_stage3_drop_invalid_price(df: pd.DataFrame) -> pd.DataFrame:
    """三级清洗：删除 UnitPrice <= 0 的行（价格无效，无法参与金额计算）。"""
    return df[df[COLUMNS["price"]] > 0].copy()


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    依次执行三级清洗，并记录每级的行数/比例。

    Args:
        df: 原始交易 DataFrame（建议来自 load_transactions）

    Returns:
        (clean_df, stage_logs): 清洗后的 DataFrame 与每级留存日志。
        stage_logs 每项包含：
            stage / rows_before / rows_removed / rows_after /
            removed_ratio（本级删除比例）/ cum_retention（累计留存率）
    """
    total = len(df)
    stages = [
        ("stage1_drop_missing_customer", clean_stage1_drop_missing_customer),
        ("stage2_drop_cancellations", clean_stage2_drop_cancellations),
        ("stage3_drop_invalid_price", clean_stage3_drop_invalid_price),
    ]

    current = df
    logs = []
    for name, fn in stages:
        rows_before = len(current)
        current = fn(current)
        rows_after = len(current)
        logs.append({
            "stage": name,
            "rows_before": rows_before,
            "rows_removed": rows_before - rows_after,
            "rows_after": rows_after,
            "removed_ratio": (rows_before - rows_after) / rows_before,
            "cum_retention": rows_after / total,
        })

    return current.reset_index(drop=True), logs


def save_cleaning_report(report: dict, output_path: Path) -> None:
    """
    将三级清洗留存报告保存为文本文件。

    Args:
        report: clean_transactions 的 stage_logs
        output_path: 输出文件路径（如 outputs/cleaning_report.txt）
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = report[0]["rows_before"]
    final_rows = report[-1]["rows_after"]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("         三级清洗漏斗留存报告\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"原始数据行数 (Raw rows)      : {total:,}\n\n")

        f.write("【清洗规则】\n")
        f.write("  一级：删除 CustomerID 缺失的行\n")
        f.write("  二级：删除取消/退货（InvoiceNo 以 C 开头，或 Quantity <= 0）\n")
        f.write("  三级：删除 UnitPrice <= 0 的行\n\n")

        f.write("【每级留存明细】\n")
        for log in report:
            f.write(f"  Stage : {log['stage']}\n")
            f.write(f"    清洗前行数 (Before)   : {log['rows_before']:>10,}\n")
            f.write(f"    删除行数 (Removed)    : {log['rows_removed']:>10,}\n")
            f.write(f"    保留行数 (After)      : {log['rows_after']:>10,}\n")
            f.write(f"    本级删除比例          : {log['removed_ratio']:.4%}\n")
            f.write(f"    累计留存率 (Cum.)     : {log['cum_retention']:.4%}\n\n")

        f.write("-" * 60 + "\n")
        f.write(f"最终保留行数 (Clean rows)    : {final_rows:,}\n")
        f.write(f"总体留存率 (Overall retain.) : {final_rows / total:.4%}\n")
        f.write("=" * 60 + "\n")


def verify_top_items_with_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    关联商品主数据，抽查清洗后 Top10 商品名称（阶段一验证）。

    Args:
        df: 清洗后的交易 DataFrame

    Returns:
        pd.DataFrame: Top10 商品及其名称/分类，缺失名称的标记为 <未匹配>
    """
    products = load_products()
    prod_cols = [COLUMNS["item"], COLUMNS["desc"], "Chinese_Description", "Product_Category"]

    # products.csv 中同一 StockCode 可能存在多行不同描述，按商品ID去重取第一条
    products_dedup = products[prod_cols].drop_duplicates(subset=COLUMNS["item"])

    top10 = (
        df[COLUMNS["item"]]
        .value_counts()
        .rename("count")
        .rename_axis(COLUMNS["item"])
        .reset_index()
        .head(10)
    )
    merged = top10.merge(products_dedup, on=COLUMNS["item"], how="left")
    n_matched = merged[COLUMNS["desc"]].notna().sum()
    merged[COLUMNS["desc"]] = merged[COLUMNS["desc"]].fillna("<未匹配>")

    print(f"    Top10 中 {n_matched}/10 个商品在 products.csv 中匹配到名称。")
    return merged