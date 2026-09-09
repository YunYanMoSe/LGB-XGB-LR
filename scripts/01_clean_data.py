"""阶段一入口：数据加载 → 原始探查 → 三级清洗 → 留存报告 → Top10 验证。"""

import sys
from pathlib import Path

# Windows 控制台默认按 GBK 输出，强制改用 UTF-8，避免中文乱码
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 将项目根目录加入 Python 路径，确保 config / src 模块可导入
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PATHS, set_seed
from src.data_cleaning import (
    load_transactions,
    profile_data,
    save_profile_report,
    clean_transactions,
    save_cleaning_report,
    verify_top_items_with_products,
)


def main():
    # 1. 固定随机种子（便于复现）
    set_seed()

    print("=" * 60)
    print("阶段一：原始数据体检 + 三级清洗 (Stage 1)")
    print("=" * 60)

    # 2. 读取原始交易数据（已做 ID 字符串转换）
    print("\n[1] 正在加载原始交易数据 ...")
    df_trans = load_transactions()
    print(f"    加载完成，共 {len(df_trans):,} 行记录。")

    # 3. 原始数据探查
    print("\n[2] 正在计算原始统计指标 ...")
    report = profile_data(df_trans)
    profile_file = PATHS["outputs_dir"] / "raw_data_profile.txt"
    save_profile_report(report, profile_file)
    print(f"    体检报告已保存至: {profile_file}")

    print("\n" + "-" * 40)
    print("原始数据核心指标速览:")
    print(f"  总行数         : {report['total_rows']:,}")
    print(f"  用户数         : {report['unique_users']:,}")
    print(f"  商品数         : {report['unique_items']:,}")
    print(f"  CustomerID缺失 : {report['missing_customer_ratio']:.4%}")
    print(f"  Top10占比      : {report['top10_share']:.4%}")
    print("-" * 40 + "\n")

    # 4. 三级清洗漏斗
    print("[3] 正在执行三级清洗漏斗 ...")
    clean_df, stage_logs = clean_transactions(df_trans)

    for log in stage_logs:
        print(f"    {log['stage']:32s} "
              f"前={log['rows_before']:>8,}  删={log['rows_removed']:>8,}  "
              f"后={log['rows_after']:>8,}  累计留存={log['cum_retention']:.4%}")
    print(f"    清洗后共 {len(clean_df):,} 行，总体留存率 {len(clean_df) / len(df_trans):.4%}")

    # 5. 输出清洗结果与留存报告
    clean_file = PATHS["clean_transactions"]
    clean_file.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(clean_file, index=False)
    print(f"    [4] 清洗结果已保存至: {clean_file}")

    report_file = PATHS["outputs_dir"] / "cleaning_report.txt"
    save_cleaning_report(stage_logs, report_file)
    print(f"    [5] 留存报告已保存至: {report_file}")

    # 6. 验证：关联 products.csv 抽查 Top10 商品名称
    print("\n[6] 正在关联商品主数据，验证 Top10 商品名称 ...")
    top10 = verify_top_items_with_products(clean_df)
    print(top10.to_string(index=False))

    verify_file = PATHS["outputs_dir"] / "top10_verification.csv"
    top10.to_csv(verify_file, index=False, encoding="utf-8-sig")
    print(f"    Top10 验证结果已保存至: {verify_file}")

    print("\n[阶段一] 全部完成！验收文件：")
    print(f"  - {profile_file}")
    print(f"  - {report_file}")
    print(f"  - {clean_file}")
    print(f"  - {verify_file}")


if __name__ == "__main__":
    main()
