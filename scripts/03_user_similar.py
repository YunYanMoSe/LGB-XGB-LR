"""CLI：输入一个用户ID，输出其相似用户推荐列表（余弦相似度 Top-K）。

用法:
    python scripts/03_user_similar.py 17850            # 指定用户，默认 Top10
    python scripts/03_user_similar.py 17850 --top 5    # 只要前5个
    python scripts/03_user_similar.py 17850 --binary   # 按「是否购买」而非数量算相似度
    python scripts/03_user_similar.py                  # 交互式输入用户ID
"""

import argparse
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.user_similarity import (  # noqa: E402
    find_similar_users,
    load_purchase_matrix,
    n_items_of,
    row_index_of,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="基于用户购买行为的相似用户推荐（余弦相似度）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python scripts/03_user_similar.py 17850\n"
            "  python scripts/03_user_similar.py 17850 --top 5 --binary"
        ),
    )
    p.add_argument("user_id", nargs="?", type=int, help="目标用户ID（不传则交互输入）")
    p.add_argument("--top", type=int, default=10, metavar="K",
                   help="返回相似用户个数（默认 10）")
    p.add_argument("--binary", action="store_true",
                   help="按是否购买(0/1)计算余弦，而非按购买数量加权")
    return p


def main() -> int:
    args = build_parser().parse_args()

    user_id = args.user_id
    if user_id is None:  # 交互式输入
        try:
            user_id = int(input("请输入用户 ID: ").strip())
        except (ValueError, EOFError):
            print("错误: 用户 ID 必须为整数。")
            return 2

    mat, user_ids, _ = load_purchase_matrix(binary=args.binary)

    pos = row_index_of(mat, user_ids, user_id)
    if pos < 0:
        samples = ", ".join(str(u) for u in user_ids[:5])
        print(f"错误: 数据中不存在用户 ID = {user_id}。")
        print(f"当前共 {len(user_ids):,} 位用户，可尝试: {samples}, ...")
        return 1

    results = find_similar_users(mat, user_ids, user_id, top_k=args.top)

    metric = "是否购买(0/1)" if args.binary else "购买数量"
    print("=" * 46)
    print(f" 相似用户推荐 · 目标用户 {user_id}")
    print("=" * 46)
    print(f"\n 目标用户购买过 {n_items_of(mat, user_ids, user_id)} 种商品")
    print(f" 按「{metric}」余弦相似度，找到 {len(results)} 位相似用户 (>0)：\n")

    if not results:
        print(" 未找到相似用户：该用户购买的商品与其他用户无重叠。")
        return 0

    print(f"  {'排名':>4}  {'相似用户':>10}  {'相似度':>8}")
    print(f"  {'────':>4}  {'────────':>10}  {'──────':>8}")
    for i, r in enumerate(results, 1):
        print(f"  {i:>4}  {r['user_id']:>10}  {r['similarity']:>8.4f}")

    print("\n 说明: 相似度越接近 1 表示购买行为越一致，0 表示无共同购买商品。")
    print("=" * 46)
    return 0


if __name__ == "__main__":
    sys.exit(main())
