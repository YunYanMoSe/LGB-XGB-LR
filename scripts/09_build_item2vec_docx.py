"""阶段五素材打包（薄封装）：md → docx。

渲染逻辑已抽取到 src/md_to_docx.py，本脚本只负责路径：
    outputs/item2vec/阶段五_Item2Vec素材.md → outputs/阶段五_Item2Vec_报告素材.docx
损失曲线图按默认 6.3in 宽；网页截图命中 key "web_similar" → 5.0in 宽。

运行：venv\\Scripts\\python.exe scripts\\09_build_item2vec_docx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.md_to_docx import render

MD = PROJECT_ROOT / "outputs" / "item2vec" / "阶段五_Item2Vec素材.md"
OUT = PROJECT_ROOT / "outputs" / "阶段五_Item2Vec_报告素材.docx"


def main() -> None:
    render(MD, OUT)
    print(f"已写出 {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
