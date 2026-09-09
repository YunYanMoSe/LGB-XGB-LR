"""阶段七素材打包（薄封装）：素材 md → docx。

渲染逻辑同 scripts/14（src/md_to_docx.render）。宽表很大不嵌入 Word，
图片 feature_importance.png 与 md 同目录，默认宽度 6.3in 排版。

运行：venv\\Scripts\\python.exe scripts\\17_build_feature_docx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.md_to_docx import render

MD = PROJECT_ROOT / "outputs" / "feature_stage" / "阶段七_特征工程与精排素材.md"
OUT = PROJECT_ROOT / "outputs" / "阶段七_特征工程与精排_报告素材.docx"


def main() -> None:
    render(MD, OUT)
    print(f"已写出 {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
