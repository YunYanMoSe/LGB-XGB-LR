"""阶段六素材打包（薄封装）：md → docx。

渲染逻辑同 scripts/09（src/md_to_docx.render）。图片在 ../chain/ 下，命中的 key
"flask_" 会以 5.6in 宽排版（竖长演示页适度收窄）。

运行：venv\\Scripts\\python.exe scripts\\14_build_chain_docx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.md_to_docx import render

MD = PROJECT_ROOT / "outputs" / "chain_report" / "阶段六_链路素材.md"
OUT = PROJECT_ROOT / "outputs" / "阶段六_召回精排_报告素材.docx"


def main() -> None:
    render(MD, OUT)
    print(f"已写出 {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
