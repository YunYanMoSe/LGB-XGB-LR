"""Markdown → Word(.docx) 轻量渲染器（报告素材通用）。

把内部约定格式的 md 渲染成 docx：
    标题 #~####、普通段落、引用 >、无序列表 -、有序列表(保留原文数字)、
    管道表格、行内 **加粗** 与 `等宽`、图片 ![](相对 md 的路径)。

图片按 rel 路径里命中的 key 选宽度（KEY_WIDTH），默认 default_width。
中文字体走 Word eastAsia（微软雅黑），Latin 用 Calibri，避免提交后缺字。

用法：
    from src.md_to_docx import render
    render(md_path, out_path)                      # 用默认 KEY_WIDTH
    render(md_path, out_path, KEY_WIDTH=[("loss", 6.3)])
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

# (rel 路径片段, 宽度 in)。先命中的生效；适合"竖长截图窄一点、横长图宽一点"。
DEFAULT_KEY_WIDTH = [("web_similar", 5.0), ("flask_", 5.6)]
DEFAULT_WIDTH_IN = 6.3

BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
CODE_RE = re.compile(r"`([^`]+)`")
IMG_RE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)$")
PIPE = re.compile(r"^\|(.+)\|$")
SEP_CELL = re.compile(r"^:?-{2,}:?$")

H_COLORS = ["1F3864", "2F5496", "3B78B4", "2A78D6"]


def _style_rfonts(style_or_rpr, *, ascii_: str, east_asia: str) -> None:
    if hasattr(style_or_rpr, "element"):
        rpr = style_or_rpr.element.get_or_add_rPr()
    else:
        rpr = style_or_rpr
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs"):
        rfonts.set(qn(attr), ascii_)
    rfonts.set(qn("w:eastAsia"), east_asia)


def _add_runs(paragraph, text: str, base_bold: bool = False) -> None:
    segs: list[tuple[int, str]] = [(0, text)]
    for regex, kind in ((BOLD_RE, 1), (CODE_RE, 2)):
        expanded: list[tuple[int, str]] = []
        for k, seg in segs:
            if k != 0:
                expanded.append((k, seg))
                continue
            pos = 0
            for m in regex.finditer(seg):
                if m.start() > pos:
                    expanded.append((0, seg[pos:m.start()]))
                expanded.append((kind, m.group(1)))
                pos = m.end()
            if pos < len(seg):
                expanded.append((0, seg[pos:]))
        segs = expanded
    for kind, seg in segs:
        if not seg:
            continue
        run = paragraph.add_run(seg)
        if base_bold or kind == 1:
            run.bold = True
        if kind == 2:
            run.font.name = "Consolas"
            run.font.size = Pt(10.5)
            run._element.rPr.get_or_add_rFonts().set(qn("w:eastAsia"), "微软雅黑")
        else:
            run.font.name = "Calibri"


def render(md_path: str | Path, out_path: str | Path,
           key_width: list[tuple[str, float]] | None = None,
           default_width: float = DEFAULT_WIDTH_IN) -> None:
    md_path = Path(md_path)
    lines = md_path.read_text(encoding="utf-8").splitlines()
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.size = Pt(11)
    _style_rfonts(normal, ascii_="Calibri", east_asia="微软雅黑")
    kwidth = key_width if key_width is not None else DEFAULT_KEY_WIDTH

    i, n = 0, len(lines)
    while i < n:
        line = lines[i].rstrip()

        if not line.strip() or line.strip() == "---":
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            h = doc.add_heading("", level=level)
            _style_rfonts(h.style, ascii_="Microsoft YaHei", east_asia="微软雅黑")
            _add_runs(h, m.group(2))
            if h.runs:
                h.runs[0].font.color.rgb = RGBColor.from_string(H_COLORS[min(level, 4) - 1])
            i += 1
            continue

        m = IMG_RE.match(line.strip())
        if m:
            alt, rel = m.group(1), m.group(2)
            img_path = (md_path.parent / rel).resolve()
            width = default_width
            for key, w in kwidth:
                if key in rel:
                    width = w
                    break
            if img_path.exists():
                doc.add_picture(str(img_path), width=Inches(width))
                doc.paragraphs[-1].alignment = 1  # 居中
                cap = doc.add_paragraph()
                run = cap.add_run(f"图：{alt}")
                run.italic = True
                run.font.size = Pt(9)
                run.font.color.rgb = RGBColor.from_string("808080")
            else:
                p = doc.add_paragraph(f"[缺图] {rel}（未找到，请手动插入）")
                p.runs[0].font.color.rgb = RGBColor.from_string("C00000")
            i += 1
            continue

        if line.startswith(">"):
            quote = []
            while i < n and lines[i].lstrip().startswith(">"):
                quote.append(lines[i].lstrip().lstrip(">").strip())
                i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.space_after = Pt(4)
            _add_runs(p, "  ".join(quote))
            for r in p.runs:
                r.font.size = Pt(10)
                r.font.color.rgb = RGBColor.from_string("595959")
            continue

        pm = PIPE.match(line)
        if pm and i + 1 < n:
            sep = PIPE.match(lines[i + 1].rstrip())
            if sep and all(SEP_CELL.match(c.strip()) for c in sep.group(1).split("|")):
                header = [c.strip() for c in pm.group(1).split("|")]
                rows = []
                i += 2
                while i < n and lines[i].strip() and PIPE.match(lines[i]):
                    rows.append([c.strip() for c in PIPE.match(lines[i]).group(1).split("|")])
                    i += 1
                table = doc.add_table(rows=1, cols=len(header))
                table.style = "Table Grid"
                for j, text in enumerate(header):
                    table.rows[0].cells[j].paragraphs[0].text = ""
                    _add_runs(table.rows[0].cells[j].paragraphs[0], text, base_bold=True)
                for row in rows:
                    cells = table.add_row().cells
                    for j, text in enumerate(row):
                        if j >= len(header):
                            break
                        cells[j].paragraphs[0].text = ""
                        _add_runs(cells[j].paragraphs[0], text)
                for r in table.rows:
                    for c in r.cells:
                        for p in c.paragraphs:
                            p.paragraph_format.space_after = Pt(1)
                doc.add_paragraph()
                continue

        if re.match(r"^\s*- ", line):
            indent = len(line) - len(line.lstrip())
            text = re.sub(r"^\s*-\s+", "", line)
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.25 + 0.25 * min(indent // 2, 3))
            p.paragraph_format.space_after = Pt(2)
            bullet = p.add_run("• ")
            bullet.font.color.rgb = RGBColor.from_string("2F5496")
            _add_runs(p, text)
            i += 1
            continue

        if re.match(r"^\s*\d+\.\s+", line):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.space_after = Pt(2)
            _add_runs(p, line.strip())
            i += 1
            continue

        p = doc.add_paragraph()
        _add_runs(p, line.strip())
        i += 1

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))


if __name__ == "__main__":
    raise SystemExit("这是库模块，请通过 render() 调用（参考 scripts/09_build_item2vec_docx.py）。")
