# @Auther  : luoscorn
# @Time    : 2026/7/30
# @File    : extract_score_rules.py
"""
一次性脚本：将 doc/score_doc/*.docx 提取为纯文本保存到 doc/score_rules/*.txt
运行方式：PYTHONPATH=. python utils/extract_score_rules.py
规则更新后重新运行即可覆盖
"""

import os
from pathlib import Path

from docx import Document
from docx.table import Table

from config import BASE_DIR

SCORE_DOC_DIR = BASE_DIR / "doc" / "score_doc"
SCORE_RULES_DIR = BASE_DIR / "doc" / "score_rules"


def _iter_block_items(parent):
    """按文档顺序交替遍历段落和表格"""
    from docx.oxml.ns import qn
    from docx.text.paragraph import Paragraph

    body = parent.element.body
    for child in body:
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


def _table_to_text(table: Table) -> str:
    """将表格转为纯文本，每行用 | 分隔"""
    lines = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def extract_docx(docx_path: Path) -> str:
    """提取 docx 全部文本内容（段落 + 表格）"""
    doc = Document(str(docx_path))
    parts = []
    for block in _iter_block_items(doc):
        if isinstance(block, Table):
            text = _table_to_text(block)
        else:
            text = block.text.strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def main():
    SCORE_RULES_DIR.mkdir(parents=True, exist_ok=True)

    docx_files = sorted(SCORE_DOC_DIR.glob("*.docx"))
    print(f"找到 {len(docx_files)} 个 docx 文件，开始提取...\n")

    for docx_path in docx_files:
        txt_name = docx_path.stem + ".txt"
        txt_path = SCORE_RULES_DIR / txt_name

        try:
            content = extract_docx(docx_path)
            txt_path.write_text(content, encoding="utf-8")
            size_kb = txt_path.stat().st_size / 1024
            print(f"✓ {txt_name}  ({size_kb:.1f} KB)")
        except Exception as e:
            print(f"✗ {docx_path.name}  错误: {e}")

    print(f"\n提取完成，输出目录: {SCORE_RULES_DIR}")


if __name__ == "__main__":
    main()
