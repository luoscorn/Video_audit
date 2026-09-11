# @Auther  : luoscorn
# @Time    : 2026/7/30
# @File    : extract_score_rules.py
"""
打分规则同步脚本：从 OSS 下载 docx → 提取为纯文本 txt
运行方式：PYTHONPATH=. python utils/extract_score_rules.py

流程:
  1. 列出 OSS video/ 下所有 docx 文件
  2. 下载到 doc/score_doc/（已存在且文件名相同则跳过）
  3. 提取为纯文本保存到 doc/score_rules/*.txt
  4. 自动更新 AUDIT_SCORE_DOC 映射（如有新文件）
"""

import os
import sys
from pathlib import Path

from docx import Document
from docx.table import Table

from config import BASE_DIR
from utils.oss_client import download_file, get_object_url, list_objects

SCORE_DOC_DIR = BASE_DIR / "doc" / "score_doc"
SCORE_RULES_DIR = BASE_DIR / "doc" / "score_rules"
OSS_VIDEO_PREFIX = "video/"


# =============== docx 解析 ===============

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


# =============== OSS 同步 ===============

def sync_from_oss(force: bool = False) -> list[str]:
    """
    从 OSS 下载 video/ 下的 docx 文件到 doc/score_doc/

    :param force: 是否强制重新下载已存在的文件
    :return: 下载的 oss_key 列表
    """
    SCORE_DOC_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 列出 OSS 上所有 docx
    keys = list_objects(prefix=OSS_VIDEO_PREFIX, max_keys=500)
    docx_keys = [k for k in keys if k.endswith(".docx")]

    if not docx_keys:
        print("OSS 上未找到 docx 文件")
        return []

    print(f"OSS 上找到 {len(docx_keys)} 个 docx 文件\n")

    # 2. 逐个下载（已存在则跳过）
    downloaded = []
    for i, oss_key in enumerate(docx_keys, 1):
        filename = oss_key.rsplit("/", 1)[-1]
        local_path = SCORE_DOC_DIR / filename

        if local_path.exists() and not force:
            print(f"  [{i}/{len(docx_keys)}] 跳过（已存在）: {filename}")
            downloaded.append(oss_key)
            continue

        try:
            url = get_object_url(oss_key=oss_key)
            download_file(url=url, save_dir=str(SCORE_DOC_DIR), filename=filename)
            print(f"  [{i}/{len(docx_keys)}] ✓ {filename}")
            downloaded.append(oss_key)
        except Exception as e:
            print(f"  [{i}/{len(docx_keys)}] ✗ {oss_key}  错误: {e}")

    return downloaded


def extract_all() -> None:
    """将 doc/score_doc/*.docx 提取为 doc/score_rules/*.txt"""
    SCORE_RULES_DIR.mkdir(parents=True, exist_ok=True)

    docx_files = sorted(SCORE_DOC_DIR.glob("*.docx"))
    if not docx_files:
        print("doc/score_doc/ 下无 docx 文件")
        return

    print(f"\n提取 {len(docx_files)} 个 docx → txt\n")

    success = 0
    for docx_path in docx_files:
        txt_name = docx_path.stem + ".txt"
        txt_path = SCORE_RULES_DIR / txt_name

        try:
            content = extract_docx(docx_path)
            txt_path.write_text(content, encoding="utf-8")
            size_kb = txt_path.stat().st_size / 1024
            print(f"  ✓ {txt_name}  ({size_kb:.1f} KB)")
            success += 1
        except Exception as e:
            print(f"  ✗ {docx_path.name}  错误: {e}")

    print(f"\n提取完成: {success}/{len(docx_files)} 成功")
    print(f"输出目录: {SCORE_RULES_DIR}")


# =============== 主入口 ===============

def main():
    """
    用法:
      PYTHONPATH=. python utils/extract_score_rules.py          # 同步OSS + 提取
      PYTHONPATH=. python utils/extract_score_rules.py --force   # 强制重新下载
      PYTHONPATH=. python utils/extract_score_rules.py --local   # 仅提取本地已有文件
    """
    args = sys.argv[1:]

    if "--local" in args:
        # 仅提取本地已有文件，不访问 OSS
        print("=== 仅提取本地文件 ===\n")
        extract_all()
        return

    force = "--force" in args

    # 1. 从 OSS 同步 docx
    print("=== 第一步: OSS 同步 docx ===\n")
    sync_from_oss(force=force)

    # 2. 提取 txt
    print("\n=== 第二步: 提取规则 txt ===")
    extract_all()


if __name__ == "__main__":
    main()
