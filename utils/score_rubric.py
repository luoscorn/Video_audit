# @Auther  : luoscorn
# @Time    : 2026/10/21
# @File    : score_rubric.py
"""xlsx 评分标准 → 结构化评分项"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from openpyxl import load_workbook

from utils.audit_type import AUDIT_SCORE_DOC, AuditType, get_score_doc_path
from utils.oss_client import download_file, get_object_url

logger = logging.getLogger(__name__)

# OSS xlsx 解析结果的进程内缓存 避免每个任务都拉一次 OSS
# 规则以数据库生效版为主 这里的 xlsx 只是兜底 容忍一定时限的旧版
_RUBRIC_TTL_SECONDS = 600

_HEADER_NAMES = {
    "dimension": "一级维度",
    "number": "编号",
    "point": "细分评分点",
    "evidence": "视频识别依据",
    "score": "分值",
    "oral_visual_weight": "口述视觉权重",
    "oral_keywords": "口述关键词",
    "image_refs": "操作图片",
}
_REQUIRED_COLUMNS = ("number", "point", "score")


@dataclass(frozen=True)
class RubricItem:
    """单个可独立判定的评分点"""
    dimension: str
    number: str
    point: str
    evidence: str
    score: float
    remark: str = ""          # 人工备注 会作为补充说明写进打分提示词
    oral_visual_weight: str = ""   # 口述/视觉权重 如 口述20% 视觉80% 写进打分提示词
    oral_keywords: str = ""        # 口述关键词 写进打分提示词
    image_refs: str = ""           # 操作图片 每行一个 OSS key 或 http 直链 首张作为规范参考图送模型

    @property
    def label(self) -> str:
        return f"{self.number} {self.point}"

    @property
    def first_image(self) -> str:
        """首张操作图片 打分时每项只送首图控制总张数"""
        for line in (self.image_refs or "").splitlines():
            ref = line.strip()
            if ref:
                return ref
        return ""


# 解析结果缓存: audit_type -> (写入时间, 评分项)
_rubric_cache: dict[AuditType, tuple[float, list[RubricItem]]] = {}


def _cell_text(value) -> str:
    return "" if value is None else str(value).strip()


def parse_xlsx_rubric(path: str | Path) -> list[RubricItem]:
    """
    解析 xlsx「评分表」工作表为结构化评分项

    :param path: 本地 xlsx 文件路径
    :return: 按表格顺序排列的评分项
    :raises ValueError: 表头缺失、无有效评分项或编号重复
    """
    wb = load_workbook(str(path), data_only=True, read_only=True)
    try:
        ws = wb["评分表"] if "评分表" in wb.sheetnames else wb.worksheets[0]
        rows = list(ws.iter_rows(values_only=True))
    finally:
        wb.close()

    header_idx = -1
    col_idx: dict[str, int] = {}
    for i, row in enumerate(rows):
        labels = [_cell_text(v) for v in row]
        if "编号" in labels and "细分评分点" in labels:
            header_idx = i
            col_idx = {
                key: labels.index(name)
                for key, name in _HEADER_NAMES.items()
                if name in labels
            }
            break

    if header_idx < 0:
        raise ValueError(f"{path} 未找到评分表表头（编号/细分评分点）")
    missing_cols = [key for key in _REQUIRED_COLUMNS if key not in col_idx]
    if missing_cols:
        raise ValueError(f"{path} 评分表缺少列: {missing_cols}")

    def cell(row, key: str) -> str:
        idx = col_idx.get(key)
        return _cell_text(row[idx]) if idx is not None and idx < len(row) else ""

    items: list[RubricItem] = []
    dimension = ""
    for row in rows[header_idx + 1:]:
        # 一级维度按合并单元格向下填充 使每个评分项都带上所属维度
        dimension = cell(row, "dimension") or dimension
        number = cell(row, "number")
        point = cell(row, "point")
        if not number or not point:
            continue
        try:
            score = float(cell(row, "score").replace("分", ""))
        except ValueError:
            continue
        if score <= 0:
            continue
        items.append(RubricItem(
            dimension=dimension,
            number=number,
            point=point,
            evidence=cell(row, "evidence"),
            score=round(score, 2),
            oral_visual_weight=cell(row, "oral_visual_weight"),
            oral_keywords=cell(row, "oral_keywords"),
            image_refs=cell(row, "image_refs"),
        ))

    if not items:
        raise ValueError(f"{path} 未解析到有效评分项")

    numbers = [item.number for item in items]
    if len(numbers) != len(set(numbers)):
        raise ValueError(f"{path} 评分项编号重复: {numbers}")

    return items


def get_score_rubric(audit_type: AuditType) -> list[RubricItem] | None:
    """
    获取结构化评分项（仅 xlsx 标准）；docx 标准返回 None，沿用文本规则

    带进程内 TTL 缓存 避免每个任务都从 OSS 拉一次 xlsx
    """
    cached = _rubric_cache.get(audit_type)
    if cached is not None and time.time() - cached[0] < _RUBRIC_TTL_SECONDS:
        return cached[1]

    items = _load_score_rubric(audit_type)
    if items:
        _rubric_cache[audit_type] = (time.time(), items)
    return items


def _load_score_rubric(audit_type: AuditType) -> list[RubricItem] | None:
    """从 OSS 拉取最新 xlsx 并解析 下载失败时回退本地缓存"""
    oss_key = AUDIT_SCORE_DOC.get(audit_type)
    if not oss_key or not oss_key.lower().endswith(".xlsx"):
        return None

    local_path = Path(get_score_doc_path(audit_type))
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        url = get_object_url(oss_key=oss_key)
        download_file(url=url, save_dir=str(local_path.parent), filename=local_path.name)
    except Exception:
        if not local_path.exists():
            raise
        logger.warning("评分表拉取失败，使用本地缓存: %s", local_path, exc_info=True)

    return parse_xlsx_rubric(local_path)
