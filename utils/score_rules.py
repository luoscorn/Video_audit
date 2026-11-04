# @Auther  : luoscorn
# @Time    : 2026/11/4
# @File    : score_rules.py
"""打分规则装载 — 数据库生效版本优先 无生效版时回落 OSS xlsx / 本地 txt

规则维护改为「数据库存结构化评分项 + 管理页维护」 一个审核类型多版本共存 只有一版生效
老链路(OSS xlsx / 镜像内置 txt)保留为兜底 保证库里还没录入的类型不受影响
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from db import db as dao
from utils.audit_type import AuditType, get_score_rule_text
from utils.score_rubric import RubricItem, get_score_rubric

logger = logging.getLogger(__name__)


@dataclass
class RuleBundle:
    """一次打分使用的规则快照"""
    kind: str                                     # rubric-结构化评分项 / text-整段文本规则
    source: str                                   # 规则来源 便于事后追溯
    rubric_items: Optional[list[RubricItem]] = None
    text: Optional[str] = None
    rule_set_id: Optional[int] = None
    version: Optional[int] = None

    @property
    def meta(self) -> dict:
        """写入 result_json 的规则来源信息"""
        return {
            "kind": self.kind,
            "source": self.source,
            "rule_set_id": self.rule_set_id,
            "version": self.version,
            "item_count": len(self.rubric_items or []),
        }


def to_rubric_items(rows) -> list[RubricItem]:
    """数据库评分项行转 RubricItem（顺序由调用方的查询排序保证）"""
    return [
        RubricItem(
            dimension=row.dimension or "",
            number=row.number,
            point=row.point,
            evidence=row.evidence or "",
            score=round(float(row.score), 2),
            remark=row.remark or "",
        )
        for row in rows
    ]


async def load_rule_bundle(session, audit_type: AuditType) -> RuleBundle:
    """
    装载审核类型的打分规则

    顺序: 数据库生效版(rubric/text) → OSS xlsx 解析 → 本地 txt
    :raises ValueError: 三处都拿不到规则
    """
    rule_set = await dao.get_active_rule_set(session, audit_type.value)
    if rule_set is not None:
        if rule_set.kind == "rubric":
            items = to_rubric_items(await dao.get_rule_items(session, rule_set.id))
            if items:
                return RuleBundle(
                    kind="rubric",
                    source=rule_set.source or "db",
                    rubric_items=items,
                    rule_set_id=rule_set.id,
                    version=rule_set.version,
                )
        elif (rule_set.content or "").strip():
            return RuleBundle(
                kind="text",
                source=rule_set.source or "db",
                text=rule_set.content.strip(),
                rule_set_id=rule_set.id,
                version=rule_set.version,
            )
        logger.warning("生效规则集 v%s 内容为空，回落 OSS/本地规则: %s", rule_set.version, audit_type.value)

    rubric_items = await asyncio.to_thread(get_score_rubric, audit_type)
    if rubric_items:
        return RuleBundle(kind="rubric", source="oss-xlsx", rubric_items=rubric_items)

    text = get_score_rule_text(audit_type)
    if text:
        return RuleBundle(kind="text", source="local-txt", text=text)

    raise ValueError(f"审核类型 '{audit_type.value}' 无可用打分规则（数据库无生效版且 OSS/本地均未找到）")


def items_from_rows(rows) -> list[RubricItem]:
    """页面提交的评分项字典列表转 RubricItem（用于预览渲染）"""
    return [
        RubricItem(
            dimension=row.get("dimension") or "",
            number=row.get("number") or "",
            point=row.get("point") or "",
            evidence=row.get("evidence") or "",
            score=round(float(row.get("score") or 0), 2),
            remark=row.get("remark") or "",
        )
        for row in rows
    ]
