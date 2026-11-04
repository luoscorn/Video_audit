# @Auther  : luoscorn
# @Time    : 2026/11/4
# @File    : import_rubric_xlsx.py
"""把本地评分标准 xlsx 导入数据库为结构化规则集（命令行用，页面上不放开）

用法:
    APP_ENV=prod python -m utils.import_rubric_xlsx \
        --type 探诊检查 --file docs/评分标准-探诊检查.xlsx --publish

不传 --publish 时只落成草稿，可在 /admin/rules 里继续核对后再发布
"""

import argparse
import asyncio
import logging

from db import db as dao
from utils.audit_type import AuditType
from utils.score_rubric import parse_xlsx_rubric

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("import_rubric_xlsx")


def _rows_of(items) -> list[dict]:
    """解析结果转评分项行（备注留空 需要补充说明到管理页再填）"""
    return [
        {
            "dimension": item.dimension,
            "number": item.number,
            "point": item.point,
            "evidence": item.evidence,
            "remark": "",
            "score": item.score,
        }
        for item in items
    ]


async def run(audit_type: AuditType, xlsx_path: str, publish: bool, operator: str) -> int:
    items = parse_xlsx_rubric(xlsx_path)
    total = round(sum(item.score for item in items), 2)
    logger.info("解析 %s: %d 个评分项 合计 %g 分", audit_type.value, len(items), total)

    async with dao.AsyncSessionLocal() as session:
        draft = await dao.get_latest_draft(session, audit_type.value)
        if draft is not None:
            logger.error(
                "该审核类型已有草稿 v%s，请先在 /admin/rules 发布或删除后再导入", draft.version
            )
            return 1

        active = await dao.get_active_rule_set(session, audit_type.value)
        if active is not None:
            logger.info("当前生效版本 v%s 导入后将被归档", active.version)

        rule_set = await dao.create_rule_draft(
            session,
            audit_type.value,
            "rubric",
            "xlsx-local",
            remark=f"从 {xlsx_path.split('/')[-1]} 导入",
            operator=operator,
        )
        await dao.save_rule_draft(session, rule_set, _rows_of(items), operator=operator)
        logger.info("已写入草稿 v%s（%d 项）", rule_set.version, len(items))

        if publish:
            await dao.publish_rule_set(session, rule_set)
            logger.info("已发布 v%s，新打分任务立即使用该版本", rule_set.version)
        else:
            logger.info("未发布，可在 /admin/rules/%s 核对后发布", audit_type.value)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="评分标准 xlsx → 数据库规则集")
    parser.add_argument("--type", required=True, help="审核类型 取值同 AuditType")
    parser.add_argument("--file", required=True, help="本地 xlsx 路径")
    parser.add_argument("--publish", action="store_true", help="导入后直接发布生效")
    parser.add_argument("--operator", default="cli", help="操作人 记录在规则集上")
    args = parser.parse_args()

    try:
        audit_type = AuditType(args.type)
    except ValueError:
        logger.error("未知审核类型: %s", args.type)
        return 2
    return asyncio.run(run(audit_type, args.file, args.publish, args.operator))


if __name__ == "__main__":
    raise SystemExit(main())
