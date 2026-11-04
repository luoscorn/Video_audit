# @Auther  : luoscorn
# @Time    : 2026/7/29: 16:30
# @File    : db.py
from datetime import datetime
import logging
from urllib.parse import quote_plus

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import settings
from db.models import Base, ScoreRuleItem, ScoreRuleSet, VideoAuditTask, VideoAuditTaskResult

logger = logging.getLogger(__name__)

# 构建异步 mysql URL
_password = quote_plus(settings.LOCAL_MYSQL_PASSWORD or "")
DATABASE_URL = (
    f"mysql+asyncmy://{settings.LOCAL_MYSQL_USER}:{_password}"
    f"@{settings.LOCAL_MYSQL_HOST}:{settings.LOCAL_MYSQL_PORT}"
    f"/{settings.LOCAL_MYSQL_DB}?charset=utf8mb4"
)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,   # 取连接前先探活 避免拿到已被mysql断开的连接
    pool_recycle=3600,    # 连接存活1小时后回收 规避mysql默认8小时超时
)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """应用启动时调用 创建不存在的表（已存在表的新增列由 _ensure_columns 补齐）"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await _ensure_columns()


_RULE_ITEM_NEW_COLUMNS = {
    # 列名 -> (期望的列定义, 期望的 data_type)
    "remark": (
        "text null comment '备注 写入打分提示词作为该项补充判定说明'",
        "text",
    ),
}


async def _ensure_columns():
    """给已存在的 score_rule_item 补新增列并纠正列类型 create_all 不会改动已建好的表"""
    if engine.dialect.name != "mysql":
        return
    async with engine.begin() as conn:
        for column, (ddl, expected_type) in _RULE_ITEM_NEW_COLUMNS.items():
            existing = await conn.exec_driver_sql(
                "select data_type from information_schema.columns "
                "where table_schema = database() and table_name = 'score_rule_item' "
                f"and column_name = '{column}'"
            )
            row = existing.first()
            if row is None:
                await conn.exec_driver_sql(f"alter table score_rule_item add column {column} {ddl}")
                logger.info("score_rule_item 补列: %s", column)
            elif str(row[0]).lower() != expected_type:
                await conn.exec_driver_sql(f"alter table score_rule_item modify column {column} {ddl}")
                logger.info("score_rule_item 列类型修正: %s -> %s", column, expected_type)


async def close_db():
    """应用关闭时调用 释放连接池"""
    await engine.dispose()


async def get_session() -> AsyncSession:
    """获取一个异步 session 用于依赖注入 异常时自动回滚"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


# =============== 业务查询函数 ===============

async def get_task(session: AsyncSession, task_id: int) -> VideoAuditTask | None:
    """根据 ID 查询任务 不存在返回 None"""
    result = await session.execute(
        select(VideoAuditTask).where(VideoAuditTask.id == task_id)
    )
    return result.scalar_one_or_none()


async def update_task_status(session: AsyncSession, task_id: int, status: int) -> None:
    """更新任务状态"""
    await session.execute(
        update(VideoAuditTask).where(VideoAuditTask.id == task_id).values(task_status=status)
    )
    await session.commit()


async def save_audit_result(
    session: AsyncSession,
    task_id: int,
    score: float,
    result_json: dict,
    status: int | None = None,
) -> int:
    """保存打分结果到 result 表 可同时更新任务状态(同一事务 避免结果落库但状态卡在进行中) 返回 result_id"""
    result_obj = VideoAuditTaskResult(task_id=task_id, score=score, result_json=result_json)
    session.add(result_obj)
    if status is not None:
        await session.execute(
            update(VideoAuditTask).where(VideoAuditTask.id == task_id).values(task_status=status)
        )
    await session.commit()
    await session.refresh(result_obj)
    return result_obj.id


async def create_video_audit_task(session: AsyncSession, audit_type: str, oss_url: str) -> int:
    """创建审核任务 返回 task_id"""
    task = VideoAuditTask(audit_type=audit_type, oss_url=oss_url)
    session.add(task)
    await session.commit()
    await session.refresh(task)
    return task.id


async def get_task_with_latest_result(session: AsyncSession, task_id: int):
    """查询任务及最新一条结果 找不到返回 None"""
    result = await session.execute(
        select(VideoAuditTask).where(VideoAuditTask.id == task_id)
    )
    task = result.scalar_one_or_none()
    if task is None:
        return None, None

    result = await session.execute(
        select(VideoAuditTaskResult)
        .where(VideoAuditTaskResult.task_id == task_id)
        .order_by(VideoAuditTaskResult.id.desc())
        .limit(1)
    )
    latest_result = result.scalar_one_or_none()
    return task, latest_result


async def list_video_audit_tasks(session: AsyncSession, page: int, page_size: int):
    """分页查询任务列表 按id倒序 返回(任务列表, 总条数)"""
    total = await session.scalar(select(func.count()).select_from(VideoAuditTask))
    result = await session.execute(
        select(VideoAuditTask)
        .order_by(VideoAuditTask.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return result.scalars().all(), total or 0


async def get_orphan_task_ids(session: AsyncSession) -> list[int]:
    """查询所有卡在'进行中'(status=1) 的孤儿任务ID（服务崩溃遗留）"""
    result = await session.execute(
        select(VideoAuditTask.id).where(VideoAuditTask.task_status == 1)
    )
    return [row[0] for row in result.all()]


async def reset_task_status(session: AsyncSession, task_ids: list[int], from_status: int, to_status: int) -> int:
    """批量重置任务状态 返回受影响行数"""
    if not task_ids:
        return 0
    stmt = (
        update(VideoAuditTask)
        .where(VideoAuditTask.id.in_(task_ids), VideoAuditTask.task_status == from_status)
        .values(task_status=to_status)
    )
    result = await session.execute(stmt)
    await session.commit()
    return result.rowcount


# =============== 打分规则集 ===============

async def list_rule_sets(session: AsyncSession, audit_type: str | None = None) -> list[ScoreRuleSet]:
    """查询规则集列表 按审核类型、版本倒序"""
    stmt = select(ScoreRuleSet).order_by(ScoreRuleSet.audit_type, ScoreRuleSet.version.desc())
    if audit_type:
        stmt = stmt.where(ScoreRuleSet.audit_type == audit_type)
    return list((await session.execute(stmt)).scalars().all())


async def get_rule_set(session: AsyncSession, rule_set_id: int) -> ScoreRuleSet | None:
    """按 ID 查询规则集"""
    return await session.get(ScoreRuleSet, rule_set_id)


async def get_active_rule_set(session: AsyncSession, audit_type: str) -> ScoreRuleSet | None:
    """查询某个审核类型当前生效的规则集"""
    stmt = (
        select(ScoreRuleSet)
        .where(ScoreRuleSet.audit_type == audit_type, ScoreRuleSet.status == "active")
        .order_by(ScoreRuleSet.version.desc())
        .limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def get_latest_draft(session: AsyncSession, audit_type: str) -> ScoreRuleSet | None:
    """查询某个审核类型最新的草稿规则集"""
    stmt = (
        select(ScoreRuleSet)
        .where(ScoreRuleSet.audit_type == audit_type, ScoreRuleSet.status == "draft")
        .order_by(ScoreRuleSet.version.desc())
        .limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def get_rule_items(session: AsyncSession, rule_set_id: int) -> list[ScoreRuleItem]:
    """按顺序查询规则集下的结构化评分项"""
    stmt = (
        select(ScoreRuleItem)
        .where(ScoreRuleItem.rule_set_id == rule_set_id)
        .order_by(ScoreRuleItem.sort_order, ScoreRuleItem.id)
    )
    return list((await session.execute(stmt)).scalars().all())


async def create_rule_draft(
    session: AsyncSession,
    audit_type: str,
    kind: str,
    source: str,
    content: str | None = None,
    remark: str | None = None,
    operator: str | None = None,
) -> ScoreRuleSet:
    """新建草稿规则集 版本号取该类型当前最大版 +1"""
    max_version = await session.scalar(
        select(func.max(ScoreRuleSet.version)).where(ScoreRuleSet.audit_type == audit_type)
    )
    rule_set = ScoreRuleSet(
        audit_type=audit_type,
        version=int(max_version or 0) + 1,
        kind=kind,
        status="draft",
        source=source,
        content=content,
        remark=remark,
        operator=operator,
    )
    session.add(rule_set)
    await session.commit()
    await session.refresh(rule_set)
    return rule_set


async def save_rule_draft(
    session: AsyncSession,
    rule_set: ScoreRuleSet,
    rows: list[dict],
    content: str | None = None,
    remark: str | None = None,
    operator: str | None = None,
) -> None:
    """全量覆盖保存草稿: 评分项先删后插 保证顺序与页面一致"""
    rule_set.remark = remark or rule_set.remark
    rule_set.operator = operator or rule_set.operator
    if content is not None:
        rule_set.content = content
    await session.execute(delete(ScoreRuleItem).where(ScoreRuleItem.rule_set_id == rule_set.id))
    for order, row in enumerate(rows):
        session.add(ScoreRuleItem(rule_set_id=rule_set.id, sort_order=order, **row))
    await session.commit()


async def publish_rule_set(session: AsyncSession, rule_set: ScoreRuleSet) -> None:
    """发布规则集: 同类型其他生效版置为归档 本版置为生效"""
    await session.execute(
        update(ScoreRuleSet)
        .where(ScoreRuleSet.audit_type == rule_set.audit_type, ScoreRuleSet.status == "active")
        .values(status="archived", update_time=datetime.now())
    )
    rule_set.status = "active"
    rule_set.published_at = datetime.now()
    await session.commit()


async def delete_rule_draft(session: AsyncSession, rule_set: ScoreRuleSet) -> None:
    """删除草稿规则集及其评分项"""
    await session.execute(delete(ScoreRuleItem).where(ScoreRuleItem.rule_set_id == rule_set.id))
    await session.delete(rule_set)
    await session.commit()
