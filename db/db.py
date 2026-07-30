# @Auther  : luoscorn
# @Time    : 2026/7/29: 16:30
# @File    : db.py
from urllib.parse import quote_plus

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import settings
from db.models import Base, VideoAuditTask, VideoAuditTaskResult

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
    """应用启动时调用 创建不存在的表"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


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


async def save_audit_result(session: AsyncSession, task_id: int, score: float, result_json: dict) -> int:
    """保存打分结果到 result 表 返回 result_id"""
    result_obj = VideoAuditTaskResult(task_id=task_id, score=score, result_json=result_json)
    session.add(result_obj)
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
