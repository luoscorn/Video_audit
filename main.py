# @Auther  : luoscorn
# @Time    : 2026/7/28: 17:01
# @File    : main.py
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException

from api.rule_admin import router as rule_admin_router
from api.video_audit import router as video_audit_router
from db.db import close_db, engine, init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时建表 + 恢复孤儿任务，关闭时释放连接池"""
    await init_db()

    # 启动时自动恢复崩溃遗留的孤儿任务
    from services.audit_worker import recover_orphan_tasks
    recovered = await recover_orphan_tasks()
    if recovered:
        logger.info("启动恢复: 已重新触发 %d 个孤儿任务", recovered)

    yield
    await close_db()


app = FastAPI(title="Video Audit", version="1.0.0", lifespan=lifespan)

app.include_router(video_audit_router)
app.include_router(rule_admin_router)


@app.get("/health", summary="健康检查")
async def health():
    """存活探针 顺带确认数据库可用(数据库不可用时返回 503 供容器重启) """
    try:
        async with engine.connect() as conn:
            await conn.exec_driver_sql("select 1")
    except Exception:
        logger.exception("健康检查: 数据库不可用")
        raise HTTPException(status_code=503, detail="database unreachable")
    return {"status": "ok", "database": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8022, reload=True)
