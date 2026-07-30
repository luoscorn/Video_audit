# @Auther  : luoscorn
# @Time    : 2026/7/28: 17:01
# @File    : main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.video_audit import router as video_audit_router
from db.db import close_db, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时建表 + 恢复孤儿任务，关闭时释放连接池"""
    await init_db()

    # 启动时自动恢复崩溃遗留的孤儿任务
    from services.audit_worker import recover_orphan_tasks
    recovered = await recover_orphan_tasks()
    if recovered:
        import logging
        logging.getLogger(__name__).info("启动恢复: 已重新触发 %d 个孤儿任务", recovered)

    yield
    await close_db()


app = FastAPI(title="Video Audit", version="1.0.0", lifespan=lifespan)

app.include_router(video_audit_router)


@app.get("/health", summary="健康检查")
async def health():
    """存活探针 供运维/网关探活用 正常返回 {"status": "ok"}"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
