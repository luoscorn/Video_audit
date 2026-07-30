# @Auther  : luoscorn
# @Time    : 2026/7/29: 16:30
# @File    : video_audit.py
import logging
from decimal import Decimal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.input import TASK_STATUS_DESC, CreateTaskReq, PageQuery
from db.db import (
    create_video_audit_task,
    get_session,
    get_task_with_latest_result,
    list_video_audit_tasks,
)
from services.audit_worker import process_audit_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/video-audit", tags=["视频审核"])


def _format_datetime(dt) -> str | None:
    """datetime 转 '%Y-%m-%d %H:%M:%S' 字符串 为空返回 None"""
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else None


def _score_to_float(score) -> float | None:
    """Decimal 分数转 float 便于 json 序列化 为空返回 None"""
    if score is None:
        return None
    return float(score) if isinstance(score, Decimal) else score


def _task_to_dict(task) -> dict:
    """任务 ORM 对象转接口返回结构"""
    return {
        "id": task.id,
        "task_status": task.task_status,
        "task_status_desc": TASK_STATUS_DESC.get(task.task_status, "未知"),
        "audit_type": task.audit_type,
        "oss_url": task.oss_url,
        "create_time": _format_datetime(task.create_time),
        "update_time": _format_datetime(task.update_time),
    }


@router.post("/task", summary="创建审核任务")
async def create_task(
    req: CreateTaskReq,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    """
    创建视频审核任务并自动触发后台 AI 打分

    入参:
    - **audit_type**: 审核类型 必填 只能传 AuditType 枚举中的值
    - **oss_url**: 待审核视频的oss地址 必填 COS路径 格式: {bucket}/{cos_key} 如 wiya-app-1346197003/video/xxx/yyy.mp4

    返回:
    - **data.task_id**: 新建任务id 后续用于查询结果

    说明: 任务初始状态为 0-已接收 创建后自动在后台触发 AI 打分流程
    """
    try:
        task_id = await create_video_audit_task(session, req.audit_type.value, req.oss_url)
        # 创建成功后触发后台 AI 打分
        background_tasks.add_task(process_audit_task, task_id)
        return {"code": 0, "msg": "success", "data": {"task_id": task_id}}
    except Exception:
        logger.exception("创建审核任务失败")
        raise HTTPException(status_code=500, detail="创建任务失败")


@router.get("/task/{task_id}/result", summary="查询任务结果")
async def get_task_result(
    task_id: int,
    session: AsyncSession = Depends(get_session),
):
    """
    查询审核任务状态及最新一条审核结果

    路径参数:
    - **task_id**: 创建任务时返回的任务id

    返回:
    - **data.task**: 任务详情 含 task_status(0-已接收 1-进行中 2-已完成 3-异常)及其描述
    - **data.result**: 最新审核结果(score + result_json) 任务未完成时为 null

    异常: 任务不存在返回 404
    """
    task, result = await get_task_with_latest_result(session, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    # 拆解 result_json，把扣分项提到顶层方便前端直接用
    result_data = None
    if result:
        rj = result.result_json or {}
        result_data = {
            "id": result.id,
            "score": _score_to_float(result.score),
            "full_score": rj.get("full_score"),
            "total_score": rj.get("total_score"),
            "deductions": rj.get("deductions", []),
            "deduction_count": len(rj.get("deductions", [])),
            "summary": rj.get("summary", ""),
            "create_time": _format_datetime(result.create_time),
            "update_time": _format_datetime(result.update_time),
        }
    data = {
        "task": _task_to_dict(task),
        "result": result_data,
    }
    return {"code": 0, "msg": "success", "data": data}


@router.get("/tasks", summary="查询任务列表")
async def list_tasks(
    query: PageQuery = Depends(),
    session: AsyncSession = Depends(get_session),
):
    """
    分页查询审核任务列表 按创建时间倒序(id倒序)

    查询参数:
    - **page**: 页码 从1开始 默认1
    - **page_size**: 每页条数 默认10 最大100

    返回:
    - **data.total**: 任务总条数 用于计算总页数
    - **data.list**: 当前页任务列表 字段与任务详情一致
    """
    tasks, total = await list_video_audit_tasks(session, query.page, query.page_size)
    data = {
        "total": total,
        "page": query.page,
        "page_size": query.page_size,
        "list": [_task_to_dict(t) for t in tasks],
    }
    return {"code": 0, "msg": "success", "data": data}
