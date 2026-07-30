# @Auther  : luoscorn
# @Time    : 2026/7/30
# @File    : audit_worker.py
"""视频审核后台 Worker — 创建任务后异步执行 AI 打分流程"""

import logging

from db.db import (
    AsyncSessionLocal,
    get_task,
    save_audit_result,
    update_task_status,
)
from services.qwen_client import score_video
from utils.audit_type import AuditType, get_score_rule_text
from utils.cos_client import get_object_url

logger = logging.getLogger(__name__)


def _extract_cos_key(oss_path: str) -> str:
    """
    从 COS 路径中提取对象 Key（去掉 bucket 前缀）

    输入格式: {bucket}/{cos_key}
    示例: wiya-app-1346197003/video/上牙槽后神经阻滞麻醉/xxx.mp4 → video/上牙槽后神经阻滞麻醉/xxx.mp4
    """
    # 取第一个 / 之后的部分作为 cos_key
    slash_idx = oss_path.find("/")
    if slash_idx == -1:
        raise ValueError(f"无效的 COS 路径: {oss_path}，期望格式: {{bucket}}/{{cos_key}}")
    return oss_path[slash_idx + 1:]


async def process_audit_task(task_id: int) -> None:
    """
    后台处理视频审核任务（由 BackgroundTasks 调度）

    流程:
    1. 查任务 → 更新状态为进行中
    2. 从 oss_url 解析 cos_key → 生成预签名 URL 给 AI
    3. 加载打分规则文本
    4. 调用千问 qwen-vl-video 打分
    5. 解析结果 → 写入 result 表 → 更新状态为完成
    6. 异常时更新状态为异常
    """
    logger.info("[Worker] 开始处理任务 task_id=%d", task_id)

    async with AsyncSessionLocal() as session:
        try:
            # 1. 查询任务
            task = await get_task(session, task_id)
            if task is None:
                logger.error("[Worker] 任务不存在 task_id=%d", task_id)
                return

            if task.task_status != 0:
                logger.warning("[Worker] 任务状态异常 task_id=%d, status=%d", task_id, task.task_status)
                return

            # 2. 更新状态: 进行中
            await update_task_status(session, task_id, 1)

            # 3. 生成视频预签名 URL（给 AI 读视频用，有效期 2 小时）
            cos_key = _extract_cos_key(task.oss_url)
            video_url = get_object_url(cos_key=cos_key, expired=7200)
            logger.info("[Worker] 视频预签名 URL 已生成, cos_key=%s", cos_key)

            # 4. 加载打分规则
            audit_type = AuditType(task.audit_type)
            rule_text = get_score_rule_text(audit_type)
            if rule_text is None:
                raise ValueError(f"审核类型 '{task.audit_type}' 无对应的打分规则文件")
            logger.info("[Worker] 打分规则已加载, type=%s, len=%d", task.audit_type, len(rule_text))

            # 5. 调用 AI 打分
            result = score_video(
                video_url=video_url,
                rule_text=rule_text,
                audit_type=task.audit_type,
            )

            total_score = result.get("total_score", 0)
            logger.info("[Worker] AI 打分完成, task_id=%d, score=%.2f", task_id, total_score)

            # 6. 保存结果 + 更新状态: 完成
            await save_audit_result(session, task_id, total_score, result)
            await update_task_status(session, task_id, 2)

            logger.info("[Worker] 任务处理完成 task_id=%d, score=%.2f", task_id, total_score)

        except Exception as e:
            logger.exception("[Worker] 任务处理异常 task_id=%d", task_id)
            try:
                await update_task_status(session, task_id, 3)
            except Exception:
                logger.exception("[Worker] 更新异常状态失败 task_id=%d", task_id)
