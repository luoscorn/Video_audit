# @Auther  : luoscorn
# @Time    : 2026/7/30
# @File    : audit_worker.py
"""视频审核后台 Worker — 创建任务后异步执行 AI 打分流程"""

import asyncio
import logging
import time

from db.db import (
    AsyncSessionLocal,
    get_orphan_task_ids,
    get_task,
    reset_task_status,
    save_audit_result,
    update_task_status,
)
from services.qwen_client import score_video
from utils.audit_type import AuditType, get_score_rule_text
from utils.oss_client import get_object_url

logger = logging.getLogger(__name__)

# AI 调用最大重试次数
_MAX_RETRIES = 3
_RETRY_DELAY = 5  # 秒


def _extract_object_key(oss_path: str) -> str:
    """
    从 OSS 路径中提取对象 Key（去掉 bucket 前缀）

    输入格式: {bucket}/{oss_key}
    示例: wyxtapp/video/上牙槽后神经阻滞麻醉/xxx.mp4 → video/上牙槽后神经阻滞麻醉/xxx.mp4
    """
    # 取第一个 / 之后的部分作为 oss_key
    slash_idx = oss_path.find("/")
    if slash_idx == -1:
        raise ValueError(f"无效的 OSS 路径: {oss_path}，期望格式: {{bucket}}/{{oss_key}}")
    return oss_path[slash_idx + 1:]


async def process_audit_task(task_id: int) -> None:
    """
    后台处理视频审核任务（由 BackgroundTasks 调度）

    流程:
    1. 查任务 → 更新状态为进行中
    2. 从 oss_url 解析 oss_key → 生成预签名 URL 给 AI
    3. 加载打分规则文本
    4. 调用千问 qwen-vl-video 打分
    5. 解析结果 → 写入 result 表 → 更新状态为完成
    6. 异常时更新状态为异常
    """
    start_time = time.time()
    print(f"\n{'='*60}", flush=True)
    print(f"[Worker] ▶ 开始处理任务 task_id={task_id}", flush=True)

    async with AsyncSessionLocal() as session:
        try:
            # 1. 查询任务
            task = await get_task(session, task_id)
            if task is None:
                print(f"[Worker] ✗ 任务不存在 task_id={task_id}", flush=True)
                return

            if task.task_status != 0:
                print(f"[Worker] ✗ 任务状态异常 task_id={task_id}, status={task.task_status}", flush=True)
                return

            print(f"[Worker]   审核类型: {task.audit_type}", flush=True)
            print(f"[Worker]   OSS路径:  {task.oss_url}", flush=True)

            # 2. 更新状态: 进行中
            await update_task_status(session, task_id, 1)
            print(f"[Worker]   状态更新: 待处理 → 进行中", flush=True)

            # 3. 生成视频预签名 URL（给 AI 读视频用，有效期 2 小时）
            oss_key = _extract_object_key(task.oss_url)
            video_url = get_object_url(oss_key=oss_key, expired=7200)
            print(f"[Worker]   视频URL已生成 (oss_key={oss_key})", flush=True)

            # 4. 加载打分规则
            audit_type = AuditType(task.audit_type)
            rule_text = get_score_rule_text(audit_type)
            if rule_text is None:
                raise ValueError(f"审核类型 '{task.audit_type}' 无对应的打分规则文件")
            print(f"[Worker]   打分规则已加载 ({len(rule_text)} 字符)", flush=True)

            # 5. 调用 AI 打分（带重试）
            print(f"[Worker]   正在调用 AI 打分 (最多重试{_MAX_RETRIES}次)...", flush=True)
            result = None
            last_error = None
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    ai_start = time.time()
                    result = score_video(
                        video_url=video_url,
                        rule_text=rule_text,
                        audit_type=task.audit_type,
                    )
                    ai_elapsed = time.time() - ai_start
                    print(f"[Worker]   AI 返回成功 (第{attempt}次, 耗时 {ai_elapsed:.1f}s)", flush=True)
                    break
                except Exception as exc:
                    last_error = exc
                    print(f"[Worker]   AI 打分失败(第{attempt}/{_MAX_RETRIES}次): {exc}", flush=True)
                    if attempt < _MAX_RETRIES:
                        wait = _RETRY_DELAY * attempt
                        print(f"[Worker]   等待 {wait}s 后重试...", flush=True)
                        await asyncio.sleep(wait)
            if result is None:
                raise RuntimeError(f"AI 打分重试 {_MAX_RETRIES} 次仍失败: {last_error}")

            total_score = result.get("total_score", 0)
            full_score = result.get("full_score", 0)
            deductions = result.get("deductions", [])

            # 服务端校验：total_score 必须 = full_score - sum(deduct_score)
            total_deduct = sum(float(d.get("deduct_score", 0) or 0) for d in deductions)
            expected_score = float(full_score or 0) - total_deduct
            if abs(float(total_score or 0) - expected_score) > 0.01:
                print(f"[Worker]   ⚠ AI 算分有误: AI返回{total_score}, 校正为{expected_score} (满分{full_score} - 扣分{total_deduct})", flush=True)
                result["total_score"] = expected_score
                total_score = expected_score

            # 6. 保存结果 + 更新状态: 完成
            await save_audit_result(session, task_id, total_score, result)
            await update_task_status(session, task_id, 2)

            elapsed = time.time() - start_time
            print(f"[Worker] ✓ 任务完成 task_id={task_id}  总分={total_score}  扣分项={len(deductions)}个  总耗时={elapsed:.1f}s", flush=True)
            if deductions:
                for d in deductions:
                    print(f"[Worker]   - {d.get('item', '?')}: 扣{d.get('deduct_score', 0)}分 ({d.get('reason', '')})", flush=True)
            summary = result.get("summary", "")
            if summary:
                print(f"[Worker]   评价: {summary}", flush=True)
            print(f"{'='*60}\n", flush=True)

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[Worker] ✗ 任务异常 task_id={task_id}  耗时={elapsed:.1f}s  错误: {e}", flush=True)
            logger.exception("[Worker] 任务处理异常 task_id=%d", task_id)
            try:
                await update_task_status(session, task_id, 3)
                print(f"[Worker]   状态更新: → 异常(3)", flush=True)
            except Exception:
                logger.exception("[Worker] 更新异常状态失败 task_id=%d", task_id)


async def recover_orphan_tasks() -> int:
    """
    启动时恢复孤儿任务：将卡在 status=1 的任务重置为 0 并重新触发

    :return: 恢复的任务数
    """
    async with AsyncSessionLocal() as session:
        orphan_ids = await get_orphan_task_ids(session)
        if not orphan_ids:
            print("[Recovery] 启动检查: 无孤儿任务", flush=True)
            return 0

        count = await reset_task_status(session, orphan_ids, from_status=1, to_status=0)
        print(f"[Recovery] 发现 {len(orphan_ids)} 个孤儿任务，已重置 {count} 个，正在重新触发...", flush=True)

    for task_id in orphan_ids:
        print(f"[Recovery]   → 重新触发 task_id={task_id}", flush=True)
        asyncio.create_task(process_audit_task(task_id))

    return len(orphan_ids)
