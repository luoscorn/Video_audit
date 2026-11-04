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
from services.qwen_client import score_video, score_video_with_rubric
from services.media_evidence import prepare_evidence
from services.supplement import supplement_rescore
from utils.audit_type import AuditType
from utils.score_grade import attach_grade
from utils.score_rules import load_rule_bundle
from utils.oss_client import get_object_url
from config import settings

logger = logging.getLogger(__name__)

# AI 调用最大重试次数
_MAX_RETRIES = 3
_RETRY_DELAY = 5  # 秒

# 同时进行的打分任务上限 避免多任务并发吃满 CPU/带宽/千问配额
_MAX_CONCURRENT_TASKS = 3
_task_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_TASKS)

# 错误信息命中这些关键词说明多模态输入本身被拒 才回退整段视频
_INPUT_REJECT_KEYWORDS = ("image", "multimodal", "invalid_parameter", "unsupported", "data_inspection")


def _is_input_rejected(exc: Exception) -> bool:
    """判断失败是否由“关键帧输入被模型拒绝”引起 限流/超时/网络类错误保留证据重试"""
    if getattr(exc, "status_code", None) == 400:
        return True
    message = str(exc).lower()
    return any(keyword in message for keyword in _INPUT_REJECT_KEYWORDS)


def _extract_object_key(oss_path: str) -> str:
    """
    从 OSS 路径中提取对象 Key（去掉 bucket 前缀）

    支持两种格式:
    - {bucket}/{oss_key}  如 wyxtapp/video/上牙槽后神经阻滞麻醉/xxx.mp4 → video/上牙槽后神经阻滞麻醉/xxx.mp4
    - {oss_key}           如 video/上牙槽后神经阻滞麻醉/xxx.mp4 → 原样返回
    """
    slash_idx = oss_path.find("/")
    if slash_idx == -1:
        raise ValueError(f"无效的 OSS 路径: {oss_path}，期望格式: {{bucket}}/{{oss_key}} 或 {{oss_key}}")

    first_segment = oss_path[:slash_idx]
    # 如果第一段是已知的 bucket 名，则去掉它；否则整个路径就是 oss_key
    if first_segment == settings.OSS_BUCKET:
        return oss_path[slash_idx + 1:]
    return oss_path


async def process_audit_task(task_id: int) -> None:
    """后台处理入口 先用信号量排队 再进入实际打分流程"""
    if _task_semaphore.locked():
        logger.warning(
            "[Worker] 并发已满(%d) task_id=%d 排队等待", _MAX_CONCURRENT_TASKS, task_id
        )
    async with _task_semaphore:
        await _process_audit_task(task_id)


async def _process_audit_task(task_id: int) -> None:
    """
    后台处理视频审核任务（由 BackgroundTasks 调度）

    流程:
    1. 查任务 → 更新状态为进行中
    2. 从 oss_url 解析 oss_key → 生成预签名 URL 给 AI
    3. 视频切片抽帧 + 逐片语音转写（失败回退整段视频）
    4. 加载打分规则（数据库生效版优先 无生效版回落 xlsx 结构化 / docx 文本）
    5. 调用千问按严格尺度逐项打分（标有画面/语音证据才给分 未达标即扣）
    6. 解析结果 → 写入 result 表 → 更新状态为完成
    7. 异常时更新状态为异常
    """
    start_time = time.time()
    logger.info("\n%s\n[Worker] 开始处理任务 task_id=%d", "=" * 60, task_id)

    async with AsyncSessionLocal() as session:
        # 本地视频临时目录持有者 任务结束时统一清理
        media_task = None
        try:
            # 1. 查询任务
            task = await get_task(session, task_id)
            if task is None:
                logger.warning("[Worker] 任务不存在 task_id=%d", task_id)
                return

            if task.task_status != 0:
                logger.warning(
                    "[Worker] 任务状态异常 task_id=%d status=%d", task_id, task.task_status
                )
                return

            logger.info(
                "[Worker]   审核类型: %s  OSS路径: %s", task.audit_type, task.oss_url
            )

            # 2. 更新状态: 进行中
            await update_task_status(session, task_id, 1)
            logger.info("[Worker]   状态更新: 待处理 → 进行中")

            # 3. 生成视频预签名 URL（给 AI 读视频用，有效期 2 小时）
            oss_key = _extract_object_key(task.oss_url)
            video_url = get_object_url(oss_key=oss_key, expired=7200)
            logger.info("[Worker]   视频URL已生成 (oss_key=%s)", oss_key)

            # 3.5 抽取打分证据：视频切片关键帧 + 逐片语音转写
            #     失败不阻断流程 仅回退为整段视频打分
            evidence = None
            if settings.AUDIT_MEDIA_ENABLED:
                media_start = time.time()
                try:
                    media_task = await asyncio.to_thread(prepare_evidence, video_url)
                    evidence = media_task
                    logger.info(
                        "[Worker]   切片抽帧+语音转写完成: %s (耗时 %.1fs)",
                        evidence.describe(),
                        time.time() - media_start,
                    )
                    transcript = evidence.transcript_text()
                    if transcript:
                        logger.debug("[Worker]   语音转写摘要: %s", transcript[:300])
                except Exception as exc:
                    evidence = None
                    logger.warning("[Worker]   切片/语音抽取失败 回退整段视频打分: %s", exc)
            else:
                logger.info("[Worker]   切片/语音抽取已关闭(AUDIT_MEDIA_ENABLED=false)")

            # 4. 加载打分规则：数据库生效版优先，无生效版回落 OSS xlsx / 本地 txt
            audit_type = AuditType(task.audit_type)
            bundle = await load_rule_bundle(session, audit_type)
            rubric_items = bundle.rubric_items
            rule_text = bundle.text
            version_label = f" 版本=v{bundle.version}" if bundle.version else ""
            if rubric_items:
                rubric_full_score = round(sum(item.score for item in rubric_items), 2)
                logger.info(
                    "[Worker]   打分规则已加载: 结构化 %d 项 (满分%s)  来源=%s%s",
                    len(rubric_items),
                    rubric_full_score,
                    bundle.source,
                    version_label,
                )
            else:
                logger.info(
                    "[Worker]   打分规则已加载: 文本 %d 字符  来源=%s%s",
                    len(rule_text or ""),
                    bundle.source,
                    version_label,
                )

            # 5. 调用 AI 打分（带重试）
            logger.info("[Worker]   正在调用 AI 打分 (最多重试%d次)...", _MAX_RETRIES)
            result = None
            last_error = None
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    ai_start = time.time()
                    # 打分是同步 HTTP 调用 必须放线程池执行 否则会阻塞事件循环导致其它接口全部卡死
                    if rubric_items:
                        result = await asyncio.to_thread(
                            score_video_with_rubric,
                            video_url=video_url,
                            rubric_items=rubric_items,
                            audit_type=task.audit_type,
                            evidence=evidence,
                        )
                    else:
                        result = await asyncio.to_thread(
                            score_video,
                            video_url=video_url,
                            rule_text=rule_text,
                            audit_type=task.audit_type,
                            evidence=evidence,
                        )
                    ai_elapsed = time.time() - ai_start
                    logger.info("[Worker]   AI 返回成功 (第%d次 耗时 %.1fs)", attempt, ai_elapsed)
                    break
                except Exception as exc:
                    last_error = exc
                    logger.warning("[Worker]   AI 打分失败(第%d/%d次): %s", attempt, _MAX_RETRIES, exc)
                    if evidence is not None and _is_input_rejected(exc):
                        # 仅当关键帧输入被拒时才回退整段视频 限流/超时保留证据重试
                        evidence = None
                        logger.warning("[Worker]   关键帧输入被拒 后续重试回退整段视频")
                    if attempt < _MAX_RETRIES:
                        wait = _RETRY_DELAY * attempt
                        logger.info("[Worker]   等待 %ds 后重试...", wait)
                        await asyncio.sleep(wait)
            if result is None:
                raise RuntimeError(f"AI 打分重试 {_MAX_RETRIES} 次仍失败: {last_error}")

            # 5.5 二次补证：对证据受限的评分项切连续片段复核 失败不阻断
            if settings.MEDIA_SUPPLEMENT_ENABLED and rubric_items and evidence is not None:
                supplement_start = time.time()
                try:
                    info = await asyncio.to_thread(
                        supplement_rescore, evidence, rubric_items, result, task.audit_type
                    )
                    if info is None:
                        logger.info("[Worker]   二次补证: 未触发或无可复核片段")
                    else:
                        logger.info(
                            "[Worker]   二次补证: 片段%d个 复核%d项 调整%d项 (耗时 %.1fs)",
                            info["clips"],
                            info["rechecked"],
                            info["changed"],
                            time.time() - supplement_start,
                        )
                        for change in info["changes"]:
                            logger.info(
                                "[Worker]   ~ %s: %s → %s (%s)",
                                change["point"],
                                change["from"],
                                change["to"],
                                change["reason"],
                            )
                except Exception as exc:
                    logger.warning("[Worker]   二次补证失败 沿用第一轮结果: %s", exc)

            total_score = result.get("total_score", 0)
            full_score = result.get("full_score", 0)
            deductions = result.get("deductions", [])

            # 服务端校验：total_score 必须 = full_score - sum(deduct_score)
            total_deduct = sum(float(d.get("deduct_score", 0) or 0) for d in deductions)
            expected_score = float(full_score or 0) - total_deduct
            if abs(float(total_score or 0) - expected_score) > 0.01:
                logger.warning(
                    "[Worker]   AI 算分有误: 返回%s 校正为%s (满分%s - 扣分%s)",
                    total_score,
                    expected_score,
                    full_score,
                    total_deduct,
                )
                result["total_score"] = expected_score
                total_score = expected_score

            # 记录本次打分使用的规则来源与版本 便于事后追溯
            result["rule_set"] = bundle.meta

            # 按得分百分率给出等级 A(>80) / B(>60) / C / D(评分数据不完整)
            attach_grade(result, len(rubric_items) if rubric_items else None)
            logger.info(
                "[Worker]   等级判定: %s 百分率=%s%s",
                result.get("grade"),
                result.get("score_rate"),
                f" ({result['grade_reason']})" if result.get("grade_reason") else "",
            )

            # 6. 保存结果 + 更新状态: 完成(同一事务 避免结果落库但状态卡在进行中)
            await save_audit_result(session, task_id, total_score, result, status=2)

            elapsed = time.time() - start_time
            logger.info(
                "[Worker] 任务完成 task_id=%d  总分=%s  扣分项=%d个  总耗时=%.1fs\n%s",
                task_id,
                total_score,
                len(deductions),
                elapsed,
                "=" * 60,
            )
            for d in deductions:
                logger.info(
                    "[Worker]   - %s: 扣%s分 (%s)",
                    d.get("item", "?"),
                    d.get("deduct_score", 0),
                    d.get("reason", ""),
                )
            summary = result.get("summary", "")
            if summary:
                logger.info("[Worker]   评价: %s", summary)

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("[Worker] 任务异常 task_id=%d  耗时=%.1fs  错误: %s", task_id, elapsed, e)
            logger.exception("[Worker] 任务处理异常 task_id=%d", task_id)
            try:
                await update_task_status(session, task_id, 3)
                logger.info("[Worker]   状态更新: → 异常(3)")
            except Exception:
                logger.exception("[Worker] 更新异常状态失败 task_id=%d", task_id)
        finally:
            # 释放本地视频临时目录（二次补证切片依赖它，必须放到最后）
            if media_task is not None:
                media_task.close()


async def recover_orphan_tasks() -> int:
    """
    启动时恢复孤儿任务：将卡在 status=1 的任务重置为 0 并重新触发

    :return: 恢复的任务数
    """
    async with AsyncSessionLocal() as session:
        orphan_ids = await get_orphan_task_ids(session)
        if not orphan_ids:
            logger.info("[Recovery] 启动检查: 无孤儿任务")
            return 0

        count = await reset_task_status(session, orphan_ids, from_status=1, to_status=0)
        logger.warning(
            "[Recovery] 发现 %d 个孤儿任务 已重置 %d 个 正在重新触发", len(orphan_ids), count
        )

    for task_id in orphan_ids:
        logger.info("[Recovery]   → 重新触发 task_id=%d", task_id)
        asyncio.create_task(process_audit_task(task_id))

    return len(orphan_ids)
