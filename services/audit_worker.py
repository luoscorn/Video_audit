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
from services.qwen_client import (
    arbitrate_conflicts,
    demote_contradicted_full_marks,
    rewrite_summary,
    score_video,
    score_video_with_rubric,
    vote_deducted_items,
)
from services.asr_client import build_asr_context
from services.media_evidence import prepare_evidence
from services.supplement import pose_probe_rescore, supplement_rescore
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

# 400 里混着大量与输入无关的成因（模型下线、欠费、配额、鉴权）
# 这些情况回退整段视频既丢证据又必然重试失败 必须先排除再按 400 处理
_NON_INPUT_400_KEYWORDS = (
    "model_not_found", "model not found", "arrearage", "insufficient balance",
    "quota", "rate limit", "throttl", "flow control",
    "api key", "apikey", "authentication", "unauthorized",
    "access denied", "forbidden", "permission",
)

# 后台任务强引用集合 asyncio 事件循环只持弱引用
# 不自己保存引用时任务可能在跑完前被 GC 回收 表现为日志说「已触发」但状态永久卡在 0
_background_tasks: set[asyncio.Task] = set()


def _spawn(coro) -> asyncio.Task:
    """创建后台任务并持有引用直到它结束"""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def _is_input_rejected(exc: Exception) -> bool:
    """判断失败是否由“关键帧输入被模型拒绝”引起 限流/超时/鉴权/欠费类错误保留证据重试"""
    message = str(exc).lower()
    if any(keyword in message for keyword in _NON_INPUT_400_KEYWORDS):
        return False
    if getattr(exc, "status_code", None) == 400:
        return True
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
    2. 加载打分规则（数据库生效版优先 无生效版回落 xlsx 结构化 / docx 文本）
    3. 从 oss_url 解析 oss_key → 生成预签名 URL 给 AI
    4. 视频切片抽帧 + 逐片语音转写（失败回退整段视频）
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

            # 3. 加载打分规则：数据库生效版优先，无生效版回落 OSS xlsx / 本地 txt
            #    必须先于抽帧与语音转写 规则缺失时立刻失败 不白烧几十秒下载/抽帧/ASR 费用
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

            # 4. 生成视频预签名 URL（给 AI 读视频用，有效期 2 小时）
            oss_key = _extract_object_key(task.oss_url)
            video_url = get_object_url(oss_key=oss_key, expired=7200)
            logger.info("[Worker]   视频URL已生成 (oss_key=%s)", oss_key)

            # 4.5 抽取打分证据：视频切片关键帧 + 逐片语音转写
            #     失败不阻断流程 仅回退为整段视频打分
            evidence = None
            if settings.AUDIT_MEDIA_ENABLED:
                media_start = time.time()
                # 评分点与口述关键词作为语音识别上下文 探针、龈沟这类术语不再被转成同音错字
                asr_context = build_asr_context(
                    task.audit_type,
                    [
                        f"{item.point} {getattr(item, 'oral_keywords', '') or ''}"
                        for item in rubric_items
                    ] if rubric_items else [rule_text or ""],
                )
                try:
                    media_task = await asyncio.to_thread(
                        prepare_evidence, video_url, asr_context=asr_context
                    )
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

            # 结构化模式必须逐项齐全: 模型返回的项全被服务端丢弃时 items 为空
            # 此时若继续走总分校正 扣分项也为空会把 total 抬到满分 形成静默假满分
            rated_items = len(result.get("items") or [])
            if rubric_items:
                if rated_items == 0:
                    raise RuntimeError("AI 未返回任何有效评分项 结果不可用 需重试或人工复核")
                if rated_items != len(rubric_items):
                    logger.warning(
                        "[Worker]   评分项数量不一致: 应有%d项 实际%d项 不按扣分项校正总分",
                        len(rubric_items),
                        rated_items,
                    )

            # 5.4 扣分项投票：同一份证据换温度重判两次 逐项取中位数
            #     单次判定会把「无名指搭在模型下颌」看成「无名指悬空」 满分项不花调用
            if settings.MEDIA_VOTE_ENABLED and rubric_items and evidence is not None:
                vote_start = time.time()
                try:
                    voted = await asyncio.to_thread(
                        vote_deducted_items,
                        video_url,
                        rubric_items,
                        result,
                        task.audit_type,
                        evidence,
                    )
                    if voted is None:
                        logger.info(
                            "[Worker]   扣分项投票: 无扣分项或结论未变 (耗时 %.1fs)",
                            time.time() - vote_start,
                        )
                    else:
                        result = voted
                        logger.info(
                            "[Worker]   扣分项投票: 参与%d项 调整%d项 (耗时 %.1fs)",
                            voted["vote"]["items"],
                            len(voted["vote"]["changed"]),
                            time.time() - vote_start,
                        )
                        for change in voted["vote"]["changed"]:
                            logger.info(
                                "[Worker]   ~ %s: %s → %s",
                                change["point"], change["from"], change["to"],
                            )
                except Exception as exc:
                    logger.warning("[Worker]   扣分项投票失败 沿用首轮结果: %s", exc)

            # 5.45 分数与理由对齐：各环节都只往上改分 没人看分数与理由是否一致
            #      实测 144 第 4 项被投票抬到满分 而理由写的是「未见考生左手持该器械」
            #      归零后交给下面两步复核 真做了会在那里被看到并升回满分
            if rubric_items:
                try:
                    aligned = demote_contradicted_full_marks(result)
                    for change in aligned:
                        logger.info(
                            "[Worker]   ~ %s: %s → %s（理由写明未做到）",
                            change["point"], change["from"], change["to"],
                        )
                except Exception as exc:
                    logger.warning("[Worker]   分数与理由对齐失败 沿用原结果: %s", exc)

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
                        # 完整摘要（片段标签/未定位项/逐项变更）由 supplement_rescore 写入 result，此处不覆盖
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

            # 5.55 姿态项聚焦复验：牵拉、无名指支点这类接触点夹在整份规则里会被直接扫过去
            #      把单项单独拿出来问「画面里有没有这个动作」 看到即满分 看不到保留原判
            if settings.MEDIA_POSE_PROBE_ENABLED and rubric_items and evidence is not None:
                probe_start = time.time()
                try:
                    probe_info = await asyncio.to_thread(
                        pose_probe_rescore,
                        evidence,
                        rubric_items,
                        result,
                        task.audit_type,
                    )
                    if probe_info is None:
                        logger.info("[Worker]   姿态复验: 无需复验或未看到 (耗时 %.1fs)", time.time() - probe_start)
                    else:
                        logger.info(
                            "[Worker]   姿态复验: 复验%d项 修正%d项 (耗时 %.1fs)",
                            probe_info["probed"],
                            len(probe_info["changes"]),
                            time.time() - probe_start,
                        )
                        for change in probe_info["changes"]:
                            logger.info(
                                "[Worker]   ~ %s: %s → %s",
                                change["point"], change["from"], change["to"],
                            )
                except Exception as exc:
                    logger.warning("[Worker]   姿态复验失败 沿用原结果: %s", exc)

            # 5.6 冲突仲裁：各项判定对同一事实给出相反结论时拿同一份证据重判这几项
            #     实测补证把第 4 项改成「左手持口镜」后 第 2 项仍带着「全程未见口镜」一起入库
            #     检测只走纯文本 只在真检出冲突时才花带证据的那一次调用
            #     无证据时不跑: 重判会退化成整段视频输入 采样不受控 不如维持原判定
            if settings.MEDIA_CONFLICT_ENABLED and rubric_items and evidence is not None:
                conflict_start = time.time()
                try:
                    arbitrated = await asyncio.to_thread(
                        arbitrate_conflicts,
                        video_url,
                        rubric_items,
                        result,
                        task.audit_type,
                        evidence,
                    )
                    if arbitrated is None:
                        logger.info(
                            "[Worker]   冲突仲裁: 无矛盾或重判后自洽 (耗时 %.1fs)",
                            time.time() - conflict_start,
                        )
                    else:
                        result = arbitrated
                        logger.info(
                            "[Worker]   冲突仲裁: 检出%d组 调整%d项 (耗时 %.1fs)",
                            arbitrated["conflicts"]["groups"],
                            len(arbitrated["conflicts"]["changed"]),
                            time.time() - conflict_start,
                        )
                        for change in arbitrated["conflicts"]["changed"]:
                            logger.info(
                                "[Worker]   ~ %s: %s → %s (%s)",
                                change["point"], change["from"], change["to"], change["fact"],
                            )
                except Exception as exc:
                    logger.warning("[Worker]   冲突仲裁失败 沿用原结果: %s", exc)

            # 5.7 摘要按最终明细重写：首轮摘要与后来改过的分数会互相矛盾
            #     实测第 2、4 项已给满分 摘要却写「未使用口镜牵拉、左手未持口镜」
            if any(result.get(key) for key in ("vote", "supplement", "pose_probe", "conflicts", "aligned")):
                try:
                    rewritten = await asyncio.to_thread(rewrite_summary, task.audit_type, result)
                    if rewritten:
                        result["summary"] = rewritten
                        logger.info("[Worker]   摘要已按最终明细重写: %s", rewritten)
                except Exception as exc:
                    logger.warning("[Worker]   摘要重写失败 保留首轮摘要: %s", exc)

            total_score = result.get("total_score", 0)
            full_score = result.get("full_score", 0)
            deductions = result.get("deductions", [])
            # 服务端校验：total_score 必须 = full_score - sum(deduct_score)
            total_deduct = sum(float(d.get("deduct_score", 0) or 0) for d in deductions)
            expected_score = float(full_score or 0) - total_deduct
            if (not rubric_items or rated_items == len(rubric_items)) and abs(
                float(total_score or 0) - expected_score
            ) > 0.01:
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
        _spawn(process_audit_task(task_id))

    return len(orphan_ids)
