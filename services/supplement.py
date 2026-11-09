# @Auther  : luoscorn
# @Time    : 2026/11/4
# @File    : supplement.py
"""二次补证 — 只对证据不足的评分项切连续片段重判

第一轮用稀疏关键帧打分 细节动作常落在两帧之间 只能落到「证据受限」档
本模块挑出这些项 按模型给出的证据时刻切出连续短视频片段 上传 OSS 后预签名直传模型复核
复核结果只在片段里确有清晰画面时才改动分数 仍看不清则保留原分
关键约束: 不改未复核项、clear=false 不动分、总分与扣分项由服务端按各项得分重算
"""

import logging
import re
import shutil
import tempfile
import uuid
from typing import Optional

from config import settings
from services.media_evidence import VideoEvidence, cut_clip
from services.qwen_client import rescore_with_clips
from utils.oss_client import delete_object, get_object_url, upload_file

logger = logging.getLogger(__name__)

# 已明确看到违规的项排在证据不足项之后 不因关键词而整项跳过复核
_VIOLATION_KEYWORDS = (
    "错误", "颠倒", "污染", "跨越", "无支点", "超范围", "绕过", "伤害",
    "次数不足", "时长不足", "相反", "矛盾",
)

_MAX_ITEMS_PER_CALL = 10     # 单次复核的评分项上限 控制 token
_CLIP_BEFORE = 4.0           # 证据时刻前预留秒数
_CLIP_AFTER = 6.0            # 证据时刻后预留秒数
_CLIP_MIN_SECONDS = 6.0
_CLIP_MAX_SECONDS = 15.0
_MERGE_GAP_SECONDS = 2.0     # 相邻片段窗口间隔小于该值时合并为一个片段
_REUSE_OVERLAP_RATIO = 0.6   # 窗口与已有片段重叠度超过该比例时复用同一片段
_SUPPLEMENT_FRAME_LIMIT = 6  # 每个片段附带的静态关键帧张数 细节判定以这些帧为准

_TIME_RANGE_RE = re.compile(r"(\d{1,3}):(\d{2})(?:\s*[-~至到]\s*(\d{1,3}):(\d{2}))?")
_CN_SECONDS_RE = re.compile(r"(?:(\d{1,3})\s*分[钟]?\s*)?(\d+(?:\.\d+)?)\s*秒")


def _fmt(seconds: float) -> str:
    """秒转 mm:ss"""
    total = int(max(0, seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def _parse_time_hint(text: str, duration: float) -> Optional[tuple[float, float]]:
    """
    解析模型返回的证据时刻 支持 mm:ss / mm:ss-mm:ss / N分N秒

    :return: (起, 止) 秒区间 解析不出来或超出片长返回 None
    """
    if not text:
        return None
    match = _TIME_RANGE_RE.search(text)
    if match:
        start = int(match.group(1)) * 60 + int(match.group(2))
        end = int(match.group(3)) * 60 + int(match.group(4)) if match.group(3) else start
    else:
        second = _CN_SECONDS_RE.search(text)
        if second is None:
            return None
        start = end = int(second.group(1) or 0) * 60 + float(second.group(2))

    if start >= duration:
        return None
    start = max(0.0, start)
    end = min(max(start, end), duration)
    return round(start, 2), round(end, 2)


def _clip_window(start: float, end: float, duration: float) -> tuple[float, float]:
    """证据区间扩展成片段窗口 居中并限制长度 贴片尾时向前补齐"""
    center = (start + end) / 2
    need = (end - start) + _CLIP_BEFORE + _CLIP_AFTER
    length = min(_CLIP_MAX_SECONDS, max(_CLIP_MIN_SECONDS, need))
    begin = max(0.0, center - length / 2)
    finish = min(float(duration), begin + length)
    begin = max(0.0, finish - length)
    return round(begin, 2), round(finish, 2)


def _needs_supplement(item: dict) -> bool:
    """未拿满分的项全部进入候选

    稀疏关键帧下模型常把「看不清」写成「未使用」「未稳定」，用关键词白名单判断
    必然漏掉本该复核的项（支点类拿 50% 但没命中任何词就会跳过复核）
    因此只排除满分项，复核名额交给 _plan_clips 按优先级分配
    """
    return float(item["score"]) < float(item["full_score"]) - 0.01


def _plan_priority(item: dict) -> tuple[int, float]:
    """排序键: 证据不足项先复核，同组内扣分多的优先占用片段名额"""
    reason = item.get("reason") or ""
    dirty = any(keyword in reason for keyword in _VIOLATION_KEYWORDS)
    return (1 if dirty else 0, float(item["score"]) - float(item["full_score"]))


def _overlap_ratio(window: tuple[float, float], plan: dict) -> float:
    """窗口与已有片段的重叠比例 以较短者为分母"""
    start = max(window[0], plan["start"])
    end = min(window[1], plan["end"])
    if end <= start:
        return 0.0
    shorter = min(window[1] - window[0], plan["end"] - plan["start"])
    return (end - start) / shorter if shorter > 0 else 0.0


def _plan_clips(evidence: VideoEvidence, check_items: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    把待复核项分配到片段窗口 扣分多的项优先占用有限的片段名额

    :return: ([{start, end, items}], 无法定位时刻的项)
    """
    located = []
    skipped = []
    for item in check_items:
        hint = _parse_time_hint(item.get("time", ""), evidence.duration)
        if hint is None:
            skipped.append(item)
            continue
        located.append((item, hint))

    located.sort(key=lambda pair: _plan_priority(pair[0]))

    # 时刻无法定位的项不能直接丢: 模型判「整项未执行」时 time 必为「无」
    # 而这类判定恰恰最容易错——画面里有口镜被白手套掩住没认出来 就会判成全程未见
    # 用片中段兜底复核: 考生主要操作集中在片中 片头多为准备 片尾多为收尾
    if skipped:
        fallback = _clip_window(
            evidence.duration * 0.25, evidence.duration * 0.75, evidence.duration
        )
        # 排在定位项之前: 这类判定判错就是把该拿的满分判成 0 代价最大
        # 排在后面会先被定位项占满名额 变成 unlocated 永远得不到纠正
        located = [(item, fallback) for item in sorted(skipped, key=_plan_priority)] + located
        skipped = []

    plans: list[dict] = []
    filled = 0
    for item, hint in located:
        if filled >= _MAX_ITEMS_PER_CALL:
            skipped.append(item)
            continue

        window = _clip_window(hint[0], hint[1], evidence.duration)
        target = next(
            (
                plan for plan in plans
                if window[0] <= plan["end"] + _MERGE_GAP_SECONDS
                and plan["start"] - _MERGE_GAP_SECONDS <= window[1]
            ),
            None,
        )
        if target is not None:
            begin = min(target["start"], window[0])
            finish = max(target["end"], window[1])
            if finish - begin <= _CLIP_MAX_SECONDS and len(target["items"]) < _MAX_ITEMS_PER_CALL:
                target["start"], target["end"] = begin, finish
                target["items"].append(item)
                filled += 1
                continue

        # 窗口与已有片段高度重合时复用它: 为了 15 秒上限另起一段几乎相同的视频
        # 会白白吃掉片段名额 后面的项只能变成 unlocated 永远不被复核
        reused = next(
            (
                plan for plan in plans
                if len(plan["items"]) < _MAX_ITEMS_PER_CALL
                and _overlap_ratio(window, plan) >= _REUSE_OVERLAP_RATIO
            ),
            None,
        )
        if reused is not None:
            reused["start"] = min(reused["start"], window[0])
            reused["end"] = max(reused["end"], window[1])
            reused["items"].append(item)
            filled += 1
            continue

        if len(plans) >= settings.MEDIA_SUPPLEMENT_CLIPS:
            skipped.append(item)
            continue
        plans.append({"start": window[0], "end": window[1], "items": [item]})
        filled += 1

    return plans, skipped


def _transcript_text(evidence: VideoEvidence, start: float, end: float) -> str:
    """取片段窗口范围内的语音转写 无重叠切片时返回空串"""
    parts = [
        f"[{s.label}] {s.transcript}"
        for s in evidence.slices
        if s.transcript and s.end > start and s.start < end
    ]
    return " ".join(parts)


def _recalc_totals(result: dict) -> None:
    """按各项得分重算扣分项与总分 保证 total = full - sum(deduct)"""
    deductions = []
    for item in result.get("items", []):
        full = float(item["full_score"])
        score = float(item["score"])
        if score < full - 0.01:
            deductions.append({
                "item": f"{item['number']} {item['point']}",
                "deduct_score": round(full - score, 2),
                "reason": item.get("reason") or "未说明扣分原因",
            })
    result["deductions"] = deductions
    result["total_score"] = round(sum(float(i["score"]) for i in result.get("items", [])), 2)


def _apply_updates(result: dict, updates: list[dict], clips: list[dict],
                   skipped: list[dict]) -> dict:
    """合并复核结果: 只有 clear=true 才改动分数 其余保持原分并留痕"""
    by_number = {item["number"]: item for item in result.get("items", [])}
    changes = []
    rechecked = 0
    for update in updates:
        item = by_number.get(update["number"])
        if item is None:
            continue
        rechecked += 1
        if not update["clear"]:
            item["recheck"] = "二次复核仍证据不足，保持原分"
            continue
        if abs(update["score"] - float(item["score"])) < 0.01:
            item["recheck"] = "二次复核确认，分数不变"
            continue

        changes.append({
            "number": update["number"],
            "point": item["point"],
            "from": float(item["score"]),
            "to": update["score"],
            "reason": update["reason"] or item.get("reason", ""),
        })
        item["score"] = update["score"]
        if update["reason"]:
            item["reason"] = update["reason"]
        item["recheck"] = "二次补证修正"

    _recalc_totals(result)
    result["supplement"] = {
        "clips": [clip["label"].strip() for clip in clips],
        "rechecked": rechecked,
        "changed": len(changes),
        "unlocated": [item["number"] for item in skipped],
        "changes": changes,
    }
    return {
        "rechecked": rechecked,
        "changed": len(changes),
        "clips": len(clips),
        "changes": changes,
    }


def supplement_rescore(
    evidence: VideoEvidence,
    rubric_items: list,
    result: dict,
    audit_type: str,
) -> Optional[dict]:
    """
    对证据不足的评分项切连续片段二次复核并合并进结果

    :param evidence: 第一轮的证据集合 需保留本地视频文件
    :param rubric_items: 结构化评分项 用于补上「视频识别依据」
    :param result: 第一轮打分结果 会被就地更新
    :param audit_type: 审核类型中文名
    :return: 变更摘要 未触发或无可用片段时返回 None
    """
    items = result.get("items") or []
    if not evidence.video_path or not items:
        return None

    rubric_by_number = {item.number: item for item in rubric_items}
    check_items = []
    for item in items:
        if not _needs_supplement(item):
            continue
        rubric = rubric_by_number.get(item["number"])
        check_items.append({
            "number": item["number"],
            "point": item["point"],
            "evidence": rubric.evidence if rubric else "",
            "remark": rubric.remark if rubric else "",
            "full_score": float(item["full_score"]),
            "score": float(item["score"]),
            "reason": item.get("reason", ""),
            "time": item.get("time", ""),
        })
    if not check_items:
        return None

    plans, skipped = _plan_clips(evidence, check_items)
    if not plans:
        logger.info("二次补证: %d 个存疑项均无可用证据时刻，跳过", len(check_items))
        return None

    work_dir = tempfile.mkdtemp(prefix="video_audit_clip_")
    uploaded: list[str] = []
    try:
        clips = []
        for idx, plan in enumerate(plans):
            local_path = cut_clip(
                evidence.video_path, plan["start"], plan["end"], work_dir, f"clip{idx}"
            )
            if local_path is None:
                continue
            oss_key = f"tmp/audit-clip/{uuid.uuid4().hex}.mp4"
            try:
                upload_file(local_path, oss_key)
                clip_url = get_object_url(oss_key=oss_key, expired=600)
            except Exception as exc:
                logger.warning("补证片段上传失败: %s", exc)
                continue
            uploaded.append(oss_key)
            # 静态帧比视频片段更容易看清器械细节 模型自己从视频里采样不可控
            frames = evidence.frames_between(
                plan["start"], plan["end"], _SUPPLEMENT_FRAME_LIMIT
            )
            clips.append({
                "label": f" {_fmt(plan['start'])}-{_fmt(plan['end'])}",
                "url": clip_url,
                "transcript": _transcript_text(evidence, plan["start"], plan["end"]),
                "frames": [(frame.label, frame.data) for frame in frames],
                "items": plan["items"],
            })

        if not clips:
            logger.warning("二次补证: 无可用片段，跳过复核")
            return None

        updates = rescore_with_clips(
            audit_type,
            [item for clip in clips for item in clip["items"]],
            clips,
        )
        if not updates:
            logger.warning("二次补证: 模型未返回有效复核结果")
            return None
        return _apply_updates(result, updates, clips, skipped)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        for oss_key in uploaded:
            try:
                delete_object(oss_key)
            except Exception as exc:
                logger.warning("补证片段临时对象清理失败 %s: %s", oss_key, exc)
