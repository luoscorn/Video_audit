# @Auther  : luoscorn
# @Time    : 2026/7/30
# @File    : qwen_client.py
"""千问 VL 视频打分客户端 — 支持整段视频与「切片关键帧 + 语音转写」两种输入"""

import json
import logging
from typing import Optional

from openai import OpenAI

from config import settings
from services.media_evidence import VideoEvidence, frames_payload
from utils.score_rubric import RubricItem

logger = logging.getLogger(__name__)

# 千问 OpenAI 兼容端点
_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 模型名走配置 供应商下线模型时改 .env 即可(qwen-vl-max 已于 2026-10 下线)
_MODEL = settings.QWEN_VL_MODEL

# 评分尺度: 按评分表严格判定 该扣必扣 但不因证据受限误判为未执行
_SCORING_POLICY = """【评分尺度：按评分表严格判定 该扣必扣 但不因证据受限误判】
1. 逐项独立判定：先确认画面或语音中是否存在该评分点的证据，再决定给分；既不因“大胆做过”给虚高分，也不因“没看清”直接判未执行。
2. 给分梯度：
   - 动作完整、证据明确 → 该项满分
   - 动作可见但不规范、不到位或不完整 → 按实际差距给该项 60%~85%
   - 证据受限（切片间隔、镜头角度、遮挡、画面未覆盖该区域）且无相反证据 → 给该项 60%~85%，reason 写明“证据受限 未见违规”
   - 仅语音声称、画面无任何对应动作迹象 → 最高给该项 50%，reason 写明“仅语音声称 画面无对应动作”
   - 语音与画面矛盾 → 按未执行或明确错误处理（0~20%）
   - 动作明确错误或与要求相反 → 给该项 0~20%
   - 整项未执行（全片所有片段的画面与语音都无任何迹象）→ 给 0
3. 以下情形必须扣分，不得豁免：器械选择错误、操作顺序颠倒、跨越无菌区、器械污染、明确缺少支点或支点不稳、明显超出或绕过规定检查范围、次数或时长明显不达标、有伤害性动作。
4. 语音不得代替操作：语音转写是操作者自己的口述，不等于动作已经发生。
   - 口述某步骤且画面有对应动作佐证（器械进入相应部位、手部位置变化、动作顺序吻合）→ 视为已执行，给 85%~100%
   - 口述某步骤且同片段内存在相关操作上下文（正使用对应器械、在该部位持续操作、动作节奏与口述吻合），但关键帧不足以确认细节 → 给 60%~75%，reason 写明“语音与操作上下文吻合 细节未清晰呈现”
   - 只有口述、画面中相应部位与器械无任何对应动作迹象 → 视为未获证实，最高给该项 50%，不能因“语音提到”就按完成计分
   - 口述与实际动作不一致（口述在做 A 而画面在做 B）、或连续报读评分点名称而操作动作几乎无变化（照读式陈述）→ 按未执行或明确错误处理，给 0~20%
5. 不得因“照顾学员感受”把未达标项写成满分，也不得因拍摄条件局限把合格操作判为未执行，更不得把口头声称当成已完成；各项判定应以有经验考官现场评分会给出的结论为基准。
6. summary 客观陈述：列出失分点及依据并给出改进方向；不使用表扬、鼓励式表述，也不写空泛套话；口述了但未做到的项要在 summary 中明确指出。"""

# 切片输入的证据说明
_EVIDENCE_NOTE = """【证据说明】本次输入是按时间轴切分的视频关键帧图片 + 对应片段的语音转写文本，不是完整视频。
关键帧数量有限，动作可能落在两帧之间、被遮挡或处于镜头外，这属于证据受限：按第 2 条给 60%~85% 并在 reason 注明“证据受限”，不得据此判为未执行；只有全片确实无任何迹象时才给 0。
另外，语音转写是操作者的口述而非操作本身，与画面不一致时以画面为准。"""

# 要求 AI 返回的 JSON 结构说明
_OUTPUT_FORMAT = """请严格按照以下 JSON 格式返回打分结果，不要输出任何多余内容：
{
  "full_score": <float, 满分，根据评分规则中所有评分项分值之和>,
  "total_score": <float, 实际得分，必须等于 full_score 减去所有 deduct_score 之和>,
  "deductions": [
    {
      "item": "<扣分项名称，对应评分规则中的具体步骤>",
      "deduct_score": <float, 该项扣分数>,
      "reason": "<扣分原因，说明操作不规范之处>"
    }
  ],
  "summary": "<总体评价，客观列出失分点及依据并给出改进方向，不使用鼓励式表述>"
}
重要：total_score 必须严格等于 full_score - sum(deduct_score)，请仔细核对计算。
各项均达标时 deductions 为空数组，total_score 等于 full_score；只要有评分点未达标就必须列出对应扣分。"""

# 结构化评分模式：AI 只按评分项给分 总分与扣分项由服务端汇总
_RUBRIC_OUTPUT_FORMAT = """请严格按照以下 JSON 格式返回评分结果，不要输出任何多余内容：
{
  "items": [
    {
      "number": "<评分项编号，如 ①>",
      "score": <float, 该项实际得分>,
      "reason": "<判定理由，说明画面或语音中可见/可闻的依据>"
    }
  ],
  "summary": "<总体评价，客观列出失分点及依据并给出改进方向，不使用鼓励式表述>"
}
要求：
1. items 必须覆盖评分规则中的全部评分项，每个编号恰好出现一次，不得遗漏、重复或编造编号。
2. score 范围为 0 到该项分值；按给分梯度判定：达标给满分、不到位或证据受限给 60%~85%、语音与操作上下文吻合但细节不清给 60%~75%、仅语音声称最高给 50%、明确错误或与画面矛盾给 0~20%、整项无任何迹象给 0，不得为凑分给出虚高分数。
3. 不要自行计算总分，服务端会根据各项得分汇总。
4. reason 必须引用画面或语音转写中的具体依据，不能只写“未做到”“不规范”等空泛描述。"""


def _get_client() -> OpenAI:
    """获取 OpenAI 兼容的千问客户端"""
    return OpenAI(
        api_key=settings.QWEN_API_KEY_1,
        base_url=_BASE_URL,
    )


def _build_system_prompt(
    audit_type: str,
    rules_block: str,
    output_format: str,
    evidence: Optional[VideoEvidence],
) -> str:
    """拼装系统提示词: 评分规则 + 严格评分尺度 + 证据说明 + 输出格式"""
    parts = [
        "你是一位严格、客观的口腔医学操作考核评分专家，只依据评分表与视频证据判定，不放宽标准。",
        f"请根据以下评分规则，对「{audit_type}」操作进行逐项打分。",
        "",
        rules_block,
        "",
        _SCORING_POLICY,
    ]
    if evidence is not None:
        parts += ["", _EVIDENCE_NOTE]
    parts += ["", "【输出要求】", output_format]
    return "\n".join(parts)


def _build_user_content(
    video_url: str,
    evidence: Optional[VideoEvidence],
    instruction: str,
) -> list[dict]:
    """构造用户消息: 有切片证据时送关键帧 + 语音转写 否则送整段视频"""
    content: list[dict] = []
    if evidence is None:
        content.append({"type": "video_url", "video_url": {"url": video_url}})
    else:
        # 先给时间轴与语音全文 再按片段顺序贴关键帧
        content.append({"type": "text", "text": evidence.timeline_text()})
        for label, frame in frames_payload(evidence):
            content.append({"type": "text", "text": f"{label}："})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
            })
    content.append({"type": "text", "text": instruction})
    return content


def _call_model(system_prompt: str, user_content: list[dict], audit_type: str) -> str:
    """调用千问多模态接口并返回原始文本"""
    client = _get_client()
    response = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
    )
    raw_text = (response.choices[0].message.content or "").strip()
    logger.info("千问 API 返回(audit_type=%s): %s", audit_type, raw_text[:500])
    return raw_text


def score_video(
    video_url: str,
    rule_text: str,
    audit_type: str,
    evidence: Optional[VideoEvidence] = None,
) -> dict:
    """
    按文本规则对视频进行打分

    :param video_url: 视频的 OSS 预签名 URL（无切片证据时由模型直接读整段视频）
    :param rule_text: 打分规则纯文本（从 doc/score_rules/ 加载）
    :param audit_type: 审核类型中文名
    :param evidence: 切片关键帧 + 语音转写证据 为 None 时回退整段视频
    :return: AI 返回的打分结果 dict，包含 total_score / deductions / summary
    """
    system_prompt = _build_system_prompt(
        audit_type,
        f"【评分规则】\n{rule_text}",
        _OUTPUT_FORMAT,
        evidence,
    )
    user_content = _build_user_content(
        video_url,
        evidence,
        "请根据上述评分规则对该操作逐项打分，以 JSON 格式返回结果。",
    )

    logger.info(
        "开始调用千问 API 打分, audit_type=%s, 输入=%s",
        audit_type,
        "切片关键帧+语音" if evidence is not None else "整段视频",
    )
    raw_text = _call_model(system_prompt, user_content, audit_type)
    return _parse_json(raw_text)


def _format_rubric(rubric_items: list[RubricItem]) -> str:
    """将结构化评分项按维度排版为提示词文本"""
    lines = []
    current_dimension = None
    for item in rubric_items:
        if item.dimension and item.dimension != current_dimension:
            current_dimension = item.dimension
            lines.append(f"\n{current_dimension}")
        lines.append(f"{item.number} {item.point}（{item.score:g}分）")
        if item.evidence:
            lines.append(f"  视频识别依据：{item.evidence}")
    return "\n".join(lines).strip()


def score_video_with_rubric(
    video_url: str,
    rubric_items: list[RubricItem],
    audit_type: str,
    evidence: Optional[VideoEvidence] = None,
) -> dict:
    """
    按结构化评分项逐项打分 总分由服务端汇总

    :param video_url: 视频的 OSS 预签名 URL（无切片证据时由模型直接读整段视频）
    :param rubric_items: 从 xlsx 解析出的评分项
    :param audit_type: 审核类型中文名
    :param evidence: 切片关键帧 + 语音转写证据 为 None 时回退整段视频
    :return: 含 full_score / total_score / deductions / items / summary 的 dict
    """
    if not rubric_items:
        raise ValueError("结构化评分项为空")

    full_score = round(sum(item.score for item in rubric_items), 2)
    system_prompt = _build_system_prompt(
        audit_type,
        f"【评分规则】共 {len(rubric_items)} 个评分项，满分 {full_score:g} 分\n{_format_rubric(rubric_items)}",
        _RUBRIC_OUTPUT_FORMAT,
        evidence,
    )
    user_content = _build_user_content(
        video_url,
        evidence,
        "请根据上述评分规则逐项打分，以 JSON 格式返回结果。",
    )

    logger.info(
        "开始调用千问 API 逐项打分, audit_type=%s, items=%d, 输入=%s",
        audit_type,
        len(rubric_items),
        "切片关键帧+语音" if evidence is not None else "整段视频",
    )
    raw_text = _call_model(system_prompt, user_content, audit_type)
    return _build_rubric_result(raw_text, rubric_items)


def _build_rubric_result(raw_text: str, rubric_items: list[RubricItem]) -> dict:
    """校验 AI 逐项结果并汇总总分、扣分项"""
    data = _parse_json(raw_text)
    raw_items = data.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("AI 返回缺少 items 数组")

    rubric_by_number = {item.number: item for item in rubric_items}
    order = {item.number: idx for idx, item in enumerate(rubric_items)}
    seen: set[str] = set()
    item_results = []
    deductions = []

    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise ValueError(f"AI 返回的评分项格式错误: {raw_item!r}")
        number = str(raw_item.get("number", "")).strip()
        if number not in rubric_by_number:
            raise ValueError(f"AI 返回未知评分项编号: {number!r}")
        if number in seen:
            raise ValueError(f"AI 返回重复评分项编号: {number!r}")
        seen.add(number)

        rubric = rubric_by_number[number]
        try:
            score = float(raw_item.get("score"))
        except (TypeError, ValueError):
            raise ValueError(f"评分项 {number} 的 score 无效: {raw_item.get('score')!r}")

        # 越界分数按该项分值区间截断 防止 AI 给出负分或超满分
        score = round(max(0.0, min(float(rubric.score), score)), 2)
        reason = str(raw_item.get("reason", "")).strip()

        item_results.append({
            "number": number,
            "dimension": rubric.dimension,
            "point": rubric.point,
            "full_score": rubric.score,
            "score": score,
            "reason": reason,
        })
        if score < rubric.score:
            deductions.append({
                "item": rubric.label,
                "deduct_score": round(rubric.score - score, 2),
                "reason": reason or "未说明扣分原因",
            })

    missing = [item.number for item in rubric_items if item.number not in seen]
    if missing:
        raise ValueError(f"AI 返回缺少评分项: {missing}")

    item_results.sort(key=lambda entry: order[entry["number"]])
    total_score = round(sum(entry["score"] for entry in item_results), 2)

    return {
        "full_score": round(sum(item.score for item in rubric_items), 2),
        "total_score": total_score,
        "deductions": deductions,
        "items": item_results,
        "summary": str(data.get("summary", "")).strip(),
        "scoring_mode": "rubric",
    }


def _parse_json(text: str) -> dict:
    """
    从 AI 返回文本中解析 JSON，兼容 markdown 代码块包裹

    :param text: AI 返回的原始文本
    :return: 解析后的 dict
    :raises ValueError: 无法解析时抛出
    """
    # 去掉 markdown 代码块标记
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # 去掉首尾的 ``` 行
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # 尝试从文本中提取第一个 JSON 对象
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"无法从 AI 返回中解析 JSON: {text[:200]}")
