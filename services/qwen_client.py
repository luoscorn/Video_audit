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

# 评分尺度: 鼓励为主 只在明确违规与明确缺失时扣分
_ENCOURAGE_POLICY = """【评分尺度：以鼓励为主 不苛求】
1. 默认按“已完成”理解：画面中能看到对应动作，或操作者在语音中说明并已执行，均视为得分依据；不要求每个细节都被镜头完整拍到。
2. 只有两种情况才扣分：一是明确违规（器械污染、跨越无菌区、顺序颠倒、明显伤害性动作等），二是明确缺失（全片没有任何画面与语音证据）。
3. 证据不充分但没有违规迹象时从宽：可给该项 80%~100% 的分数，reason 写清“未完整呈现但无违规迹象”。
4. 扣分力度：不构成整项缺失的扣分，单项不超过该项分值的 20%；明确违规最多扣该项分值的 50%；只有整项完全缺失才可扣至满分。
5. 语音提及同样有效：语音转写中提到某步骤并已执行时，该步骤直接按完成计分；即使画面未拍到、遮挡或看不清细节，只要没有相反的画面证据，就不因“没看到”扣分。
6. 扣分前先自问两个问题：是否存在明确的违规动作？该步骤是否在画面与语音中都完全没有出现？两者都不成立时不扣分。
7. summary 用鼓励的语气：先肯定完成得好的部分，再简要提示 1~2 点可提升之处，不用苛责表述。"""

# 切片输入的证据说明
_EVIDENCE_NOTE = """【证据说明】本次输入是按时间轴切分的视频关键帧图片 + 对应片段的语音转写文本，不是完整视频；
动作可能发生在两帧之间，请结合语音转写综合判断，不要因片段间隔或未拍到而扣分。"""

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
  "summary": "<总体评价，先肯定完成得好的部分，再简要提示可提升之处>"
}
重要：total_score 必须严格等于 full_score - sum(deduct_score)，请仔细核对计算。
如果未发现明确违规或明确缺失，deductions 为空数组，total_score 等于 full_score。"""

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
  "summary": "<总体评价，先肯定完成得好的部分，再简要提示可提升之处>"
}
要求：
1. items 必须覆盖评分规则中的全部评分项，每个编号恰好出现一次，不得遗漏、重复或编造编号。
2. score 范围为 0 到该项分值；有画面或语音证据即按鼓励原则给分，证据不充分但无违规给 80%~100%；只有整项完全缺失才给 0。
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
    """拼装系统提示词: 评分规则 + 鼓励式尺度 + 证据说明 + 输出格式"""
    parts = [
        "你是一位专业的口腔医学操作考核评分专家，评分以鼓励学员为导向，不苛求细节。",
        f"请根据以下评分规则，对「{audit_type}」操作进行逐项打分。",
        "",
        rules_block,
        "",
        _ENCOURAGE_POLICY,
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
