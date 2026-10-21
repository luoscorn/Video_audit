# @Auther  : luoscorn
# @Time    : 2026/7/30
# @File    : qwen_client.py
"""千问 VL 视频打分客户端 — 通过 OpenAI 兼容模式调用 qwen3-vl-plus"""

import json
import logging

from openai import OpenAI

from config import settings
from utils.score_rubric import RubricItem

logger = logging.getLogger(__name__)

# 千问 OpenAI 兼容端点
_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# qwen-vl-max 已下线(model_not_found) 2026-10 实测当前可用: qwen3-vl-plus
_MODEL = "qwen3-vl-plus"

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
  "summary": "<总体评价，概述操作规范程度和主要失分点>"
}
重要：total_score 必须严格等于 full_score - sum(deduct_score)，请仔细核对计算。
如果操作完全规范，deductions 为空数组，total_score 等于 full_score。"""

# 结构化评分模式：AI 只按评分项给分 总分与扣分项由服务端汇总
_RUBRIC_OUTPUT_FORMAT = """请严格按照以下 JSON 格式返回评分结果，不要输出任何多余内容：
{
  "items": [
    {
      "number": "<评分项编号，如 ①>",
      "score": <float, 该项实际得分>,
      "reason": "<判定理由，说明视频中可见的依据>"
    }
  ],
  "summary": "<总体评价，概述操作规范程度和主要失分点>"
}
要求：
1. items 必须覆盖评分规则中的全部评分项，每个编号恰好出现一次，不得遗漏、重复或编造编号。
2. score 范围为 0 到该项分值；完全符合得满分，未做或明显错误得 0，部分符合可给部分分。
3. 不要自行计算总分，服务端会根据各项得分汇总。
4. reason 必须引用视频中的可见证据，不能只写“未做到”“不规范”等空泛描述。"""


def _get_client() -> OpenAI:
    """获取 OpenAI 兼容的千问客户端"""
    return OpenAI(
        api_key=settings.QWEN_API_KEY_1,
        base_url=_BASE_URL,
    )


def score_video(video_url: str, rule_text: str, audit_type: str) -> dict:
    """
    调用千问 qwen-vl-video 对视频进行打分

    :param video_url: 视频的 OSS 预签名 URL（千问 VL 可直接读取）
    :param rule_text: 打分规则纯文本（从 doc/score_rules/ 加载）
    :param audit_type: 审核类型中文名
    :return: AI 返回的打分结果 dict，包含 total_score / deductions / summary
    """
    client = _get_client()

    system_prompt = (
        f"你是一位专业的口腔医学操作考核评分专家。\n"
        f"请根据以下评分规则，对视频中的「{audit_type}」操作进行逐项打分。\n\n"
        f"【评分规则】\n{rule_text}\n\n"
        f"【输出要求】\n{_OUTPUT_FORMAT}"
    )

    user_content = [
        {
            "type": "video_url",
            "video_url": {"url": video_url},
        },
        {
            "type": "text",
            "text": "请仔细观看这段视频，根据上述评分规则对该操作进行逐项打分，以 JSON 格式返回结果。",
        },
    ]

    logger.info("开始调用千问 API 打分, audit_type=%s, video_url=%s", audit_type, video_url[:80])

    response = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
    )

    raw_text = response.choices[0].message.content.strip()
    logger.info("千问 API 原始返回: %s", raw_text[:500])

    # 解析 AI 返回的 JSON（兼容 markdown 代码块包裹）
    result = _parse_json(raw_text)
    return result


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
) -> dict:
    """
    按结构化评分项调用千问逐项打分 总分由服务端汇总

    :param video_url: 视频的 OSS 预签名 URL
    :param rubric_items: 从 xlsx 解析出的评分项
    :param audit_type: 审核类型中文名
    :return: 含 full_score / total_score / deductions / items / summary 的 dict
    """
    if not rubric_items:
        raise ValueError("结构化评分项为空")

    client = _get_client()
    full_score = round(sum(item.score for item in rubric_items), 2)

    system_prompt = (
        f"你是一位专业的口腔医学操作考核评分专家。\n"
        f"请根据以下评分规则，对视频中的「{audit_type}」操作进行逐项打分。\n\n"
        f"【评分规则】共 {len(rubric_items)} 个评分项，满分 {full_score:g} 分\n"
        f"{_format_rubric(rubric_items)}\n\n"
        f"【输出要求】\n{_RUBRIC_OUTPUT_FORMAT}"
    )

    user_content = [
        {
            "type": "video_url",
            "video_url": {"url": video_url},
        },
        {
            "type": "text",
            "text": "请仔细观看这段视频，根据上述评分规则逐项打分，以 JSON 格式返回结果。",
        },
    ]

    logger.info("开始调用千问 API 逐项打分, audit_type=%s, items=%d", audit_type, len(rubric_items))

    response = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
    )

    raw_text = response.choices[0].message.content.strip()
    logger.info("千问 API 原始返回: %s", raw_text[:500])
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
