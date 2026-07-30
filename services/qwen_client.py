# @Auther  : luoscorn
# @Time    : 2026/7/30
# @File    : qwen_client.py
"""千问 VL 视频打分客户端 — 通过 OpenAI 兼容模式调用 qwen-vl-video"""

import json
import logging

from openai import OpenAI

from config import settings

logger = logging.getLogger(__name__)

# 千问 OpenAI 兼容端点
_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_MODEL = "qwen-vl-max"

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


def _get_client() -> OpenAI:
    """获取 OpenAI 兼容的千问客户端"""
    return OpenAI(
        api_key=settings.QWEN_API_KEY_1,
        base_url=_BASE_URL,
    )


def score_video(video_url: str, rule_text: str, audit_type: str) -> dict:
    """
    调用千问 qwen-vl-video 对视频进行打分

    :param video_url: 视频的 COS 预签名 URL（千问 VL 可直接读取）
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
