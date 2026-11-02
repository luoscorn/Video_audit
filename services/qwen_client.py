# @Auther  : luoscorn
# @Time    : 2026/7/30
# @File    : qwen_client.py
"""千问 VL 视频打分客户端 — 支持整段视频与「切片关键帧 + 语音转写」两种输入"""

import json
import logging
import re
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
   - 口述完整（包含动作要素）→ 视为已完成，给该项 85%~100%
   - 动作可见但不规范、不到位或不完整 → 按实际差距给该项 60%~85%
   - 证据受限（切片间隔、镜头角度、遮挡、画面未覆盖该区域）且无相反证据 → 给该项 60%~85%，reason 写明“证据受限 未见违规”
   - 口述简略（仅提及步骤名称或关键词）且画面无相关动作 → 最高给该项 50%
   - 动作明确错误、与要求相反，或口述与画面明显矛盾 → 给该项 0~20%
   - 整项未执行（全片所有片段的画面与口述都无任何迹象）→ 给 0
3. 以下情形必须扣分，不得豁免：器械选择错误、操作顺序颠倒、跨越无菌区、器械污染、明确缺少支点或支点不稳、明显超出或绕过规定检查范围、次数或时长明显不达标、有伤害性动作。
4. 口述与画面的判定：本考核在模拟模型上完成，细节大多靠口述交代，口述是重要的完成依据，但“完整”与“简略”要分开看。
   - 口述完整：把该评分点的动作要素说清了（用什么器械、在哪个部位、怎么做、观察什么、判断结果如何）→ 视为已完成，给 85%~100%；即使画面未清晰呈现也不扣分，reason 中引用口述原文
   - 口述完整且画面有对应动作佐证 → 给满分
   - 口述简略：只提了步骤名称或关键词，没说清怎么做、看什么 → 画面有相关动作给 60%~85%，画面无相关动作最高 50%
   - 口述内容与画面明显矛盾（口述在 A 部位而画面在 B 部位、口述用大弯端而画面用三弯端）→ 以画面为准，给 0~20%
   - 整项既无口述也无画面迹象 → 给 0
5. 不得因“照顾学员感受”把确实未达标的项写成满分，也不得因拍摄条件局限把合格操作判为未执行；口述完整即视为已完成，但口述与画面矛盾时以画面为准。各项判定应以有经验考官现场评分会给出的结论为基准。
6. summary 客观陈述：列出失分点及依据并给出改进方向；不使用表扬、鼓励式表述，也不写空泛套话；口述简略且画面无证据的项要在 summary 中明确指出。"""

# 切片输入的证据说明
_EVIDENCE_NOTE = """【证据说明】本次输入是按时间轴切分的视频关键帧图片 + 对应片段的语音转写文本，不是完整视频。
关键帧数量有限，动作可能落在两帧之间、被遮挡或处于镜头外，这属于证据受限：按第 2 条给 60%~85% 并在 reason 注明“证据受限”，不得据此判为未执行。
口述（语音转写）是本次考核的重要完成依据：口述把动作要素说清即视为已完成；只有口述与画面明显矛盾时，才以画面为准。"""

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
2. score 范围为 0 到该项分值；按给分梯度判定：达标给满分、口述完整视为已完成给 85%~100%、不到位或证据受限给 60%~85%、口述简略且画面无动作最高给 50%、明确错误或口述与画面矛盾给 0~20%、整项无任何迹象给 0，不得为凑分给出虚高分数。
3. 不要自行计算总分，服务端会根据各项得分汇总。
4. reason 必须引用口述原文或画面中的具体依据，不能只写“未做到”“不规范”等空泛描述。
5. reason 用陈述句直接写判定依据与结论，不超过 100 字；禁止出现任何算式、分值换算（如 0.2×0.8=0.16）、“取上限/取下限”、自问自答或自我纠正，只给最终结论。"""

# reason 里模型常附带分数换算过程 统一剔除 只保留判定依据
# 例: “给60%即0.12分”、“按80%×0.1=0.08分”、“四舍五入为0.15分”、“扣0.05分”、“得0.14”
_REASON_CALC_RE = re.compile(
    r"[，,；;：:]?[（(]?\s*(?:按|给|取|即)?\s*\d+(?:\.\d+)?\s*%?\s*(?:[×xX*]|=)\s*\d+(?:\.\d+)?\s*%?[^，,；;。）)]*"
    r"|[，,；;：:]?\s*(?:按|给|取)\s*\d+(?:\.\d+)?\s*%\s*(?:即|给)?\s*\d+(?:\.\d+)?\s*分"
    r"|[，,；;：:]?\s*(?:按|给|取)\s*\d+(?:\.\d+)?\s*%\s*(?:给|计)分"
    r"|[，,；;：:]?\s*四舍五入为?\s*\d+(?:\.\d+)?\s*分"
    r"|[，,；;：:]\s*扣\s*\d+(?:\.\d+)?\s*分"
    r"|[，,；;：:]\s*得\s*\d+(?:\.\d+)?\s*分?\s*$"
)


def _strip_unpaired_brackets(text: str) -> str:
    """删除无法配对的括号 保留配对完整的括号内容"""
    drop: set[int] = set()
    stack: list[int] = []
    for idx, ch in enumerate(text):
        if ch in "（(":
            stack.append(idx)
        elif ch in "）)":
            if stack:
                stack.pop()
            else:
                drop.add(idx)
    drop.update(stack)
    return "".join(ch for idx, ch in enumerate(text) if idx not in drop)


def _clean_reason(reason: str) -> str:
    """去掉 reason 中的分数换算描述 只保留判定依据"""
    if not reason:
        return ""
    cleaned = _REASON_CALC_RE.sub("", reason)
    # 算式被剔除后可能残留空括号或落单括号 一并清理
    cleaned = re.sub(r"[（(]\s*[）)]", "", cleaned)
    cleaned = _strip_unpaired_brackets(cleaned)
    cleaned = re.sub(r"[，,；;：:]{2,}", "，", cleaned).strip(" ，,；;：:")
    # 清洗后为空则退回原文 避免整段判定依据被误删
    return cleaned or reason


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
        reason = _clean_reason(str(raw_item.get("reason", "")).strip())

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
