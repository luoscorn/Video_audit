# @Auther  : luoscorn
# @Time    : 2026/7/30
# @File    : qwen_client.py
"""千问 VL 视频打分客户端 — 支持整段视频与「切片关键帧 + 语音转写」两种输入"""

import base64
import json
import logging
import re
import time
from typing import Optional

import requests
from openai import OpenAI

from config import settings
from services.media_evidence import VideoEvidence, frames_payload
from utils.oss_client import get_object_url
from utils.score_rubric import RubricItem

logger = logging.getLogger(__name__)

# 进程内复用的客户端 首次调用时创建
_client: Optional[OpenAI] = None

# 千问 OpenAI 兼容端点
_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 模型名走配置 供应商下线模型时改 .env 即可(qwen-vl-max 已于 2026-10 下线)
_MODEL = settings.QWEN_VL_MODEL

# 单次请求超时 180s 且不让 openai 内部重试: 重试权由 audit_worker 统一掌握
# openai 默认 timeout 600s + max_retries 2 会与 worker 的 3 次重试叠加 把单任务拖到小时级并占住并发名额
_REQUEST_TIMEOUT = 180.0
# 缺项补问轮数 与 补问仍缺时的兜底给分比例(取证据受限档下限 60%)
_MAX_REPAIR_ROUNDS = 1
_MISSING_SCORE_RATIO = 0.6
_MISSING_ITEM_REASON = "模型未返回该评分项 判定缺失 按证据受限档下限暂记 需人工复核"
# summary 长度上限 与 _SCORING_POLICY 声明的 60 字保持一致
_SUMMARY_MAX_CHARS = 60
# 规范参考图总张数上限 每项只取首图 避免叠加切片关键帧后单请求图片过多
_MAX_REFERENCE_IMAGES = 8
# 参考图下载超时 与 进程内缓存时长(参考图随规则版本固定 改图会换 key)
_REFERENCE_TIMEOUT = 15.0
_REFERENCE_CACHE_TTL = 600
_REFERENCE_CACHE_LIMIT = 200
_reference_cache: dict[str, tuple[float, str]] = {}


# 模型偶发用 1 / (1) / ①) 这类写法 统一归到 ① 形式再与评分项匹配
_NUMBER_ALIASES = {
    str(idx): circled
    for idx, circled in enumerate("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳", start=1)
}

# 评分尺度: 按评分表严格判定 该扣必扣 但不因证据受限误判为未执行
_SCORING_POLICY = """【评分尺度：按评分表严格判定 该扣必扣 但不因证据受限误判】
0. 优先级：本次给出的评分表是最终判定口径，高于本文件其余通用条文。每一项的「评分点」「视频识别依据」「备注」共同构成该项的判定标准：
   - 表格或备注已写明达成条件、判定方式或给分档位时，一律以表格为准，即使与下列通用梯度不同（例如某项备注写明“画面中能看到支点即视为达标给满分”，就按满分给，不再以“支点偶有悬空”扣分）。
   - 表格只描述正确做法、未写明判定口径时，才按下述梯度判定。
1. 逐项独立判定：先确认画面或语音中是否存在该评分点的证据，再决定给分；既不因“大胆做过”给虚高分，也不因“没看清”直接判未执行。
2. 给分梯度：从上往下判定 命中即停 不叠加
   - 口述完整（把动作要素说清） → 该项满分，不论画面是否清晰呈现；reason 引用口述原文，画面有佐证时一并引用
   - 动作可见但不规范、不到位或不完整 → 60%~85%
   - 口述简略但画面有相关动作 → 60%~85%
   - 证据受限且无相反证据（画面未拍到、遮挡、处于镜头外，口述也未交代） → 60%~85%，reason 注明“证据受限 未见违规”
   - 口述简略且画面无相关动作 → 最高 50%
   - 明确错误、与要求相反，或同一评分点内口述与画面矛盾 → 以画面为准，0~20%
   - 整项未执行（所有片段的画面与口述都无任何迹象）→ 0
3. 以下情形必须扣分，不得豁免：器械选择错误、操作顺序颠倒、跨越无菌区、器械污染、明确缺少支点或支点不稳、明显超出或绕过规定检查范围、次数或时长明显不达标、有伤害性动作。
4. 判定原则
   - 口述完整：把该评分点的动作要素说清（用什么器械、在哪个部位、怎么做、观察什么、判断结果如何）；只提了步骤名称或关键词属口述简略，按第 2 条对应档位给分。
   - 口述与画面分开识别、各自归属：口述内容归到它描述的评分点，画面动作归到它实际对应的评分点。同一时刻口述在说 A 而画面在做 B，是两个不同评分点的证据，不构成矛盾，不得据此扣 A 的分，也不得把 B 计到 A 上。
   - 只有针对同一个评分点、口述与画面相互排斥时才算矛盾（如该项口述说用大弯端，画面清楚显示用的是三弯端），以画面为准给 0~20%，reason 同时写出“口述原文”与画面依据。
   - 各项判定以有经验考官现场评分会给出的结论为基准：不因照顾学员感受把未达标写成满分，也不因拍摄条件局限把合格操作判为未执行。
   - 否定性结论必须给出反证：确认为未执行或给低分时，reason 必须写出关键帧里与该评分点直接冲突的具体画面；若只是画面未拍到、角度欠佳、细节模糊，一律表述为「证据受限」并按证据受限档给分，禁止写成「未使用」「未执行」「未做到」「不稳定」。
   - 口镜牵拉只看镜面是否到位：画面中口镜工作端（镜面）抵住或撑开一侧口角、颊侧黏膜，即认定「口镜牵拉口角」达成，不要求看到软组织明显变形（牙模型口角不易形变），也不要求全程持续。
   - 模型牙/仿真头模是标准考核场景：不得以「不是在患者口腔内」「模型无牙龈、无唾液、无真实口角」为由扣分或降档，只按动作在模型上是否规范完成判定。
   - 持续型姿态按出现过即认定：口镜牵拉口角、无名指支点、执笔式握持这类是维持性姿态，只要若干关键帧中出现过该姿态即认定达成，不要求看到动作全过程；相邻帧之间姿态的短暂变化不构成支点不稳。
   - 手部持械分工只看两手是否各持不同器械：镜头存在镜像、左右判读有歧义时，只要画面中两手分别持用不同器械（一为口镜、一为探针）即视为分工明确，不得仅因左右手判反或口镜用途未逐帧可见而判该项未执行。
5. summary 只写总体结论：用一两句话概括整体完成情况与主要失分方向（不超过 60 字），不逐条罗列失分点、不复述各扣分明细（细节放在对应 items 的 reason 里），不使用表扬、鼓励式表述，也不写空泛套话。"""

# 切片输入的证据说明
_EVIDENCE_NOTE = """【证据说明】本次输入是按时间轴组织的视频关键帧图片 + 对应片段的语音转写文本，不是完整视频。
关键帧取自画面变化最明显的时刻（换器械、换部位、开始/结束动作），标签中的 @mm:ss 是该帧的准确时刻。口述内容与画面动作要分开归属到各自的评分点，再只看同一评分点内两者是否相互排斥，不要因为同一时刻口述与画面在别的事上就判矛盾，也不要跳片段混用证据。
关键帧数量有限，动作可能落在两帧之间、被遮挡或处于镜头外：口述已把动作要素说清时仍按第 2 条第 1 档给满分；口述也未交代时才算证据受限，按评分尺度的证据受限档给分并在 reason 注明“证据受限”，任何情况下都不得据此判为未执行。
口述（语音转写）是本次考核的完成依据：口述把动作要素说清即给满分，不因画面未清晰呈现扣分；只有同一评分点内口述与画面相互排斥时，才以画面为准。"""

# 规范参考图说明: 图来自评分标准的正确操作示意 只能当规范参照 不能当考生证据
_REFERENCE_NOTE = """【规范参考图说明】下面先给出若干张来自评分标准的标准操作示意图，用于说明该项要求的规范形态（器械选型、握持方式、位置等），帮助你判断考生做法是否规范。
参考图不是考生本人的操作证据：不得因为某项配了参考图就推定考生已完成该评分点，也不得把参考图里的画面当作考生画面引用。每一项都必须依据考生视频的关键帧与语音转写判定，参考图只用于比对“规范”与“不规范”的差别。"""

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
  "summary": "<总体结论，一两句话概括完成情况与主要失分方向，不逐条罗列失分点，不超过 60 字>"
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
      "time": "<该评分点证据所在时刻，格式 mm:ss 或 mm:ss-mm:ss，必须取自关键帧标签中的 @mm:ss；全片无任何迹象填 无>",
      "reason": "<判定理由，说明画面或语音中可见/可闻的依据>"
    }
  ],
  "summary": "<总体结论，一两句话概括完成情况与主要失分方向，不逐条罗列失分点，不超过 60 字>"
}
要求：
1. items 必须覆盖评分规则中的全部评分项，每个编号恰好出现一次，不得遗漏、重复或编造编号。
2. score 范围为 0 到该项分值；先按该评分项正文与备注给出的判定口径给分，未写明时按评分尺度的给分梯度判定，不得为凑分给出虚高分数。
3. 不要自行计算总分，服务端会根据各项得分汇总。
4. reason 必须引用口述原文或画面中的具体依据，不能只写“未做到”“不规范”等空泛描述。
5. reason 用陈述句直接写判定依据与结论，不超过 100 字；禁止出现任何算式、分值换算（如 0.2×0.8=0.16）、“取上限/取下限”、百分比档位描述（如“给60%”“按80%给分”）、自问自答或自我纠正，只给最终结论；分数只由 score 字段表达。
6. time 用于证据受限项的二次复核定位，必须直接引用关键帧标签里的 @mm:ss，不得自行推算；动作持续一段时写起止两个时刻，无法定位填 无。
7. summary 不重复 items 里的扣分点，只给总体完成度式的结论。"""

# reason 里模型常附带分数换算过程 统一剔除 只保留判定依据
# 例: “给60%即0.12分”、“按80%×0.1=0.08分”、“四舍五入为0.15分”、“扣0.05分”、“得0.14”
_REASON_CALC_RE = re.compile(
    r"[，,；;：:]?[（(]?\s*(?:按|给|取|即)?\s*\d+(?:\.\d+)?\s*%?\s*(?:[×xX*]|=)\s*\d+(?:\.\d+)?\s*%?[^，,；;。）)]*"
    r"|[，,；;：:]?\s*(?:按|给|取)\s*\d+(?:\.\d+)?\s*%\s*(?:即|给)?\s*\d+(?:\.\d+)?\s*分"
    r"|[，,；;：:]?\s*(?:按|给|取)\s*\d+(?:\.\d+)?\s*%\s*(?:给|计)分"
    r"|[，,；;：:]?\s*四舍五入为?\s*\d+(?:\.\d+)?\s*分"
    r"|[，,；;：:]?\s*扣\s*\d+(?:\.\d+)?\s*分"
    r"|[，,；;：:]?\s*得\s*\d+(?:\.\d+)?\s*分?\s*$"
    r"|[，,；;：:]?\s*(?:按|给|取)\s*\d+(?:\.\d+)?\s*%\s*[~～〜-]\s*\d+(?:\.\d+)?\s*%\s*(?:给|计)?\s*分?"
    r"|[，,；;：:]?\s*(?:按|给|取)\s*\d+(?:\.\d+)?\s*%\s*(?:即|给|计)?\s*分?"
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
    # 清洗后为空且原文本身就是分数换算 则不再退回原文 避免把换算过程当日判定依据
    if not cleaned:
        return "" if _REASON_CALC_RE.search(reason) else reason
    return cleaned


def _inline(text: str) -> str:
    """把多行文本压成单行 备注允许换行录入 但进提示词必须是单行"""
    return " ".join(str(text or "").split())


def _normalize_number(number) -> str:
    """把 1 / (1) / 1) 这类编号归一到 ① 形式"""
    text = str(number or "").strip()
    core = re.sub(r"[()（）\s.]", "", text)
    return _NUMBER_ALIASES.get(core, text)


_SUMMARY_LIST_RE = re.compile(r"(?:(?<=^)|(?<=[\s，,；;。]))(?:\d+[.、)）]|[①-⑳]|[-•·])\s*")


def _clean_summary(summary: str) -> str:
    """summary 兜底: 去列表符号与换行 并按 60 字截断"""
    text = _inline(summary)
    text = _SUMMARY_LIST_RE.sub("", text)
    text = re.sub(r"[，,；;、]{2,}", "，", text).strip(" ，,；;、")
    if len(text) > _SUMMARY_MAX_CHARS:
        text = text[:_SUMMARY_MAX_CHARS].rstrip(" ，,；;、") + "…"
    return text


def _get_client() -> OpenAI:
    """获取进程内复用的千问客户端 复用连接池避免每次调用重建 TLS"""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.QWEN_API_KEY_1,
            base_url=_BASE_URL,
            timeout=_REQUEST_TIMEOUT,
            max_retries=0,
        )
    return _client


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


def _image_data_uri(ref: str) -> str:
    """操作图片转 data URI: OSS key 先预签名再下载 http 直链直接下载 失败返回空串

    与关键帧一样用 base64 内联 不把外链交给模型去拉 避免签名过期或模型侧抓取失败
    """
    cached = _reference_cache.get(ref)
    if cached is not None and time.time() - cached[0] < _REFERENCE_CACHE_TTL:
        return cached[1]
    try:
        url = ref if ref.startswith(("http://", "https://")) else get_object_url(oss_key=ref, expired=3600)
        resp = requests.get(url, timeout=_REFERENCE_TIMEOUT)
        resp.raise_for_status()
        mime = (resp.headers.get("content-type") or "").split(";")[0].strip()
        if not mime.startswith("image/"):
            mime = "image/jpeg"
        data_uri = f"data:{mime};base64,{base64.b64encode(resp.content).decode()}"
    except Exception:
        logger.warning("规范参考图获取失败 已跳过: %s", ref, exc_info=True)
        return ""
    if len(_reference_cache) >= _REFERENCE_CACHE_LIMIT:
        _reference_cache.clear()
    _reference_cache[ref] = (time.time(), data_uri)
    return data_uri


def _reference_images(rubric_items: list[RubricItem]) -> list[tuple[str, str]]:
    """收集各评分项的首张操作图片作为规范参考图 总张数受 _MAX_REFERENCE_IMAGES 约束"""
    images: list[tuple[str, str]] = []
    for item in rubric_items:
        if len(images) >= _MAX_REFERENCE_IMAGES:
            break
        ref = item.first_image
        if not ref:
            continue
        data_uri = _image_data_uri(ref)
        if data_uri:
            images.append((f"参考图 {item.number}", data_uri))
    return images


def _build_user_content(
    video_url: str,
    evidence: Optional[VideoEvidence],
    instruction: str,
    reference_images: Optional[list[tuple[str, str]]] = None,
) -> list[dict]:
    """构造用户消息: 规范参考图 + 证据(切片关键帧+语音 或 整段视频) + 判分指令"""
    content: list[dict] = []
    if reference_images:
        # 参考图放在考生证据之前 并显式声明非考生证据 避免模型把规范图当成本次操作画面
        content.append({"type": "text", "text": _REFERENCE_NOTE})
        for label, url in reference_images:
            content.append({"type": "text", "text": f"{label}："})
            content.append({"type": "image_url", "image_url": {"url": url}})
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
    data = _parse_json(raw_text)
    data["summary"] = _clean_summary(data.get("summary", ""))
    # 文本规则模式仍保留作为兜底 但标记来源: 该口径没有逐项分值 总分由模型自算 服务端无法逐项汇总校验
    data["scoring_mode"] = "legacy_text"
    logger.warning(
        "打分走文本规则模式 audit_type=%s 该类型建议迁移为结构化评分项", audit_type
    )
    return data


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
        weight = _inline(getattr(item, "oral_visual_weight", ""))
        if weight:
            lines.append(f"  口述视觉权重：{weight}")
        keywords = _inline(getattr(item, "oral_keywords", ""))
        if keywords:
            lines.append(f"  口述关键词：{keywords}")
        remark = _inline(getattr(item, "remark", ""))
        if remark:
            lines.append(f"  备注：{remark}")
    return "\n".join(lines).strip()


def format_rubric_preview(rubric_items: list[RubricItem]) -> str:
    """供规则管理页预览 与写入提示词的评分规则排版保持一致"""
    return _format_rubric(rubric_items)


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
    reference_images = _reference_images(rubric_items)
    user_content = _build_user_content(
        video_url,
        evidence,
        "请根据上述评分规则逐项打分，以 JSON 格式返回结果。",
        reference_images,
    )

    logger.info(
        "开始调用千问 API 逐项打分, audit_type=%s, items=%d, 规范参考图=%d张, 输入=%s",
        audit_type,
        len(rubric_items),
        len(reference_images),
        "切片关键帧+语音" if evidence is not None else "整段视频",
    )
    raw_text = _call_model(system_prompt, user_content, audit_type)
    return _finalize_rubric_result(raw_text, rubric_items, audit_type, video_url, evidence)


def _assemble_result(
    raw_items: list,
    summary: str,
    rubric_items: list[RubricItem],
) -> tuple[dict, list[str]]:
    """
    校验 AI 逐项结果并汇总总分、扣分项

    :return: (结果 dict, 缺失的评分项编号列表)；缺项不再抛错 由调用方补问或兜底
    """
    # 编号不能只归一化模型侧: 规则里常写成 1/2/3 而模型可能返回 ①/(1)
    # 只归一侧会让全部评分项被判为未知编号丢弃(实测规则编号为 1~9 时 items 全空)
    key_of = _normalize_number
    rubric_by_number = {key_of(item.number): item for item in rubric_items}
    order = {key_of(item.number): idx for idx, item in enumerate(rubric_items)}
    seen: set[str] = set()
    item_results = []
    deductions = []

    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            logger.warning("忽略格式错误的评分项: %r", raw_item)
            continue
        number = key_of(raw_item.get("number", ""))
        if number not in rubric_by_number:
            logger.warning("忽略未知评分项编号: %r", raw_item.get("number"))
            continue
        if number in seen:
            logger.warning("忽略重复评分项编号: %s", number)
            continue

        rubric = rubric_by_number[number]
        try:
            score = float(raw_item.get("score"))
        except (TypeError, ValueError):
            # 分数无效视同缺项 交给补问处理
            logger.warning("评分项 %s 的 score 无效: %r", number, raw_item.get("score"))
            continue
        seen.add(number)

        # 越界分数按该项分值区间截断 防止 AI 给出负分或超满分
        score = round(max(0.0, min(float(rubric.score), score)), 2)
        reason = _clean_reason(str(raw_item.get("reason", "")).strip())

        item_results.append({
            "number": rubric.number,        # 统一回规则里的原始编号 与前端、二次补证保持一致
            "dimension": rubric.dimension,
            "point": rubric.point,
            "full_score": rubric.score,
            "score": score,
            "time": str(raw_item.get("time", "")).strip(),   # 证据时刻 供二次补证定位
            "reason": reason,
            "fallback": bool(raw_item.get("fallback")),       # 补问仍缺时的兜底项 需人工复核
        })
        if score < rubric.score:
            deductions.append({
                "item": rubric.label,
                "deduct_score": round(rubric.score - score, 2),
                "reason": reason or "未说明扣分原因",
            })

    missing = [item.number for item in rubric_items if key_of(item.number) not in seen]
    item_results.sort(key=lambda entry: order[key_of(entry["number"])])
    total_score = round(sum(entry["score"] for entry in item_results), 2)

    return {
        "full_score": round(sum(item.score for item in rubric_items), 2),
        "total_score": total_score,
        "deductions": deductions,
        "items": item_results,
        "summary": _clean_summary(summary),
        "scoring_mode": "rubric",
    }, missing


def _reask_missing(
    missing: list[str],
    rubric_items: list[RubricItem],
    audit_type: str,
    video_url: str,
    evidence: Optional[VideoEvidence],
) -> list[dict]:
    """只对缺失评分项补问一次 复用同一份证据 避免整单重跑视频与转写"""
    targets = [item for item in rubric_items if item.number in set(missing)]
    target_full = round(sum(item.score for item in targets), 2)
    system_prompt = _build_system_prompt(
        audit_type,
        f"【评分规则】以下 {len(targets)} 个评分项上一轮漏答 只需补这部分，满分 {target_full:g} 分\n"
        f"{_format_rubric(targets)}",
        _RUBRIC_OUTPUT_FORMAT,
        evidence,
    )
    user_content = _build_user_content(
        video_url, evidence, "请只对上述评分项打分，以 JSON 格式返回结果。",
        _reference_images(targets),
    )
    data = _parse_json(_call_model(system_prompt, user_content, audit_type))
    patch = data.get("items")
    if not isinstance(patch, list):
        raise ValueError("补问返回缺少 items 数组")
    return [entry for entry in patch if isinstance(entry, dict)]


def _finalize_rubric_result(
    raw_text: str,
    rubric_items: list[RubricItem],
    audit_type: str,
    video_url: str,
    evidence: Optional[VideoEvidence],
) -> dict:
    """校验逐项结果: 缺项先补问一次 补问仍缺按证据受限档下限兜底 不让整单失败重跑"""
    data = _parse_json(raw_text)
    raw_items = data.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("AI 返回缺少 items 数组")
    raw_items = [entry for entry in raw_items if isinstance(entry, dict)]
    summary = data.get("summary", "")

    result, missing = _assemble_result(raw_items, summary, rubric_items)
    for _ in range(_MAX_REPAIR_ROUNDS):
        if not missing:
            break
        logger.warning("打分结果缺少评分项 %s 补问一次", missing)
        try:
            raw_items = raw_items + _reask_missing(
                missing, rubric_items, audit_type, video_url, evidence
            )
        except Exception:
            logger.exception("补问缺失评分项失败 直接按兜底处理")
            break
        result, missing = _assemble_result(raw_items, summary, rubric_items)

    if missing:
        logger.error(
            "补问后仍缺少评分项 %s 按证据受限档下限 %.0f%% 兜底",
            missing,
            _MISSING_SCORE_RATIO * 100,
        )
        by_number = {item.number: item for item in rubric_items}
        fallback = [
            {
                "number": number,
                "score": round(float(by_number[number].score) * _MISSING_SCORE_RATIO, 2),
                "reason": _MISSING_ITEM_REASON,
                "fallback": True,
            }
            for number in missing
        ]
        result, _ = _assemble_result(raw_items + fallback, summary, rubric_items)
    return result


# 二次补证: 只看连续片段复核存疑评分项
_SUPPLEMENT_FORMAT = """请严格按照以下 JSON 格式返回复核结果，不要输出任何多余内容：
{
  "items": [
    {
      "number": "<评分项编号>",
      "clear": <true/false, 片段中能否清楚看到该项要求的关键动作>,
      "score": <float, 复核后的该项得分；clear 为 false 时原样填写上一轮分数>,
      "reason": "<判定依据，clear 为 true 时必须引用片段中的具体画面>"
    }
  ]
}
要求：
1. 只输出【复核评分项】中列出的编号，每个恰好一次，不得增删。
2. clear 为 false 时 score 必须与上一轮分数完全一致，不得借“看不清”加减分；唯一例外是本轮片段语音已把该项动作要素说清，此时按评分尺度第 2 条第 1 档给满分。
3. score 范围为 0 到该项分值，判定尺度与第一轮完全一致。
4. reason 用陈述句，不超过 100 字，不得出现算式或分值换算。"""

_SUPPLEMENT_POLICY = """【二次复核说明】本次输入是上一轮按关键帧判定后存在证据缺口的评分项，以及按时间切出的连续视频片段（动作过程连续可见）。
1. 先判断 clear：片段里能否看清该项要求的关键动作细节（器械、部位、手法、次数、时长、是否污染或跨越无菌区）。
2. clear 为 true 时按评分尺度重新给分，reason 必须写出片段中看到的具体画面，不得笼统写“已完成”“规范”。
3. clear 为 false 时默认保留上一轮分数，reason 说明无法确认的原因；片段没拍到不等于做得不对，也不等于做到了。例外：本轮片段语音把该项动作要素说清、而画面未呈现时，按评分尺度第 2 条第 1 档给满分，reason 引用口述原文——口述完整与 clear 无关。
4. 只依据随附片段与语音转写判定，不得引入第一轮之外的其他推测。"""


def _as_bool(value) -> bool:
    """宽松解析布尔值 兼容模型返回的字符串 true/false"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "是")
    return bool(value)


def rescore_with_clips(
    audit_type: str,
    check_items: list[dict],
    clips: list[dict],
) -> list[dict]:
    """
    二次复核: 用连续视频片段重判证据受限的评分项

    :param audit_type: 审核类型中文名
    :param check_items: 待复核评分项 每项含 number/point/evidence/full_score/score/reason/time
    :param clips: 片段列表 每项含 label/url/transcript
    :return: [{number, clear, score, reason}] 仅含成功解析且编号在复核范围内的项
    """
    if not check_items or not clips:
        return []

    lines = [f"【复核评分项】共 {len(check_items)} 项"]
    for item in check_items:
        lines.append(
            f"{item['number']} {item['point']}（满分{item['full_score']:g}分，上一轮{item['score']:g}分）"
        )
        if item.get("evidence"):
            lines.append(f"  视频识别依据：{item['evidence']}")
        if item.get("remark"):
            lines.append(f"  备注：{_inline(item['remark'])}")
        lines.append(f"  上一轮判定：{item.get('reason') or '无'}；证据时刻：{item.get('time') or '无'}")

    system_prompt = "\n".join([
        "你是一位严格、客观的口腔医学操作考核评分专家，正在对上一轮存疑的评分项做二次复核。",
        _SCORING_POLICY,
        "",
        _SUPPLEMENT_POLICY,
        "",
        "【输出要求】",
        _SUPPLEMENT_FORMAT,
    ])

    content: list[dict] = [{"type": "text", "text": "\n".join(lines)}]
    for clip in clips:
        content.append({
            "type": "text",
            "text": f"【片段{clip['label']}】语音转写：{clip.get('transcript') or '（该片段无语音内容）'}",
        })
        content.append({"type": "video_url", "video_url": {"url": clip["url"]}})
    content.append({"type": "text", "text": "请对上述评分项逐项复核，以 JSON 格式返回结果。"})

    raw_text = _call_model(system_prompt, content, audit_type)
    return _parse_supplement(raw_text, check_items)


def _parse_supplement(raw_text: str, check_items: list[dict]) -> list[dict]:
    """校验二次复核返回: 忽略未知编号与无法解析的项 分数按该项分值截断"""
    data = _parse_json(raw_text)
    raw_items = data.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("二次复核返回缺少 items 数组")

    pending = {_normalize_number(item["number"]): item for item in check_items}
    results = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        number = _normalize_number(raw.get("number", ""))
        item = pending.pop(number, None)
        if item is None:
            continue
        try:
            score = float(raw.get("score"))
        except (TypeError, ValueError):
            continue
        results.append({
            "number": item["number"],      # 回填规则原始编号 供 _apply_updates 对上结果项
            "clear": _as_bool(raw.get("clear")),
            "score": round(max(0.0, min(float(item["full_score"]), score)), 2),
            "reason": _clean_reason(str(raw.get("reason", "")).strip()),
        })
    return results


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
