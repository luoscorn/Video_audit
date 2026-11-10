# @Auther  : luoscorn
# @Time    : 2026/10/30
# @File    : asr_client.py
"""千问语音转写客户端 — 通过 OpenAI 兼容模式调用 qwen3-asr-flash"""

import base64
import logging
from typing import Optional

from openai import OpenAI

from config import settings

logger = logging.getLogger(__name__)

# 千问 OpenAI 兼容端点
_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 音频 MIME 映射（ffmpeg 抽出的音频统一为 mp3）
_MIME_BY_SUFFIX = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
}

# 领域上下文上限 qwen3-asr-flash 的 context 支持一万 token 但词表越长干扰越大 留够专有名词即可
_CONTEXT_MAX_CHARS = 1500


def build_asr_context(audit_type: str, texts: list[str]) -> str:
    """拼出语音识别的领域上下文

    考生口述里的「探针」「龈沟」「三弯端」不给上下文时常被转成同音错字
    转写错了 口述类评分项就会被当成没说 模型并不会去猜同音词
    :param texts: 评分点、口述关键词等 原样作为背景文本 不需要整理成词表
    """
    body = "\n".join(text.strip() for text in texts if text and text.strip())
    head = f"以下是口腔医学临床技能考核「{audit_type}」的考生操作口述，涉及的专业术语见下列评分要点。"
    context = f"{head}\n{body}" if body else head
    return context[:_CONTEXT_MAX_CHARS]


def _get_client() -> OpenAI:
    """获取 OpenAI 兼容的千问客户端"""
    return OpenAI(
        api_key=settings.QWEN_API_KEY_1,
        base_url=_BASE_URL,
    )


def _extract_text(content) -> str:
    """
    从模型返回中提取纯文本

    兼容三种形态: 字符串 / content 列表 / 带 text 字段的 dict
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()

    parts = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
    elif isinstance(content, dict):
        parts.append(str(content.get("text") or ""))
    return "".join(parts).strip()


def transcribe_audio(audio_bytes: bytes, mime: str = "audio/mpeg", context: str = "") -> str:
    """
    将一段音频转写为文本

    :param audio_bytes: 音频二进制内容
    :param mime: 音频 MIME 类型
    :param context: 领域上下文 通过 system 消息传给模型 提升专有名词识别率
    :return: 转写文本，无语音内容时返回空字符串
    """
    if not audio_bytes:
        return ""

    data = f"data:{mime};base64,{base64.b64encode(audio_bytes).decode()}"
    messages = []
    if context:
        messages.append({"role": "system", "content": [{"text": context}]})
    response = _get_client().chat.completions.create(
        model=settings.QWEN_ASR_MODEL,
        messages=[
            *messages,
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": data},
                    }
                ],
            }
        ],
    )
    return _extract_text(response.choices[0].message.content)


def transcribe_audio_file(audio_path: str, mime: Optional[str] = None, context: str = "") -> str:
    """
    转写本地音频文件

    :param audio_path: 音频文件路径
    :param mime: 音频 MIME 类型，默认按扩展名推断
    :param context: 领域上下文 见 build_asr_context
    :return: 转写文本
    """
    suffix = ("." + audio_path.rsplit(".", 1)[-1].lower()) if "." in audio_path else ""
    mime = mime or _MIME_BY_SUFFIX.get(suffix, "audio/mpeg")
    with open(audio_path, "rb") as f:
        return transcribe_audio(f.read(), mime=mime, context=context)
