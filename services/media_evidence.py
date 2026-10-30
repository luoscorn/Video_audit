# @Auther  : luoscorn
# @Time    : 2026/10/30
# @File    : media_evidence.py
"""视频证据抽取 — 下载视频后按时间轴切片 逐片抽关键帧并转写该片语音

打分链路原本把整段视频 URL 直接交给多模态模型 模型只能看画面
本模块补齐两类证据:
  1. 视频切片 + 帧识别: 按时间轴均匀切片 每片取关键帧 覆盖全片操作过程
  2. 语音提及: 逐片抽音频交给 ASR 转写 操作者口述"已完成某步"同样算完成依据
任一步失败都只降级不阻断 由 audit_worker 回退为整段视频打分
"""

import base64
import logging
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import requests

from config import settings
from services.asr_client import transcribe_audio_file

logger = logging.getLogger(__name__)

# 切片与抽帧参数
_MAX_SLICES = 8              # 最多切 8 片
_MIN_SLICE_SECONDS = 15      # 每片最短 15 秒 短片自动减少片数
_FRAMES_PER_SLICE = 2        # 每片取 2 帧(25%/70% 位置)
_MAX_TOTAL_FRAMES = 16       # 单次请求图片上限 控制 token
_FRAME_WIDTH = 768
_FRAME_QUALITY = 6           # ffmpeg -q:v 数值越大画质越低

# 音频参数 单声道 16k 32kbps 兼顾体积与识别率
_AUDIO_RATE = "16000"
_AUDIO_BITRATE = "32k"
_MAX_SLICE_AUDIO_SECONDS = 120  # 单片音频上限 避免单次转写超限

# 下载与 ffmpeg 超时
_DOWNLOAD_TIMEOUT = (10, 300)
_MAX_VIDEO_BYTES = 1024 ** 3
_FFMPEG_TIMEOUT = 120
_ASR_WORKERS = 4

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_AUDIO_STREAM_RE = re.compile(r"Stream #\d+:\d+.*?: Audio:")


@dataclass
class VideoSlice:
    """一个视频切片: 时间段 + 关键帧 + 该片语音转写"""
    index: int
    start: float
    end: float
    frames: list[str] = field(default_factory=list)   # base64 编码的 jpeg
    transcript: str = ""

    @property
    def label(self) -> str:
        """切片时间标签，如 00:12-00:35"""
        return f"{_fmt_time(self.start)}-{_fmt_time(self.end)}"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class VideoEvidence:
    """整段视频的证据集合"""
    duration: float
    slices: list[VideoSlice]
    has_audio: bool
    spoken_slices: int = 0   # 有语音转写内容的切片数

    @property
    def frame_count(self) -> int:
        return sum(len(s.frames) for s in self.slices)

    @property
    def has_evidence(self) -> bool:
        """至少要有画面或语音才能用于打分"""
        return self.frame_count > 0 or self.spoken_slices > 0

    def describe(self) -> str:
        """一行日志描述"""
        return (
            f"时长 {self.duration:.0f}s, 切片 {len(self.slices)} 段, "
            f"关键帧 {self.frame_count} 张, 语音转写 {self.spoken_slices} 段"
            + ("（视频无音轨）" if not self.has_audio else "")
        )

    def timeline_text(self) -> str:
        """按切片组织的时间轴文本 画面帧与语音一一对应"""
        lines = [f"视频总时长 {self.duration:.0f} 秒，共切分为 {len(self.slices)} 个片段。"]
        for s in self.slices:
            lines.append(f"\n【片段{s.index + 1}】{s.label}")
            lines.append(f"语音转写：{s.transcript or '（该片段无语音内容）'}")
        return "\n".join(lines)

    def transcript_text(self) -> str:
        """仅语音转写的全文 用于日志留痕"""
        parts = [f"[{s.label}] {s.transcript}" for s in self.slices if s.transcript]
        return "\n".join(parts)


def _fmt_time(seconds: float) -> str:
    """秒转 mm:ss"""
    total = int(max(0, seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def _ffmpeg_binary() -> str:
    """
    定位 ffmpeg 可执行文件

    顺序: settings.FFMPEG_BINARY > PATH 中的 ffmpeg > imageio-ffmpeg 内置二进制
    """
    if settings.FFMPEG_BINARY:
        return settings.FFMPEG_BINARY

    found = shutil.which("ffmpeg")
    if found:
        return found

    import imageio_ffmpeg  # 延迟导入 未装该依赖时由上层回退整段视频

    return imageio_ffmpeg.get_ffmpeg_exe()


def _run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    """执行 ffmpeg 命令 失败时抛出带 stderr 摘要的异常"""
    cmd = [_ffmpeg_binary(), "-hide_banner", "-loglevel", "error", *args]
    proc = subprocess.run(cmd, capture_output=True, timeout=_FFMPEG_TIMEOUT)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "ignore").strip()
        raise RuntimeError(f"ffmpeg 执行失败: {stderr[-400:] or proc.returncode}")
    return proc


def _probe(video_path: str) -> tuple[float, bool]:
    """
    探测视频时长与是否含音轨

    :return: (时长秒, 是否有音频流)
    """
    cmd = [_ffmpeg_binary(), "-hide_banner", "-i", video_path]
    proc = subprocess.run(cmd, capture_output=True, timeout=_FFMPEG_TIMEOUT)
    output = proc.stderr.decode("utf-8", "ignore")

    match = _DURATION_RE.search(output)
    if match is None:
        raise RuntimeError("无法解析视频时长，可能不是有效的视频文件")
    hours, minutes, seconds = match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return duration, bool(_AUDIO_STREAM_RE.search(output))


def _download_video(video_url: str, save_dir: str) -> str:
    """
    下载预签名 URL 指向的视频到本地临时目录

    :return: 本地文件路径
    """
    parsed_suffix = os.path.splitext(video_url.split("?")[0])[1].lower()
    suffix = parsed_suffix if parsed_suffix and len(parsed_suffix) <= 5 else ".mp4"
    local_path = os.path.join(save_dir, f"source{suffix}")

    downloaded = 0
    with requests.get(video_url, stream=True, timeout=_DOWNLOAD_TIMEOUT) as resp:
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > _MAX_VIDEO_BYTES:
                    raise RuntimeError(f"视频超过 {_MAX_VIDEO_BYTES // 1024 ** 3}GB 上限")
                f.write(chunk)
    return local_path


def _slice_ranges(duration: float, max_slices: int) -> list[tuple[float, float]]:
    """按总时长均匀切分时间轴 片段数受最短片长与上限约束"""
    count = int(duration // _MIN_SLICE_SECONDS) or 1
    count = max(1, min(count, max_slices, _MAX_SLICES))
    step = duration / count
    return [(round(i * step, 2), round(min(duration, (i + 1) * step), 2)) for i in range(count)]


def _frame_offsets(slice_item: VideoSlice) -> list[float]:
    """切片内的取帧位置(相对秒) 取 25%/70% 处 避开起止的黑场与过渡帧"""
    if slice_item.duration <= 1.5 or _FRAMES_PER_SLICE <= 1:
        return [slice_item.duration / 2]
    ratios = [0.25, 0.7][:_FRAMES_PER_SLICE]
    return [round(slice_item.duration * r, 2) for r in ratios]


def _extract_frames(video_path: str, slice_item: VideoSlice, save_dir: str) -> list[str]:
    """抽取一个切片的关键帧并返回 base64 列表"""
    frames = []
    for j, offset in enumerate(_frame_offsets(slice_item)):
        out_path = os.path.join(save_dir, f"slice{slice_item.index}_frame{j}.jpg")
        _run_ffmpeg([
            "-ss", f"{slice_item.start + offset:.2f}",
            "-i", video_path,
            "-frames:v", "1",
            "-vf", f"scale={_FRAME_WIDTH}:-2",
            "-q:v", str(_FRAME_QUALITY),
            "-y", out_path,
        ])
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            with open(out_path, "rb") as f:
                frames.append(base64.b64encode(f.read()).decode())
    return frames


def _cut_audio(video_path: str, slice_item: VideoSlice, save_dir: str) -> Optional[str]:
    """切出切片对应的音频 转 mp3 单声道 16k"""
    length = min(slice_item.duration, _MAX_SLICE_AUDIO_SECONDS)
    out_path = os.path.join(save_dir, f"slice{slice_item.index}.mp3")
    try:
        _run_ffmpeg([
            "-ss", f"{slice_item.start:.2f}",
            "-t", f"{length:.2f}",
            "-i", video_path,
            "-vn", "-ac", "1", "-ar", _AUDIO_RATE, "-b:a", _AUDIO_BITRATE,
            "-y", out_path,
        ])
    except Exception as exc:
        logger.warning("切片 %d 音频抽取失败: %s", slice_item.index, exc)
        return None
    return out_path if os.path.exists(out_path) and os.path.getsize(out_path) > 0 else None


def _safe_transcribe(audio_path: Optional[str]) -> str:
    """转写单个切片音频 失败返回空串不阻断整体流程"""
    if audio_path is None:
        return ""
    try:
        return transcribe_audio_file(audio_path).strip()
    except Exception as exc:
        logger.warning("切片音频转写失败 %s: %s", audio_path, exc)
        return ""


def prepare_evidence(video_url: str, max_slices: int = _MAX_SLICES) -> VideoEvidence:
    """
    下载视频并抽取打分证据: 切片关键帧 + 逐片语音转写

    :param video_url: 视频预签名下载 URL
    :param max_slices: 最多切片数
    :return: VideoEvidence
    :raises RuntimeError: 下载/解码失败等无法产出证据的情况 由调用方回退
    """
    work_dir = tempfile.mkdtemp(prefix="video_audit_")
    try:
        video_path = _download_video(video_url, work_dir)
        duration, has_audio = _probe(video_path)
        if duration <= 0:
            raise RuntimeError("视频时长为 0")

        slices = [
            VideoSlice(index=i, start=start, end=end)
            for i, (start, end) in enumerate(_slice_ranges(duration, max_slices))
        ]

        # 关键帧: 逐片抽帧 总帧数受 _MAX_TOTAL_FRAMES 约束
        remaining = _MAX_TOTAL_FRAMES
        for slice_item in slices:
            if remaining <= 0:
                break
            slice_item.frames = _extract_frames(video_path, slice_item, work_dir)[:remaining]
            remaining -= len(slice_item.frames)

        # 语音: 逐片切音频后并发转写
        spoken = 0
        if has_audio:
            audio_paths = [_cut_audio(video_path, s, work_dir) for s in slices]
            with ThreadPoolExecutor(max_workers=min(_ASR_WORKERS, len(slices))) as pool:
                transcripts = list(pool.map(_safe_transcribe, audio_paths))
            for slice_item, text in zip(slices, transcripts):
                slice_item.transcript = text
                spoken += 1 if text else 0

        evidence = VideoEvidence(
            duration=duration,
            slices=slices,
            has_audio=has_audio,
            spoken_slices=spoken,
        )
        if not evidence.has_evidence:
            raise RuntimeError("未能抽取到任何画面或语音证据")
        return evidence
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def frames_payload(evidence: VideoEvidence) -> list[tuple[str, str]]:
    """
    展平为 (标签, base64) 列表 供提示词按片段顺序插入图片
    """
    payload = []
    for slice_item in evidence.slices:
        for j, frame in enumerate(slice_item.frames):
            payload.append((f"片段{slice_item.index + 1}({slice_item.label}) 画面{j + 1}", frame))
    return payload
