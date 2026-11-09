# @Auther  : luoscorn
# @Time    : 2026/10/30
# @File    : media_evidence.py
"""视频证据抽取 — 下载视频后抽关键帧 + 按时间轴切片转写语音

打分链路原本把整段视频 URL 直接交给多模态模型 模型只能看画面
本模块补齐两类证据:
  1. 关键帧: 全片先抽低分辨率候选帧 再按「画面差异 + 时间分散」挑选关键帧
     换器械/换部位/动作变化最大的时刻优先入选 近乎相同的冗余帧自动丢弃
     选中的时刻再按原片比例重抽高清图 每帧带精确时间点 便于与语音转写对照
  2. 语音提及: 按时间轴切片抽音频交给 ASR 转写 操作者口述"已完成某步"同样算完成依据
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

# 切片参数
_MAX_SLICES = 8              # 最多切 8 片
_MIN_SLICE_SECONDS = 15      # 每片最短 15 秒 短片自动减少片数

# 关键帧参数
_MIN_FRAME_INTERVAL = 2.0    # 相邻关键帧最短间隔 短片不因预算富余而堆满冗余帧
_FRAME_QUALITY = 3           # ffmpeg -q:v 数值越大画质越低
_FRAME_WORKERS = 4           # 抽帧并发数 每帧一次 ffmpeg 快速 seek

# 候选帧参数: 低分辨率灰度解码 只用于挑选画面差异最大的时刻
_CANDIDATE_WIDTH = 64
_CANDIDATE_HEIGHT = 36
_CANDIDATE_FPS = 2.0         # 候选帧采样频率上限
_CANDIDATE_MAX_FRAMES = 600  # 候选帧总量上限 超出则自动降低采样频率
_CANDIDATE_MIN_RANGE = 12    # 像素极差低于该值视为黑场/纯色过渡帧 直接丢弃
_SPREAD_WEIGHT = 0.35        # 选帧打分中「时间分散」的权重 相机固定时靠它保证覆盖

# 音频参数 单声道 16k 32kbps 兼顾体积与识别率
_AUDIO_RATE = "16000"
_AUDIO_BITRATE = "32k"
_MAX_SLICE_AUDIO_SECONDS = 120  # 单片音频上限 避免单次转写超限

# 二次补证片段参数 重编码为小体积 mp4 直传模型
_CLIP_MAX_WIDTH = 720
_CLIP_CRF = 28

# 下载与 ffmpeg 超时
_DOWNLOAD_TIMEOUT = (10, 300)
_MAX_VIDEO_BYTES = 1024 ** 3
_FFMPEG_TIMEOUT = 120
_CANDIDATE_TIMEOUT = 600      # 全片候选帧解码 长视频耗时远高于单帧抽取
_ASR_WORKERS = 4

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_AUDIO_STREAM_RE = re.compile(r"Stream #\d+:\d+.*?: Audio:")
_VIDEO_SIZE_RE = re.compile(r",\s*(\d{2,5})x(\d{2,5})")


@dataclass
class Frame:
    """一张关键帧: 相对视频起点的精确秒数 + base64 jpeg"""
    seconds: float
    data: str

    @property
    def label(self) -> str:
        """帧时间标签，如 00:12"""
        return _fmt_time(self.seconds)


@dataclass
class VideoSlice:
    """一个视频切片: 时间段 + 该片关键帧 + 该片语音转写"""
    index: int
    start: float
    end: float
    frames: list[Frame] = field(default_factory=list)
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
    video_path: str = ""     # 本地视频文件 供二次补证切片 为空表示已清理

    @property
    def frame_count(self) -> int:
        return sum(len(s.frames) for s in self.slices)

    @property
    def has_evidence(self) -> bool:
        """至少要有画面或语音才能用于打分"""
        return self.frame_count > 0 or self.spoken_slices > 0

    def close(self) -> None:
        """清理本地视频临时目录 二次补证结束后必须调用"""
        if not self.video_path:
            return
        shutil.rmtree(os.path.dirname(self.video_path), ignore_errors=True)
        self.video_path = ""

    def frames_between(self, start: float, end: float, limit: int = 6) -> list["Frame"]:
        """取时间窗口内的关键帧 超过 limit 张时等距抽样

        二次补证要靠这些帧看清器械形态、持械分工、镜面朝向等细节
        视频交给模型后由它自己采样 分辨率与采样点都不可控 远不如直接给原图
        """
        picked = [
            frame
            for slice_ in self.slices
            for frame in slice_.frames
            if start <= frame.seconds <= end
        ]
        if limit <= 0 or len(picked) <= limit:
            return picked
        if limit == 1:
            return [picked[len(picked) // 2]]
        step = (len(picked) - 1) / (limit - 1)
        return [picked[round(index * step)] for index in range(limit)]

    def describe(self) -> str:
        """一行日志描述"""
        return (
            f"时长 {self.duration:.0f}s, 切片 {len(self.slices)} 段, "
            f"关键帧 {self.frame_count} 张, 语音转写 {self.spoken_slices} 段"
            + ("（视频无音轨）" if not self.has_audio else "")
        )

    def timeline_text(self) -> str:
        """按切片组织的时间轴文本 关键帧时刻与语音一一对应"""
        lines = [
            f"视频总时长 {self.duration:.0f} 秒，共切分为 {len(self.slices)} 个片段。",
            "关键帧按时间顺序贴图，标签中的 @mm:ss 即该帧的准确时刻，可与语音转写的时间段对照。",
        ]
        for s in self.slices:
            marks = "、".join(f.label for f in s.frames) or "（本段无关键帧）"
            lines.append(f"\n【片段{s.index + 1}】{s.label}")
            lines.append(f"关键帧时刻：{marks}")
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


def _probe(video_path: str) -> tuple[float, bool, int, int]:
    """
    探测视频时长、音轨与画面尺寸

    :return: (时长秒, 是否有音频流, 画面宽, 画面高)，尺寸解析失败时宽高为 0
    """
    cmd = [_ffmpeg_binary(), "-hide_banner", "-i", video_path]
    proc = subprocess.run(cmd, capture_output=True, timeout=_FFMPEG_TIMEOUT)
    output = proc.stderr.decode("utf-8", "ignore")

    match = _DURATION_RE.search(output)
    if match is None:
        raise RuntimeError("无法解析视频时长，可能不是有效的视频文件")
    hours, minutes, seconds = match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    size_match = _VIDEO_SIZE_RE.search(output)
    width = int(size_match.group(1)) if size_match else 0
    height = int(size_match.group(2)) if size_match else 0
    return duration, bool(_AUDIO_STREAM_RE.search(output)), width, height


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


def _sample_candidates(video_path: str, duration: float) -> list[tuple[float, bytes]]:
    """
    全片抽低分辨率灰度候选帧

    每秒最多 _CANDIDATE_FPS 帧 总量受 _CANDIDATE_MAX_FRAMES 约束
    黑场/纯色过渡帧直接丢弃 避免选帧时占到帧预算

    :return: [(相对秒, 灰度像素字节), ...]
    """
    interval = max(1.0 / _CANDIDATE_FPS, duration / _CANDIDATE_MAX_FRAMES)
    frame_size = _CANDIDATE_WIDTH * _CANDIDATE_HEIGHT
    cmd = [
        _ffmpeg_binary(), "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-an", "-sn",
        "-vf", f"fps={1.0 / interval:.4f},scale={_CANDIDATE_WIDTH}:{_CANDIDATE_HEIGHT},format=gray",
        "-f", "rawvideo", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=_CANDIDATE_TIMEOUT)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "ignore").strip()
        raise RuntimeError(f"候选帧采样失败: {stderr[-400:] or proc.returncode}")

    blob = proc.stdout
    candidates = []
    for i in range(len(blob) // frame_size):
        chunk = blob[i * frame_size:(i + 1) * frame_size]
        if max(chunk) - min(chunk) < _CANDIDATE_MIN_RANGE:
            continue
        candidates.append((round(i * interval, 3), chunk))
    return candidates


def _gap_fill_index(selected: list[int], times: list[float], max_gap: float) -> int:
    """找当前最大时间空档内的中间候选帧 空档都不超过 max_gap 时返回 -1

    纯按画面差异贪心会在动作密集的片段堆帧 而在镜头几乎不变的关键动作段留出空洞
    （口镜到位、支点建立这类姿态变化小但判定关键的时刻最容易落在空洞里）
    """
    ordered = sorted(selected, key=lambda idx: times[idx])
    bounds = [times[0], *[times[idx] for idx in ordered], times[-1]]
    best_lo, best_hi, best_gap = 0.0, 0.0, max_gap
    for lo, hi in zip(bounds, bounds[1:]):
        if hi - lo > best_gap:
            best_lo, best_hi, best_gap = lo, hi, hi - lo
    if best_gap <= max_gap:
        return -1

    mid = (best_lo + best_hi) / 2
    taken = set(selected)
    for idx in sorted(range(len(times)), key=lambda i: abs(times[i] - mid)):
        if idx not in taken and best_lo < times[idx] < best_hi:
            return idx
    return -1


def _select_timestamps(candidates: list[tuple[float, bytes]], count: int) -> list[float]:
    """
    从候选帧里挑出 count 个时刻: 优先画面差异大的帧 同时保证时间分散

    先补最大时间空档（不超过 span/count 的 1.5 倍） 再逐次选「与已选帧最小差异最大」的帧
    差异用像素字节的汉明距离衡量 差异接近时由时间间隔决定

    :return: 按时间升序排列的时刻
    """
    if count <= 0 or not candidates:
        return []
    if len(candidates) <= count:
        return [t for t, _ in candidates]

    pixels = [int.from_bytes(chunk, "big") for _, chunk in candidates]
    times = [t for t, _ in candidates]
    total_bits = len(candidates[0][1]) * 8
    span = max(1e-6, times[-1] - times[0])
    max_gap = span / count * 1.5

    selected = [len(candidates) // 2]   # 从时间中点起步 首帧落在片中
    chosen = {selected[0]}
    while len(selected) < count:
        gap_idx = _gap_fill_index(selected, times, max_gap)
        if gap_idx >= 0:
            chosen.add(gap_idx)
            selected.append(gap_idx)
            continue

        best_idx, best_score = -1, -1.0
        for idx in range(len(candidates)):
            if idx in chosen:
                continue
            diff = min((pixels[idx] ^ pixels[s]).bit_count() for s in selected) / total_bits
            spread = min(abs(times[idx] - times[s]) for s in selected) / span
            score = diff + _SPREAD_WEIGHT * spread
            if score > best_score:
                best_idx, best_score = idx, score
        if best_idx < 0:
            break
        chosen.add(best_idx)
        selected.append(best_idx)
    return [times[idx] for idx in sorted(selected)]


def _uniform_timestamps(duration: float, count: int) -> list[float]:
    """均匀铺开 count 个时刻 取每段中点 避开片头片尾"""
    count = max(1, count)
    step = duration / count
    return [round(step * (i + 0.5), 2) for i in range(count)]


def _frame_scale(width: int, height: int) -> Optional[str]:
    """关键帧输出尺寸: 长边压到 MEDIA_FRAME_WIDTH 以内 小片不放大"""
    limit = settings.MEDIA_FRAME_WIDTH
    if width <= 0 or height <= 0 or limit <= 0:
        return None
    scale = min(1.0, limit / max(width, height))
    return f"{max(2, int(width * scale) // 2 * 2)}:{max(2, int(height * scale) // 2 * 2)}"


def _extract_frame(video_path: str, seconds: float, out_path: str,
                   size: Optional[str]) -> Optional[Frame]:
    """按时刻抽一张关键帧 失败返回 None 不影响其余帧"""
    args = ["-ss", f"{seconds:.2f}", "-i", video_path, "-frames:v", "1"]
    if size:
        args += ["-vf", f"scale={size}"]
    args += ["-q:v", str(_FRAME_QUALITY), "-y", out_path]
    try:
        _run_ffmpeg(args)
    except Exception as exc:
        logger.warning("抽帧失败 @%.2fs: %s", seconds, exc)
        return None
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return None
    with open(out_path, "rb") as f:
        return Frame(seconds=seconds, data=base64.b64encode(f.read()).decode())


def _extract_frames(video_path: str, timestamps: list[float], save_dir: str,
                   width: int, height: int) -> list[Frame]:
    """并发抽取指定时刻的关键帧 返回按时间升序的帧列表"""
    if not timestamps:
        return []
    size = _frame_scale(width, height)
    jobs = [
        (ts, os.path.join(save_dir, f"frame{i}.jpg"))
        for i, ts in enumerate(timestamps)
    ]
    workers = max(1, min(_FRAME_WORKERS, len(jobs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(lambda job: _extract_frame(video_path, job[0], job[1], size), jobs))
    return sorted((f for f in frames if f is not None), key=lambda f: f.seconds)


def _pick_keyframes(video_path: str, duration: float, save_dir: str,
                   width: int, height: int) -> list[Frame]:
    """
    关键帧抽取入口: 算帧预算 → 采样候选帧 → 选时刻 → 重抽高清图

    帧预算 = min(MEDIA_MAX_FRAMES, 时长/最短帧间隔) 短片不堆满无信息量的帧
    """
    budget = max(1, min(settings.MEDIA_MAX_FRAMES, int(duration / _MIN_FRAME_INTERVAL) + 1))
    try:
        timestamps = _select_timestamps(_sample_candidates(video_path, duration), budget)
    except Exception as exc:
        logger.warning("候选帧采样失败 回退均匀抽帧: %s", exc)
        timestamps = []
    if not timestamps:
        timestamps = _uniform_timestamps(duration, budget)
    frames = _extract_frames(video_path, timestamps, save_dir, width, height)
    logger.info("关键帧选中时刻: %s", [f"{f.seconds:.1f}s" for f in frames])
    return frames


def _slice_of(slices: list[VideoSlice], seconds: float) -> Optional[VideoSlice]:
    """定位某个时刻所属的切片 越界时归入最后一段"""
    if not slices:
        return None
    for slice_item in slices:
        if slice_item.start <= seconds < slice_item.end:
            return slice_item
    return slices[-1]


def cut_clip(video_path: str, start: float, end: float, save_dir: str,
             name: str = "clip") -> Optional[str]:
    """
    切出一段短视频供二次补证直传模型

    重编码为 h264 + faststart 并去掉音轨 保证模型能直接拉流
    :return: 本地片段路径 失败返回 None
    """
    length = max(1.0, end - start)
    out_path = os.path.join(save_dir, f"{name}.mp4")
    try:
        _run_ffmpeg([
            "-ss", f"{max(0.0, start):.2f}",
            "-t", f"{length:.2f}",
            "-i", video_path,
            "-an", "-sn",
            "-vf", f"scale='min({_CLIP_MAX_WIDTH},iw)':-2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(_CLIP_CRF),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-y", out_path,
        ])
    except Exception as exc:
        logger.warning("补证片段 %s 生成失败: %s", name, exc)
        return None
    return out_path if os.path.exists(out_path) and os.path.getsize(out_path) > 0 else None


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
        duration, has_audio, width, height = _probe(video_path)
        if duration <= 0:
            raise RuntimeError("视频时长为 0")

        slices = [
            VideoSlice(index=i, start=start, end=end)
            for i, (start, end) in enumerate(_slice_ranges(duration, max_slices))
        ]

        # 关键帧: 全片按画面差异挑帧 再归入时刻所属的切片
        for frame in _pick_keyframes(video_path, duration, work_dir, width, height):
            owner = _slice_of(slices, frame.seconds)
            if owner is not None:
                owner.frames.append(frame)

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
            video_path=video_path,
        )
        if not evidence.has_evidence:
            raise RuntimeError("未能抽取到任何画面或语音证据")
        return evidence
    except Exception:
        # 失败时立即清理 成功时保留本地视频供二次补证切片 由调用方 close() 释放
        shutil.rmtree(work_dir, ignore_errors=True)
        raise


def frames_payload(evidence: VideoEvidence) -> list[tuple[str, str]]:
    """
    展平为 (标签, base64) 列表 供提示词按片段顺序插入图片

    标签带帧的准确时刻 便于模型把画面与语音转写的时间段对应起来
    """
    payload = []
    for slice_item in evidence.slices:
        for j, frame in enumerate(slice_item.frames):
            label = f"片段{slice_item.index + 1}({slice_item.label}) 第{j + 1}帧 @{frame.label}"
            payload.append((label, frame.data))
    return payload
