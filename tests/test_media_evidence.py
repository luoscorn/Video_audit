# @Auther  : luoscorn
# @Time    : 2026/11/9
# @File    : test_media_evidence.py
"""关键帧选点的回归用例 只用合成候选帧 不需要网络与 ffmpeg

运行: ./venv/bin/python -m unittest discover -s tests -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from services import asr_client as a
from services import media_evidence as m
from services.media_evidence import (
    ASR_FAILED_MARK,
    Frame,
    VideoEvidence,
    VideoSlice,
    _audio_windows,
    _dhash,
    _gap_fill_index,
    _hash_bits,
    _MAX_SLICE_AUDIO_SECONDS,
    _select_timestamps,
)


def _candidates(spec: list[tuple[float, int]]) -> list[tuple[float, bytes]]:
    """按 (秒, 灰度值) 造候选帧 灰度值相同的帧视为画面完全相同"""
    return [(seconds, bytes([value]) * 4) for seconds, value in spec]


def _uniform(duration: float, step: float, value: int = 200) -> list[tuple[float, int]]:
    """等间隔同画面的候选帧"""
    count = int(duration / step) + 1
    return [(round(i * step, 3), value) for i in range(count)]


class FramesBetweenTest(unittest.TestCase):
    """二次补证取窗口内关键帧: 超出张数时等距抽样 静态帧供模型看清器械细节"""

    def _evidence(self) -> VideoEvidence:
        frames = [Frame(seconds=float(t), data=f"d{t}") for t in (0, 5, 10, 15, 20, 25, 30)]
        return VideoEvidence(
            duration=31.0,
            slices=[VideoSlice(index=1, start=0.0, end=31.0, frames=frames)],
            has_audio=True,
        )

    def test_keeps_only_frames_inside_window(self):
        self.assertEqual(
            [frame.seconds for frame in self._evidence().frames_between(8, 22, 6)],
            [10.0, 15.0, 20.0],
        )

    def test_uniform_sampling_when_over_limit(self):
        # 7 张取 4 张: 首尾必保留 中间等距
        self.assertEqual(
            [frame.seconds for frame in self._evidence().frames_between(0, 30, 4)],
            [0.0, 10.0, 20.0, 30.0],
        )

    def test_limit_one_takes_the_middle_frame(self):
        self.assertEqual([frame.seconds for frame in self._evidence().frames_between(0, 30, 1)], [15.0])

    def test_empty_window_returns_nothing(self):
        self.assertEqual(self._evidence().frames_between(40, 50, 6), [])


class GapFillIndexTest(unittest.TestCase):
    def test_picks_middle_of_largest_hole(self):
        # 已选 0s 与 30s 30 秒的空档超过阈值 应优先补中间
        times = [float(t) for t in range(0, 31)]
        idx = _gap_fill_index([0], times, max_gap=10.0)
        self.assertEqual(times[idx], 15.0)

    def test_returns_minus_one_when_holes_are_small(self):
        # 最大空档 5 秒未超过阈值 不再补帧 交回差异贪心
        times = [float(t) for t in range(0, 11, 5)]
        self.assertEqual(_gap_fill_index([0, 1, 2], times, max_gap=5.0), -1)

    def test_never_returns_selected_index(self):
        times = [float(t) for t in range(0, 31)]
        self.assertNotIn(_gap_fill_index([0, 15, 30], times, max_gap=4.0), {0, 15, 30})


class SelectTimestampsTest(unittest.TestCase):
    def test_static_tail_gets_frames(self):
        # 前段画面剧变 后段完全静止: 纯差异贪心会把帧全堆在前段
        # 补空档后必须覆盖到静止段
        spec = [(round(i * 0.5, 3), i % 7 * 20) for i in range(20)]
        spec += _uniform(40, 0.5)[20:]
        times = _select_timestamps(_candidates(spec), count=6)
        self.assertEqual(len(times), 6)
        self.assertTrue(any(t >= 30 for t in times), times)

    def test_frame_count_and_order(self):
        times = _select_timestamps(_candidates(_uniform(50, 0.5)), count=8)
        self.assertEqual(len(times), 8)
        self.assertEqual(times, sorted(times))

    def test_few_candidates_returned_as_is(self):
        spec = [(0.0, 10), (1.0, 20), (2.0, 30)]
        self.assertEqual(_select_timestamps(_candidates(spec), count=8), [0.0, 1.0, 2.0])


def _frame(fn) -> bytes:
    """按坐标造一张 64x36 灰度候选帧"""
    return bytes(max(0, min(255, int(fn(x, y)))) for y in range(36) for x in range(64))


class DhashTest(unittest.TestCase):
    """选帧差异度量: 整体明暗变化不算画面变化 结构变了才算"""

    def _diff(self, a: bytes, b: bytes) -> float:
        return (_dhash(a) ^ _dhash(b)).bit_count() / _hash_bits(a)

    def test_global_brightness_shift_is_not_a_change(self):
        base = _frame(lambda x, y: 60 + x * 2)
        brighter = _frame(lambda x, y: 61 + x * 2)
        self.assertEqual(self._diff(base, brighter), 0.0)

    def test_flat_noise_is_ignored(self):
        flat = _frame(lambda x, y: 127)
        noisy = _frame(lambda x, y: 127 + (x + y) % 2)
        self.assertEqual(self._diff(flat, noisy), 0.0)

    def test_moved_object_is_a_change(self):
        # 一根亮色器械从左侧挪到右侧
        left = _frame(lambda x, y: 220 if 10 <= x < 16 else 40)
        right = _frame(lambda x, y: 220 if 46 <= x < 52 else 40)
        self.assertGreater(self._diff(left, right), 0.05)


class AudioWindowsTest(unittest.TestCase):
    def test_short_slice_is_padded_on_both_sides(self):
        self.assertEqual(_audio_windows(15.0, 30.0, 60.0), [(14.0, 31.0)])

    def test_padding_is_clamped_to_video_bounds(self):
        self.assertEqual(_audio_windows(0.0, 15.0, 15.0), [(0.0, 15.0)])

    def test_long_slice_is_split_without_dropping_the_tail(self):
        windows = _audio_windows(0.0, 300.0, 300.0)
        self.assertGreater(len(windows), 1)
        self.assertEqual(windows[0][0], 0.0)
        self.assertEqual(windows[-1][1], 300.0)
        for start, end in windows:
            self.assertLessEqual(end - start, _MAX_SLICE_AUDIO_SECONDS)
        # 相邻子段互相重叠 边界上的话不会被切掉
        for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
            self.assertLess(next_start, prev_end)


class TranscriptLineTest(unittest.TestCase):
    def test_failed_asr_is_not_reported_as_silence(self):
        failed = VideoSlice(index=0, start=0.0, end=15.0, asr_failed=True)
        self.assertEqual(failed.transcript_line, ASR_FAILED_MARK)
        self.assertNotIn("无语音内容", failed.transcript_line)

    def test_partial_failure_keeps_text_and_flags_it(self):
        part = VideoSlice(index=0, start=0.0, end=15.0, transcript="探针探入龈沟", asr_failed=True)
        self.assertTrue(part.transcript_line.startswith("探针探入龈沟"))
        self.assertIn("语音转写失败", part.transcript_line)

    def test_silence_stays_silence(self):
        self.assertEqual(VideoSlice(index=0, start=0.0, end=15.0).transcript_line, "（该片段无语音内容）")


class AsrContextTest(unittest.TestCase):
    def test_context_contains_type_and_terms_without_blank_lines(self):
        text = a.build_asr_context("探诊检查", ["  龋洞  ", "", "  ", "龈沟 三弯端"])
        self.assertIn("探诊检查", text)
        self.assertIn("龋洞\n龈沟 三弯端", text)
        self.assertNotIn("\n\n", text)

    def test_context_is_bounded(self):
        self.assertLessEqual(len(a.build_asr_context("探诊检查", ["龋洞" * 2000])), a._CONTEXT_MAX_CHARS)

    def test_context_is_sent_as_system_background(self):
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=" 龋洞 "))])
        with patch.object(a, "_get_client") as client:
            create = client.return_value.chat.completions.create
            create.return_value = response
            self.assertEqual(a.transcribe_audio(b"audio", context="龋洞 龈沟"), "龋洞")
            messages = create.call_args.kwargs["messages"]
            self.assertEqual(messages[0], {"role": "system", "content": [{"text": "龋洞 龈沟"}]})
            self.assertEqual(messages[1]["content"][0]["type"], "input_audio")

    def test_empty_context_keeps_audio_only_input(self):
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))])
        with patch.object(a, "_get_client") as client:
            create = client.return_value.chat.completions.create
            create.return_value = response
            self.assertEqual(a.transcribe_audio(b"audio"), "")
            self.assertEqual([msg["role"] for msg in create.call_args.kwargs["messages"]], ["user"])


class AsrRetryTest(unittest.TestCase):
    def test_retry_preserves_context_and_recovers(self):
        with patch.object(m, "transcribe_audio_file", side_effect=[RuntimeError("限流"), " 龋洞 "]) as transcribe, \
                patch.object(m.time, "sleep") as sleep:
            self.assertEqual(m._safe_transcribe("audio.mp3", "龋洞 龈沟"), "龋洞")
            self.assertEqual(transcribe.call_count, m._ASR_ATTEMPTS)
            transcribe.assert_called_with("audio.mp3", context="龋洞 龈沟")
            sleep.assert_called_once_with(m._ASR_RETRY_DELAY)

    def test_exhausted_retry_is_failure_not_silence(self):
        with patch.object(m, "transcribe_audio_file", side_effect=RuntimeError("超时")) as transcribe, \
                patch.object(m.time, "sleep"):
            self.assertIsNone(m._safe_transcribe("audio.mp3"))
            self.assertEqual(transcribe.call_count, m._ASR_ATTEMPTS)

    def test_successful_silence_does_not_retry(self):
        with patch.object(m, "transcribe_audio_file", return_value="") as transcribe:
            self.assertEqual(m._safe_transcribe("audio.mp3"), "")
            transcribe.assert_called_once()

    def test_failed_audio_extraction_does_not_call_asr(self):
        with patch.object(m, "transcribe_audio_file") as transcribe:
            self.assertIsNone(m._safe_transcribe(None))
            transcribe.assert_not_called()

    def test_long_slice_keeps_successful_tail_after_partial_failure(self):
        slice_ = VideoSlice(index=0, start=0, end=300)
        with patch.object(m, "_cut_audio", return_value="audio.mp3") as cut, \
                patch.object(m, "_safe_transcribe", side_effect=["探针", None, "检查完毕"]) as transcribe:
            text, failed = m._transcribe_slice("video.mp4", slice_, 300, "unused", "领域术语")
            self.assertEqual(text, "探针 检查完毕")
            self.assertTrue(failed)
            self.assertEqual(cut.call_args_list[-1].args[2], 300)
            self.assertEqual(transcribe.call_args.args[1], "领域术语")


if __name__ == "__main__":
    unittest.main()
