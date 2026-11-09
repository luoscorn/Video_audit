# @Auther  : luoscorn
# @Time    : 2026/11/9
# @File    : test_media_evidence.py
"""关键帧选点的回归用例 只用合成候选帧 不需要网络与 ffmpeg

运行: ./venv/bin/python -m unittest discover -s tests -v
"""

import unittest

from services.media_evidence import Frame, VideoEvidence, VideoSlice, _gap_fill_index, _select_timestamps


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


if __name__ == "__main__":
    unittest.main()
