# @Auther  : luoscorn
# @Time    : 2026/11/9
# @File    : test_supplement.py
"""补证筛选与排序的回归用例 不需要数据库与网络

运行: ./venv/bin/python -m unittest discover -s tests -v
"""

import logging
import unittest

from services import media_evidence as m
from services import supplement as s
from services.media_evidence import Frame, VideoEvidence


def setUpModule():
    logging.getLogger("services.supplement").setLevel(logging.CRITICAL)


def _item(number: str, score: float, full: float, reason: str) -> dict:
    return {
        "number": number,
        "score": score,
        "full_score": full,
        "reason": reason,
    }


class NeedsSupplementTest(unittest.TestCase):
    def test_full_score_is_skipped(self):
        self.assertFalse(s._needs_supplement(_item("①", 0.2, 0.2, "符合要求")))

    def test_partial_score_enters_candidate_even_with_violation_wording(self):
        # 支点类被写成「不稳」只拿 50% 也必须进复核 否则就是这次的漏判
        self.assertTrue(s._needs_supplement(_item("⑤", 0.1, 0.2, "无名指支点不稳定")))

    def test_zero_score_enters_candidate(self):
        self.assertTrue(s._needs_supplement(_item("④", 0.0, 0.15, "器械使用错误")))


class PlanPriorityTest(unittest.TestCase):
    def test_evidence_limited_ranks_before_violation(self):
        limited = _item("②", 0.1, 0.2, "画面证据受限")
        violation = _item("⑦", 0.0, 0.2, "动作明显错误")
        self.assertLess(s._plan_priority(limited), s._plan_priority(violation))

    def test_bigger_deduction_first_within_group(self):
        small = _item("③", 0.05, 0.1, "证据受限")
        big = _item("②", 0.0, 0.2, "证据受限")
        self.assertLess(s._plan_priority(big), s._plan_priority(small))


class PlanClipsFallbackTest(unittest.TestCase):
    """模型判「整项未执行」时 time 为「无」 这类项也必须进复核 否则永远得不到纠正"""

    def _evidence(self, duration: float = 51.0) -> VideoEvidence:
        return VideoEvidence(duration=duration, slices=[], has_audio=True)

    def _unlocated(self, number: str = "④") -> dict:
        return {**_item(number, 0.0, 0.15, "全程未见口镜"), "time": "无"}

    def test_unlocated_item_gets_mid_video_window(self):
        plans, skipped = s._plan_clips(self._evidence(), [self._unlocated()])
        self.assertEqual(len(plans), 1)
        self.assertEqual(skipped, [])
        # 兜底窗口取片中 必须盖住主要操作时段
        self.assertLessEqual(plans[0]["start"], 28)
        self.assertGreaterEqual(plans[0]["end"], 31)

    def test_unlocated_and_located_both_scheduled(self):
        located = {**_item("⑤", 0.0, 0.2, "支点缺失"), "time": "00:22-00:25"}
        plans, skipped = s._plan_clips(self._evidence(), [self._unlocated(), located])
        self.assertEqual(skipped, [])
        numbers = [item["number"] for plan in plans for item in plan["items"]]
        self.assertCountEqual(numbers, ["④", "⑤"])

    def test_short_video_window_stays_in_range(self):
        plans, _ = s._plan_clips(self._evidence(10.0), [self._unlocated()])
        self.assertGreaterEqual(plans[0]["start"], 0)
        self.assertLessEqual(plans[0]["end"], 10.0)


class PlanClipsCapacityTest(unittest.TestCase):
    """片段名额有限: 重合窗口要复用 未定位项要先占名额"""

    def _evidence(self, duration: float = 51.0) -> VideoEvidence:
        return VideoEvidence(duration=duration, slices=[], has_audio=True)

    def _unlocated(self, number: str = "④") -> dict:
        return {**_item(number, 0.0, 0.15, "全程未见口镜"), "time": "无"}

    def _with_slots(self, slots: int):
        original = s.settings.MEDIA_SUPPLEMENT_CLIPS
        s.settings.MEDIA_SUPPLEMENT_CLIPS = slots
        self.addCleanup(setattr, s.settings, "MEDIA_SUPPLEMENT_CLIPS", original)

    def test_overlapping_windows_share_one_clip(self):
        # 三个项的证据时刻挤在 20~23 秒: 为 15 秒上限切出三段几乎一样的视频
        # 会白白吃掉名额 让后面的项无处复核
        items = [
            {**_item("②", 0.0, 0.2, "未见牵拉"), "time": "00:20"},
            {**_item("⑤", 0.16, 0.2, "证据受限"), "time": "00:22"},
            {**_item("⑦", 0.16, 0.2, "证据受限"), "time": "00:23"},
        ]
        self._with_slots(1)
        plans, skipped = s._plan_clips(self._evidence(), items)
        self.assertEqual(len(plans), 1)
        self.assertEqual(skipped, [])
        self.assertEqual(len(plans[0]["items"]), 3)

    def test_unlocated_takes_the_slot_before_located(self):
        # 判「整项未执行」的项判错就是把满分判成 0 名额紧张时必须先给它
        # 用 0.16 而非 0 分: 0 分会触发片尾宽窗口 与兜底窗口重叠后复用同一片段
        located = {**_item("⑤", 0.16, 0.2, "支点证据受限"), "time": "00:45"}
        self._with_slots(1)
        plans, skipped = s._plan_clips(self._evidence(), [located, self._unlocated()])
        numbers = [item["number"] for plan in plans for item in plan["items"]]
        self.assertEqual(numbers, ["④"])
        self.assertEqual([item["number"] for item in skipped], ["⑤"])

    def test_zero_score_item_requests_full_frames(self):
        # 判 0 分时模型给的 time 不可靠 支点这类姿态可能到片尾才出现
        # 窗口只能看十八秒 必须改用全片关键帧才能看到整段
        item = {**_item("⑤", 0.0, 0.2, "全程未见支点"), "time": "00:22"}
        plans, _ = s._plan_clips(self._evidence(), [item])
        self.assertTrue(plans[0]["dense"])

    def test_partial_score_item_keeps_window_frames(self):
        item = {**_item("⑦", 0.16, 0.2, "证据受限"), "time": "00:22"}
        plans, _ = s._plan_clips(self._evidence(), [item])
        self.assertFalse(plans[0]["dense"])


class FrameTimestampsTest(unittest.TestCase):
    """复核窗口内重抽帧的时刻计算 直接决定模型能看到多密的动作"""

    def test_one_second_step_covers_window(self):
        # 六秒窗口按一秒间隔出七张 末帧压在终点内侧
        stamps = m.frame_timestamps(10.0, 16.0, 1.0)
        self.assertEqual(len(stamps), 7)
        self.assertEqual(stamps[:6], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        self.assertTrue(15.0 < stamps[-1] < 16.0)

    def test_limit_widens_step_instead_of_dropping_tail(self):
        # 帧数受限时放宽间隔 不能只抽前几秒把后半段的换端动作漏掉
        stamps = m.frame_timestamps(0.0, 20.0, 1.0, limit=4)
        self.assertEqual(len(stamps), 4)
        self.assertGreaterEqual(stamps[-1], 14.0)

    def test_last_stamp_never_reaches_end(self):
        # 末帧落在窗口终点会被 seek 到下一段 内容错位
        stamps = m.frame_timestamps(0.0, 3.0, 1.0)
        self.assertTrue(all(stamp < 3.0 for stamp in stamps))

    def test_empty_window(self):
        self.assertEqual(m.frame_timestamps(5.0, 5.0, 1.0), [])

    def test_no_video_file_returns_empty(self):
        evidence = VideoEvidence(duration=30, slices=[], has_audio=False)
        self.assertEqual(evidence.extract_frames_at(0, 10), [])


class MergeFramesTest(unittest.TestCase):
    """0 分项要全片帧与窗口密帧一起给 重复时刻只留一张"""

    def test_near_duplicates_dropped(self):
        base = [Frame(seconds=10.0, data="a"), Frame(seconds=12.0, data="b")]
        extra = [Frame(seconds=10.2, data="c"), Frame(seconds=11.0, data="d")]
        merged = s._merge_frames(base, extra)
        self.assertEqual([frame.data for frame in merged], ["a", "d", "b"])


if __name__ == "__main__":
    unittest.main()
