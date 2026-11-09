# @Auther  : luoscorn
# @Time    : 2026/11/9
# @File    : test_supplement.py
"""补证筛选与排序的回归用例 不需要数据库与网络

运行: ./venv/bin/python -m unittest discover -s tests -v
"""

import logging
import unittest

from services import supplement as s
from services.media_evidence import VideoEvidence


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
        located = {**_item("⑤", 0.0, 0.2, "支点缺失"), "time": "00:45"}
        self._with_slots(1)
        plans, skipped = s._plan_clips(self._evidence(), [located, self._unlocated()])
        numbers = [item["number"] for plan in plans for item in plan["items"]]
        self.assertEqual(numbers, ["④"])
        self.assertEqual([item["number"] for item in skipped], ["⑤"])


if __name__ == "__main__":
    unittest.main()
