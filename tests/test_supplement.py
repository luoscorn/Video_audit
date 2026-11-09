# @Auther  : luoscorn
# @Time    : 2026/11/9
# @File    : test_supplement.py
"""补证筛选与排序的回归用例 不需要数据库与网络

运行: ./venv/bin/python -m unittest discover -s tests -v
"""

import logging
import unittest

from services import supplement as s


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


if __name__ == "__main__":
    unittest.main()
