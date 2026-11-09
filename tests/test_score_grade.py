"""等级判定回归 — 重点覆盖兜底项对档位的影响"""

import unittest

from utils.score_grade import GRADE_INCOMPLETE, evaluate_grade


def _items(count: int, fallback: int = 0) -> list[dict]:
    """造评分项 前 fallback 项标记为兜底"""
    return [
        {"number": f"{i + 1}", "score": 1, "full_score": 1, "fallback": i < fallback}
        for i in range(count)
    ]


class GradeBoundaryTest(unittest.TestCase):
    def test_rate_boundaries_are_strict(self):
        # 80% 不算 A，60% 不算 B
        self.assertEqual(evaluate_grade({"full_score": 10, "total_score": 8})["grade"], "B")
        self.assertEqual(evaluate_grade({"full_score": 10, "total_score": 6})["grade"], "C")
        self.assertEqual(evaluate_grade({"full_score": 10, "total_score": 8.01})["grade"], "A")

    def test_missing_scores_are_incomplete(self):
        self.assertEqual(evaluate_grade({})["grade"], GRADE_INCOMPLETE)
        self.assertEqual(evaluate_grade({"full_score": 0, "total_score": 0})["grade"], GRADE_INCOMPLETE)
        self.assertEqual(evaluate_grade({"full_score": 10})["grade"], GRADE_INCOMPLETE)

    def test_item_count_mismatch_is_incomplete(self):
        result = {"full_score": 10, "total_score": 10, "items": _items(3)}
        self.assertEqual(evaluate_grade(result, expected_items=9)["grade"], GRADE_INCOMPLETE)


class FallbackGradeTest(unittest.TestCase):
    def test_all_fallback_is_incomplete(self):
        items = _items(9, fallback=9)
        result = {"full_score": 9, "total_score": 5.4, "items": items}
        grade = evaluate_grade(result, expected_items=9)
        self.assertEqual(grade["grade"], GRADE_INCOMPLETE)
        self.assertIn("9 项中 9 项", grade["grade_reason"])
        # D 档仍保留百分率 前端不必再自己算一遍
        self.assertEqual(grade["score_rate"], 60.0)

    def test_half_fallback_is_incomplete(self):
        items = _items(9, fallback=5)
        result = {"full_score": 9, "total_score": 8.1, "items": items}
        grade = evaluate_grade(result, expected_items=9)
        self.assertEqual(grade["grade"], GRADE_INCOMPLETE)
        self.assertIn("兜底分", grade["grade_reason"])

    def test_few_fallback_keeps_grade_but_flags(self):
        items = _items(9, fallback=1)
        result = {"full_score": 9, "total_score": 9, "items": items}
        grade = evaluate_grade(result, expected_items=9)
        self.assertEqual(grade["grade"], "A")
        self.assertIn("1/9", grade["grade_reason"])

    def test_no_fallback_keeps_reason_empty(self):
        result = {"full_score": 9, "total_score": 9, "items": _items(9)}
        grade = evaluate_grade(result, expected_items=9)
        self.assertEqual(grade["grade_reason"], "")

    def test_api_path_without_expected_items(self):
        # 接口层现算不传规则项数 用结果自带 items 数当分母 同样能识别兜底占比
        result = {"full_score": 9, "total_score": 5.4, "items": _items(9, fallback=9)}
        self.assertEqual(evaluate_grade(result)["grade"], GRADE_INCOMPLETE)


if __name__ == "__main__":
    unittest.main()
