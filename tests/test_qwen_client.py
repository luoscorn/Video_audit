# @Auther  : luoscorn
# @Time    : 2026/11/4
# @File    : test_qwen_client.py
"""qwen_client 结果解析与兜底逻辑的回归用例 不需要数据库与网络

运行: ./venv/bin/python -m unittest discover -s tests -v
"""

import logging
import unittest

from services import qwen_client as q
from utils.score_rubric import RubricItem


def setUpModule():
    # 用例会走补问/兜底分支 关闭告警日志 保持输出干净
    logging.getLogger("services.qwen_client").setLevel(logging.CRITICAL)


def _items():
    return [
        RubricItem(dimension="基本", number="①", point="支点", evidence="手指支撑", score=0.5),
        RubricItem(dimension="基本", number="②", point="角度", evidence="器械与牙长轴一致", score=1.0, remark="第一行\n第二行"),
    ]


class CleanReasonTest(unittest.TestCase):
    def test_strips_percent_and_arithmetic(self):
        cases = {
            "给60%即0.12分": "",
            "口述完整按85%~100%给分": "口述完整",
            "支点不稳 扣0.05分": "支点不稳",
            "覆盖约60%牙面": "覆盖约60%牙面",   # 事实描述不能被误删
        }
        for raw, expected in cases.items():
            self.assertEqual(q._clean_reason(raw), expected, raw)


class CleanSummaryTest(unittest.TestCase):
    def test_removes_list_marks(self):
        self.assertEqual(q._clean_summary("1. 操作基本规范。2. 支点不稳"), "操作基本规范。支点不稳")

    def test_truncates_to_limit(self):
        long_text = "整" * 200
        cleaned = q._clean_summary(long_text)
        self.assertLessEqual(len(cleaned), q._SUMMARY_MAX_CHARS + 1)
        self.assertTrue(cleaned.endswith("…"))

    def test_empty(self):
        self.assertEqual(q._clean_summary(None), "")


class NormalizeNumberTest(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(q._normalize_number("1"), "①")
        self.assertEqual(q._normalize_number("（2）"), "②")
        self.assertEqual(q._normalize_number(" ② "), "②")
        self.assertEqual(q._normalize_number("?"), "?")


class FormatRubricTest(unittest.TestCase):
    def test_remark_is_inlined(self):
        text = q._format_rubric(_items())
        self.assertIn("备注：第一行 第二行", text)
        self.assertIn("视频识别依据：手指支撑", text)


class AssembleResultTest(unittest.TestCase):
    def test_missing_items_are_reported_not_raised(self):
        result, missing = q._assemble_result(
            [{"number": "1", "score": 0.4, "reason": "支点稍松"}], "还行", _items()
        )
        self.assertEqual(missing, ["②"])
        self.assertEqual(result["total_score"], 0.4)
        self.assertEqual(len(result["deductions"]), 1)

    def test_unknown_and_duplicate_numbers_ignored(self):
        result, missing = q._assemble_result(
            [
                {"number": "①", "score": 0.5},
                {"number": "①", "score": 0.1},
                {"number": "⑨", "score": 0.9},
                {"oops": 1},
            ],
            "还行",
            _items(),
        )
        self.assertEqual(missing, ["②"])
        self.assertEqual([i["score"] for i in result["items"]], [0.5])

    def test_invalid_score_treated_as_missing(self):
        result, missing = q._assemble_result(
            [{"number": "①", "score": "n/a"}, {"number": "②", "score": 0.9}], "", _items()
        )
        self.assertEqual(missing, ["①"])
        self.assertEqual(result["total_score"], 0.9)

    def test_score_is_clamped(self):
        result, _ = q._assemble_result(
            [{"number": "①", "score": 9}, {"number": "②", "score": -3}], "", _items()
        )
        self.assertEqual([i["score"] for i in result["items"]], [0.5, 0.0])


class FinalizeResultTest(unittest.TestCase):
    def setUp(self):
        self._original_call_model = q._call_model
        self.calls = []

    def tearDown(self):
        q._call_model = self._original_call_model

    def _first_round(self):
        return '{"items":[{"number":"①","score":0.4,"reason":"支点稍松"}],"summary":"完成度一般"}'

    def test_missing_item_is_reasked_with_only_that_item(self):
        def fake(system_prompt, user_content, audit_type):
            self.calls.append(system_prompt)
            return '{"items":[{"number":"2","score":0.9,"reason":"角度与牙长轴一致"}]}'

        q._call_model = fake
        result = q._finalize_rubric_result(
            self._first_round(), _items(), "测试项目", "http://example/video.mp4", None
        )
        self.assertEqual(result["total_score"], 1.3)
        self.assertFalse(any(i["fallback"] for i in result["items"]))
        self.assertEqual(len(self.calls), 1)
        self.assertIn("② 角度", self.calls[0])
        self.assertNotIn("① 支点", self.calls[0])

    def test_still_missing_falls_back_to_lower_bound(self):
        q._call_model = lambda *a, **kw: '{"items":[]}'
        result = q._finalize_rubric_result(
            self._first_round(), _items(), "测试项目", "http://example/video.mp4", None
        )
        fallback = [i for i in result["items"] if i["fallback"]]
        self.assertEqual(len(fallback), 1)
        self.assertEqual(fallback[0]["number"], "②")
        self.assertEqual(fallback[0]["score"], round(1.0 * q._MISSING_SCORE_RATIO, 2))
        self.assertEqual(result["total_score"], round(0.4 + 0.6, 2))

    def test_no_missing_skips_repair(self):
        q._call_model = lambda *a, **kw: self.fail("不应发起补问")
        result = q._finalize_rubric_result(
            '{"items":[{"number":"①","score":0.5},{"number":"②","score":1.0}],"summary":"ok"}',
            _items(),
            "测试项目",
            "http://example/video.mp4",
            None,
        )
        self.assertEqual(result["total_score"], 1.5)
        self.assertEqual(result["deductions"], [])


class PromptPolicyTest(unittest.TestCase):
    def test_gradient_single_source_of_truth(self):
        # 档位百分比只允许出现在 _SCORING_POLICY 若其它片段复制档位 改档时必然漏改
        for fragment in (q._EVIDENCE_NOTE, q._RUBRIC_OUTPUT_FORMAT, q._SUPPLEMENT_POLICY):
            self.assertNotIn("85%~100%", fragment)
            self.assertNotIn("60%~85%", fragment)

    def test_summary_limit_is_consistent(self):
        self.assertIn(f"不超过 {q._SUMMARY_MAX_CHARS} 字", q._SCORING_POLICY)
        self.assertIn(f"不超过 {q._SUMMARY_MAX_CHARS} 字", q._RUBRIC_OUTPUT_FORMAT)


if __name__ == "__main__":
    unittest.main()
