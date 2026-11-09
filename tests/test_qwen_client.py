# @Auther  : luoscorn
# @Time    : 2026/11/4
# @File    : test_qwen_client.py
"""qwen_client 结果解析与兜底逻辑的回归用例 不需要数据库与网络

运行: ./venv/bin/python -m unittest discover -s tests -v
"""

import base64
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

    def test_oral_fields_are_rendered(self):
        item = RubricItem(
            dimension="基本", number="③", point="覆盖全口牙列", evidence="探针依次移动",
            score=0.1, oral_visual_weight="口述100%", oral_keywords="右上、左上\n左下、右下",
        )
        text = q._format_rubric([item])
        self.assertIn("口述视觉权重：口述100%", text)
        # 多行关键词压成单行 避免打散提示词排版
        self.assertIn("口述关键词：右上、左上 左下、右下", text)

    def test_oral_fields_absent_when_empty(self):
        text = q._format_rubric(_items())
        self.assertNotIn("口述视觉权重：", text)
        self.assertNotIn("口述关键词：", text)


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


class ReferenceImageTest(unittest.TestCase):
    """规范参考图: 每项首图 总张数受限 下载失败跳过"""

    def setUp(self):
        self._original_data_uri = q._image_data_uri
        q._reference_cache.clear()

    def tearDown(self):
        q._image_data_uri = self._original_data_uri
        q._reference_cache.clear()

    def _item(self, number: str, image_refs: str = "") -> RubricItem:
        return RubricItem(
            dimension="", number=number, point="要点", evidence="", score=1.0,
            image_refs=image_refs,
        )

    def test_only_first_image_per_item(self):
        q._image_data_uri = lambda ref: f"data:uri:{ref}"
        images = q._reference_images([
            self._item("①", "rule-admin/a/1.jpg\nrule-admin/a/2.jpg"),
            self._item("②", "https://cdn.example/2.png"),
            self._item("③"),
        ])
        self.assertEqual([label for label, _ in images], ["参考图 ①", "参考图 ②"])
        # 多图只取首张 无图项跳过
        self.assertEqual(images[0][1], "data:uri:rule-admin/a/1.jpg")
        self.assertEqual(images[1][1], "data:uri:https://cdn.example/2.png")

    def test_capped_by_max_images(self):
        q._image_data_uri = lambda ref: f"data:uri:{ref}"
        items = [self._item(str(idx), f"k{idx}.jpg") for idx in range(20)]
        self.assertEqual(len(q._reference_images(items)), q._MAX_REFERENCE_IMAGES)

    def test_download_failure_skips_image(self):
        q._image_data_uri = lambda ref: ""
        self.assertEqual(q._reference_images([self._item("①", "k.jpg")]), [])

    def test_builds_base64_and_hits_cache(self):
        calls = []

        def fake_url(oss_key, expired=None):
            calls.append(("sign", oss_key))
            return f"https://oss.example/{oss_key}"

        class _Resp:
            headers = {"content-type": "image/png"}
            content = b"png-bytes"

            def raise_for_status(self):
                pass

        def fake_get(url, timeout=None):
            calls.append(("get", url))
            return _Resp()

        original_url, original_get = q.get_object_url, q.requests.get
        q.get_object_url, q.requests.get = fake_url, fake_get
        try:
            uri = q._image_data_uri("rule-admin/a/1.png")
            self.assertTrue(uri.startswith("data:image/png;base64,"))
            self.assertEqual(
                uri.split(",", 1)[1], base64.b64encode(b"png-bytes").decode()
            )
            # 第二次命中进程内缓存 不重复签名与下载
            self.assertEqual(q._image_data_uri("rule-admin/a/1.png"), uri)
            self.assertEqual(len(calls), 2)
        finally:
            q.get_object_url, q.requests.get = original_url, original_get

    def test_returns_empty_on_download_error(self):
        original_get = q.requests.get

        def boom(*args, **kwargs):
            raise RuntimeError("network down")

        q.requests.get = boom
        try:
            self.assertEqual(q._image_data_uri("https://cdn.example/x.jpg"), "")
        finally:
            q.requests.get = original_get

    def test_reference_block_precedes_candidate_evidence(self):
        content = q._build_user_content(
            "http://example/video.mp4", None, "打分", [("参考图 ①", "https://x/1.jpg")]
        )
        self.assertIn("不是考生本人的操作证据", content[0]["text"])
        self.assertEqual(
            [entry["type"] for entry in content], ["text", "text", "image_url", "video_url", "text"]
        )
        self.assertEqual(content[-1]["text"], "打分")

    def test_no_reference_keeps_original_shape(self):
        content = q._build_user_content("http://example/video.mp4", None, "打分")
        self.assertEqual([entry["type"] for entry in content], ["video_url", "text"])


def _plain_items():
    """规则编号写成阿拉伯数字的情形 管理页录入的规则普遍是这种"""
    return [
        RubricItem(dimension="基本", number="1", point="选用探针", evidence="", score=0.2),
        RubricItem(dimension="基本", number="2", point="口镜牵拉", evidence="", score=0.3),
    ]


class PlainNumberRubricTest(unittest.TestCase):
    """规则编号为 1/2/3 时 模型返回 1 或 ① 都必须能对上 否则所有评分项被当成未知编号丢弃"""

    def setUp(self):
        self._original_call_model = q._call_model

    def tearDown(self):
        q._call_model = self._original_call_model

    def test_matches_arabic_model_number(self):
        result, missing = q._assemble_result(
            [{"number": "1", "score": 0.2}, {"number": "2", "score": 0.1}], "", _plain_items()
        )
        self.assertEqual(missing, [])
        self.assertEqual([i["number"] for i in result["items"]], ["1", "2"])
        self.assertEqual(result["total_score"], 0.3)

    def test_matches_circled_model_number(self):
        result, missing = q._assemble_result(
            [{"number": "①", "score": 0.2}, {"number": "（2）", "score": 0.3}], "", _plain_items()
        )
        self.assertEqual(missing, [])
        self.assertEqual([i["number"] for i in result["items"]], ["1", "2"])
        self.assertEqual(result["total_score"], 0.5)

    def test_fallback_applies_when_rule_numbers_are_arabic(self):
        q._call_model = lambda *a, **kw: '{"items":[]}'
        result = q._finalize_rubric_result(
            '{"items":[]}', _plain_items(), "测试项目", "http://example/video.mp4", None
        )
        self.assertEqual(len(result["items"]), 2)
        self.assertTrue(all(i["fallback"] for i in result["items"]))
        self.assertEqual([i["number"] for i in result["items"]], ["1", "2"])
        self.assertEqual(result["total_score"], round((0.2 + 0.3) * q._MISSING_SCORE_RATIO, 2))

    def test_supplement_accepts_circled_number(self):
        raw = '{"items":[{"number":"①","clear":true,"score":0.2,"reason":"片段中可见"}]}'
        out = q._parse_supplement(
            raw, [{"number": "1", "point": "选用探针", "full_score": 0.2, "score": 0.1}]
        )
        self.assertEqual([i["number"] for i in out], ["1"])
        self.assertTrue(out[0]["clear"])


class PromptPolicyTest(unittest.TestCase):
    def test_gradient_single_source_of_truth(self):
        # 档位百分比只允许出现在 _SCORING_POLICY 若其它片段复制档位 改档时必然漏改
        for fragment in (
            q._EVIDENCE_NOTE,
            q._RUBRIC_OUTPUT_FORMAT,
            q._SUPPLEMENT_POLICY,
            q._REFERENCE_NOTE,
        ):
            self.assertNotIn("85%~100%", fragment)
            self.assertNotIn("60%~85%", fragment)

    def test_summary_limit_is_consistent(self):
        self.assertIn(f"不超过 {q._SUMMARY_MAX_CHARS} 字", q._SCORING_POLICY)
        self.assertIn(f"不超过 {q._SUMMARY_MAX_CHARS} 字", q._RUBRIC_OUTPUT_FORMAT)


if __name__ == "__main__":
    unittest.main()
