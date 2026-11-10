# @Auther  : luoscorn
# @Time    : 2026/11/4
# @File    : test_qwen_client.py
"""qwen_client 结果解析与兜底逻辑的回归用例 不需要数据库与网络

运行: ./venv/bin/python -m unittest discover -s tests -v
"""

import base64
import io
import json
import logging
import unittest

from PIL import Image

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

    def test_reasoning_tail_is_cut(self):
        # 实测模型会把自问自答与再核算写进 reason 前面是判定依据后面全是思考过程
        cases = {
            "口述提及“无名指作为支点”；画面@00:26可见右手无名指抵在模型下颌基座，视觉达成；两项均满足，但按规则应直接给满分0.2？不，规则明确“口述20%，视觉80%”为权重分配":
                "口述提及“无名指作为支点”；画面@00:26可见右手无名指抵在模型下颌基座，视觉达成；两项均满足",
            "口述明确“三弯端检查邻面”；画面@00:29可见探针三弯端伸入邻间隙，故应得0.2。":
                "口述明确“三弯端检查邻面”；画面@00:29可见探针三弯端伸入邻间隙",
            "口述说明大弯端用法；画面@00:26可见大弯端接触咬合面 再核: 仅一次属不完整":
                "口述说明大弯端用法；画面@00:26可见大弯端接触咬合面",
            "视觉占80%达成，口述占20%达成，合计达标，给满分0.2分注：实际得分为0.16":
                "视觉占80%达成，口述占20%达成，合计达标",
        }
        for raw, expected in cases.items():
            self.assertEqual(q._clean_reason(raw), expected, raw)

    def test_plain_reason_untouched(self):
        text = "画面@00:26可见左手持口镜牵拉左侧口角，镜面到位"
        self.assertEqual(q._clean_reason(text), text)

    def test_overlong_reason_truncated(self):
        cleaned = q._clean_reason("画" * 300)
        self.assertLessEqual(len(cleaned), q._REASON_MAX_CHARS + 1)
        self.assertTrue(cleaned.endswith("…"))

    def test_pure_reasoning_becomes_empty(self):
        # 整段都是思考过程时不能退回原文 否则脏文本会当日判定依据入库
        self.assertEqual(q._clean_reason("但按规则应直接给满分0.2，故应得0.2"), "")


class AlignScoreWithReasonTest(unittest.TestCase):
    """reason 已判定达成 分数却按份额打折时必须把分数提回满分"""

    def test_reason_saying_all_evidence_done_promotes(self):
        cases = [
            "视觉占80%达成、口述20%补充，合计达标",
            "视觉80%+口述20%达成",
            "画面与口述两项均满足",
            # 实测原句: 结论落在句尾 前面只是证据叙述
            "口述占20%、视觉占80%，画面清晰可见三弯端用于邻面，达标",
            "口述与视觉均支持，达标",
            "口述完整，视觉可见器械特征，达标",
            # 实测原句: 结尾直接写“满分”也是结论 不应继续打折
            "握持姿势稳定，探针控制良好，无滑脱抖动，视觉100%规范，满分",
        ]
        for reason in cases:
            score, promoted = q._align_score_with_reason(0.16, reason, 0.2)
            self.assertTrue(promoted, reason)
            self.assertEqual(score, 0.2, reason)

    def test_negative_conclusion_blocks_promotion(self):
        # “合计达标但支点不稳”不能因为前半句就把扣分抹掉
        score, promoted = q._align_score_with_reason(0.16, "合计达标，但支点不稳", 0.2)
        self.assertFalse(promoted)
        self.assertEqual(score, 0.16)

    def test_partial_wording_is_untouched(self):
        for reason in (
            "动作可见但不完整，未覆盖多面",
            "该端仅单次使用",
            "部分牙位达标",
            "基本达成",
            "未达满分",
        ):
            score, promoted = q._align_score_with_reason(0.16, reason, 0.2)
            self.assertFalse(promoted, reason)
            self.assertEqual(score, 0.16)

    def test_full_score_and_empty_reason_short_circuit(self):
        self.assertEqual(q._align_score_with_reason(0.2, "合计达标", 0.2), (0.2, False))
        self.assertEqual(q._align_score_with_reason(0.16, "", 0.2), (0.16, False))

    def test_assemble_applies_promotion(self):
        result, _ = q._assemble_result(
            [
                {"number": "1", "score": 0.4, "reason": "画面达成、口述也说清，合计达标"},
                {"number": "2", "score": 1.0, "reason": "角度一致"},
            ],
            "",
            _items(),
        )
        self.assertEqual(result["total_score"], 1.5)
        self.assertEqual(result["deductions"], [])

    def test_supplement_promotion_respects_clear_flag(self):
        targets = [{"number": "1", "full_score": 0.5}]
        done = q._parse_supplement(
            '{"items": [{"number": "1", "clear": true, "score": 0.4, "reason": "画面与口述均达成"}]}',
            targets,
        )
        self.assertEqual(done[0]["score"], 0.5)
        # clear=false 表示证据仍不足 这一档本来就保留原分 不能因措辞抬分
        unsure = q._parse_supplement(
            '{"items": [{"number": "1", "clear": false, "score": 0.4, "reason": "画面与口述均达成"}]}',
            targets,
        )
        self.assertEqual(unsure[0]["score"], 0.4)

    def test_supplement_full_mark_with_failing_reason_is_not_trusted(self):
        # 实测 126: clear=true、满分 reason 却写「分工不符要求」 合并后分数升了 reason 仍是上一轮的「未执行」
        out = q._parse_supplement(
            '{"items": [{"number": "1", "clear": true, "score": 0.5, '
            '"reason": "口镜仍在托盘内未被拿起，两手未各持口镜与探针，分工不符要求"}]}',
            [{"number": "1", "full_score": 0.5}],
        )
        self.assertFalse(out[0]["clear"])
        self.assertEqual(out[0]["score"], 0.0)

    def test_supplement_full_mark_with_missing_hand_assignment_is_not_trusted(self):
        out = q._parse_supplement(
            json.dumps({"items": [{"number": "1", "clear": True, "score": 0.5, "reason":
                "托盘内可见口镜，但全程未见左手持口镜、右手持探针的分工状态，实际为单手操作探针、另一手持模型。"}]}),
            [{"number": "1", "full_score": 0.5}],
        )
        self.assertFalse(out[0]["clear"])
        self.assertEqual(out[0]["score"], 0.0)

    def test_supplement_clear_without_reason_is_not_trusted(self):
        out = q._parse_supplement(
            '{"items": [{"number": "1", "clear": true, "score": 0.5, "reason": ""}]}',
            [{"number": "1", "full_score": 0.5}],
        )
        self.assertFalse(out[0]["clear"])
        self.assertEqual(out[0]["score"], 0.0)

    def test_positive_negations_are_not_verdicts(self):
        # 「未见戳刺」是正面描述 「画面无法确认但口述完整」是第 1 档 都不算与满分矛盾
        self.assertFalse(q._contradicts_full_mark(0.1, "@00:26 动作轻柔 未见戳刺牙龈", 0.1))
        self.assertFalse(q._contradicts_full_mark(0.2, "画面无法确认 口述完整交代顺序", 0.2))
        # 非满分时写缺陷是正常的部分分档
        self.assertFalse(q._contradicts_full_mark(0.1, "支点未满足全程稳定", 0.2))
        self.assertTrue(q._contradicts_full_mark(0.2, "支点悬空，未达标", 0.2))
        self.assertTrue(q._contradicts_full_mark(0.15, "右手持探针，左手未持口镜，仅稳定模型", 0.15))


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

    def test_group_titles_can_be_suppressed(self):
        # 只判部分项时维度标题会被模型当成评分项 实测它直接用标题当 number 回结果
        items = [
            RubricItem(dimension="器械", number="2", point="口镜牵拉", evidence="", score=0.2),
            RubricItem(dimension="支点", number="5", point="无名指支点", evidence="", score=0.2),
        ]
        text = q._format_rubric(items, group=False)
        self.assertTrue(text.startswith("2 口镜牵拉"))
        self.assertNotIn("\n器械", text)
        self.assertNotIn("\n支点", text)


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
        def fake(system_prompt, user_content, audit_type, temperature=0.1):
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


class VoteDeductedItemsTest(unittest.TestCase):
    """扣分项投票: 只对已扣分项重判 取中位数 样本不齐不凑平均"""

    def setUp(self):
        self._original_judge = q._judge_items

    def tearDown(self):
        q._judge_items = self._original_judge

    def _result(self, first_score: float) -> dict:
        return {
            "items": [
                {"number": "①", "score": first_score, "reason": "无名指悬空", "time": "无"},
                {"number": "②", "score": 1.0, "reason": "角度一致", "time": "@00:10"},
            ],
            "summary": "完成度一般",
        }

    def _mock_scores(self, scores: list):
        """每次调用只回第一项 返回的分数按顺序取"""
        seq = iter(scores)
        last = scores[-1]

        def fake(*args, **kwargs):
            return [{"number": "①", "score": next(seq, last), "reason": "抵在模型下颌基座", "time": "@00:26"}]

        q._judge_items = fake

    def test_full_score_skips_voting(self):
        q._judge_items = lambda *a, **kw: self.fail("满分项不应发起投票")
        self.assertIsNone(
            q.vote_deducted_items("http://example/video.mp4", _items(), self._result(0.5), "测试项目")
        )

    def test_majority_restores_full_score(self):
        # 首轮 0 分 各轮独立判定都认为达标 中位数把扣分推翻
        self._mock_scores([0.5] * len(q._vote_temperatures()))
        voted = q.vote_deducted_items(
            "http://example/video.mp4", _items(), self._result(0.0), "测试项目"
        )
        self.assertIsNotNone(voted)
        self.assertEqual(voted["total_score"], 1.5)
        self.assertEqual(voted["deductions"], [])
        self.assertEqual(voted["vote"]["changed"][0]["to"], 0.5)

    def test_minority_high_score_keeps_deduction(self):
        # 只有一次说达标 中位数仍是 0 不能把偶发高分当真相
        rounds = len(q._vote_temperatures())
        self._mock_scores([0.5] + [0.0] * (rounds - 1))
        self.assertIsNone(
            q.vote_deducted_items("http://example/video.mp4", _items(), self._result(0.0), "测试项目")
        )

    def test_one_dissent_does_not_block_full_score(self):
        # 首轮 0.16 四轮里有三次 0.2 中位数应拉到 0.2 而不是被第一票钉住
        self._mock_scores([0.2, 0.2, 0.16, 0.2])
        voted = q.vote_deducted_items(
            "http://example/video.mp4", _items(), self._result(0.16), "测试项目"
        )
        self.assertIsNotNone(voted)
        self.assertEqual(voted["vote"]["changed"][0]["to"], 0.2)

    def test_vote_ticket_reason_alignment_promotes(self):
        # 四轮 reason 都写“达标”却都填 0.16 对齐后每票都是满分 不能继续钉在打折值
        q._judge_items = lambda *a, **kw: [
            {"number": "①", "score": 0.16, "reason": "右手无名指抵住模型下颌基座，达标", "time": "@00:26"}
        ]
        voted = q.vote_deducted_items(
            "http://example/video.mp4", _items(), self._result(0.16), "测试项目"
        )
        self.assertIsNotNone(voted)
        self.assertEqual(voted["vote"]["changed"][0]["to"], 0.5)

    def test_one_failed_round_still_decides(self):
        # 一轮失败还剩四票 去掉首轮那票凑成奇数 前面几次调用不白花
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("模型超时")
            return [{"number": "①", "score": 0.5, "reason": "达标", "time": "@00:26"}]

        q._judge_items = flaky
        voted = q.vote_deducted_items("http://example/video.mp4", _items(), self._result(0.0), "测试项目")
        self.assertIsNotNone(voted)
        self.assertEqual(voted["vote"]["changed"][0]["to"], 0.5)

    def test_too_few_samples_keeps_first_round(self):
        # 只剩首轮加一次重判 两票取中位就是押宝 维持首轮
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("模型超时")
            return [{"number": "①", "score": 0.5, "reason": "达标", "time": "@00:26"}]

        q._judge_items = flaky
        self.assertIsNone(
            q.vote_deducted_items("http://example/video.mp4", _items(), self._result(0.0), "测试项目")
        )

    def test_full_score_without_evidence_is_revoted(self):
        # 满分项 reason 只写「规范」 没有时刻也没有口述引文 多数重判认为不达标才扣
        q._judge_items = lambda targets, *a, **kw: [
            {"number": "①", "score": 0.2, "reason": "@00:26 无名指悬空", "time": "@00:26"}
        ]
        result = self._result(0.5)
        result["items"][0].update({"reason": "规范", "time": "无"})
        voted = q.vote_deducted_items("http://example/video.mp4", _items(), result, "测试项目")
        self.assertIsNotNone(voted)
        self.assertEqual(voted["vote"]["changed"][0]["to"], 0.2)

    def test_full_score_with_failing_reason_is_revoted(self):
        # 满分但 reason 写「未执行」 即使带了时刻也要重判
        q._judge_items = lambda targets, *a, **kw: [
            {"number": "①", "score": 0.0, "reason": "@00:26 无名指悬空", "time": "@00:26"}
        ]
        result = self._result(0.5)
        result["items"][0].update({"reason": "@00:26 无名指未建立支点，未执行", "time": "@00:26"})
        voted = q.vote_deducted_items("http://example/video.mp4", _items(), result, "测试项目")
        self.assertIsNotNone(voted)
        self.assertEqual(voted["vote"]["changed"][0]["to"], 0.0)

    def test_contradictory_votes_are_not_counted(self):
        # 满分票的结论是否定时不计入 剩下的票不足三票时维持首轮
        for reason in ("两手分工不符要求", "右手持探针，左手未持口镜，仅稳定模型"):
            with self.subTest(reason=reason):
                q._judge_items = lambda targets, *a, **kw: [
                    {"number": "①", "score": 0.5, "reason": reason, "time": "无"}
                ]
                self.assertIsNone(
                    q.vote_deducted_items("http://example/video.mp4", _items(), self._result(0.0), "测试项目")
                )

    def test_full_score_with_evidence_is_not_revoted(self):
        q._judge_items = lambda *a, **kw: self.fail("有据可查的满分项不应重判")
        result = self._result(0.5)
        result["items"][0].update({"reason": "口述“无名指支点”", "time": "无"})
        self.assertIsNone(
            q.vote_deducted_items("http://example/video.mp4", _items(), result, "测试项目")
        )

    def test_fallback_items_are_excluded(self):
        # 兜底项本来就没有可信判定 投票只会把它变成另一个拍脑袋的数
        q._judge_items = lambda *a, **kw: self.fail("兜底项不应参与投票")
        result = self._result(0.5)
        result["items"][0].update({"score": 0.0, "fallback": True})
        self.assertIsNone(
            q.vote_deducted_items("http://example/video.mp4", _items(), result, "测试项目")
        )


class ConflictArbitrationTest(unittest.TestCase):
    """冲突仲裁: 检出各项对同一事实的相反认定后用同一份证据重判"""

    def setUp(self):
        self._detect = q._detect_conflicts
        self._judge = q._judge_items
        self._call = q._call_model

    def tearDown(self):
        q._detect_conflicts = self._detect
        q._judge_items = self._judge
        q._call_model = self._call

    def _result(self) -> dict:
        return {
            "items": [
                {"number": "①", "point": "支点", "score": 0.0, "full_score": 0.5,
                 "reason": "全程未见口镜使用", "time": "无"},
                {"number": "②", "point": "角度", "score": 1.0, "full_score": 1.0,
                 "reason": "左手持口镜", "time": "@00:21"},
            ],
            "summary": "完成度一般",
        }

    def test_no_conflict_skips_arbitration(self):
        q._detect_conflicts = lambda *a, **kw: []
        q._judge_items = lambda *a, **kw: self.fail("无冲突不应发起仲裁")
        self.assertIsNone(q.arbitrate_conflicts("u", _items(), self._result(), "测试项目"))

    def test_conflict_group_is_rejudged_and_totals_rebuilt(self):
        q._detect_conflicts = lambda *a, **kw: [
            {"fact": "左手是否持口镜", "numbers": {"①", "②"}}
        ]
        q._judge_items = lambda targets, *a, **kw: [
            {"number": "①", "score": 0.5, "reason": "左手持口镜牵拉口角，达标", "time": "@00:22"},
            {"number": "②", "score": 0.8, "reason": "探针角度偏向颊侧", "time": "@00:21"},
        ]
        # 两项都是往上走 仲裁只负责救回被判 0 的项 不做降级
        result = self._result()
        result["items"][1]["score"] = 0.6
        voted = q.arbitrate_conflicts("u", _items(), result, "测试项目")
        self.assertIsNotNone(voted)
        self.assertEqual(voted["total_score"], 1.3)
        changes = {c["number"]: c for c in voted["conflicts"]["changed"]}
        self.assertEqual(changes["①"]["to"], 0.5)
        self.assertEqual(changes["①"]["fact"], "左手是否持口镜")

    def test_unchanged_verdict_returns_none(self):
        # 重判后结论本来是自洽的 不能凭空产出一份新结果
        q._detect_conflicts = lambda *a, **kw: [{"fact": "左手是否持口镜", "numbers": {"①", "②"}}]
        q._judge_items = lambda targets, *a, **kw: [
            {"number": "①", "score": 0.0, "reason": "确实未见", "time": "无"},
            {"number": "②", "score": 1.0, "reason": "左手持口镜", "time": "@00:21"},
        ]
        self.assertIsNone(q.arbitrate_conflicts("u", _items(), self._result(), "测试项目"))

    def test_raise_with_failing_reason_is_rejected(self):
        q._detect_conflicts = lambda *a, **kw: [{"fact": "左手是否持口镜", "numbers": {"①", "②"}}]
        q._judge_items = lambda targets, *a, **kw: [
            {"number": "①", "score": 0.5, "reason": "口镜未被拿起，不符要求", "time": "无"},
            {"number": "②", "score": 1.0, "reason": "左手持口镜", "time": "@00:21"},
        ]
        self.assertIsNone(q.arbitrate_conflicts("u", _items(), self._result(), "测试项目"))

    def test_arbitration_failure_keeps_result(self):
        q._detect_conflicts = lambda *a, **kw: [{"fact": "左手是否持口镜", "numbers": {"①", "②"}}]

        def boom(*a, **kw):
            raise RuntimeError("模型超时")

        q._judge_items = boom
        self.assertIsNone(q.arbitrate_conflicts("u", _items(), self._result(), "测试项目"))

    def test_arbitration_never_downgrades_a_scored_item(self):
        # 补证用窗口密帧认出左手持口镜给到 0.8 仲裁拿首轮稀疏关键帧重判不能压回 0
        q._detect_conflicts = lambda *a, **kw: [{"fact": "左手是否持口镜", "numbers": {"①", "②"}}]
        q._judge_items = lambda targets, *a, **kw: [
            {"number": "①", "score": 0.5, "reason": "无名指抵住基座，达标", "time": "@00:26"},
            {"number": "②", "score": 0.0, "reason": "未见口镜", "time": "无"},
        ]
        result = self._result()
        result["items"][1]["score"] = 0.8
        voted = q.arbitrate_conflicts("u", _items(), result, "测试项目")
        self.assertIsNotNone(voted)
        scores = {i["number"]: i["score"] for i in voted["items"]}
        self.assertEqual(scores["①"], 0.5)
        self.assertEqual(scores["②"], 0.8)
        self.assertEqual([c["number"] for c in voted["conflicts"]["changed"]], ["①"])

    def test_detect_normalizes_numbers_caps_groups_and_sends_text_only(self):
        captured = {}

        def fake(system_prompt, user_content, audit_type, temperature=0.1):
            captured["content"] = user_content
            return (
                '{"conflicts":[{"fact":"左手是否持口镜","numbers":["1","(2)"]},'
                '{"fact":"第二组","numbers":["1","2"]},'
                '{"fact":"第三组","numbers":["1","2"]}]}'
            )

        q._call_model = fake
        result = self._result()
        result["items"].append(
            {"number": "③", "point": "兜底项", "score": 0.2, "full_score": 0.2,
             "reason": "未返回", "fallback": True, "time": "无"}
        )
        groups = q._detect_conflicts(result, "测试项目")
        self.assertEqual(len(groups), q._CONFLICT_MAX_GROUPS)
        self.assertEqual(groups[0]["numbers"], {"①", "②"})
        # 检测只喂文本 不重送视频与关键帧 兜底项本来就没有可信判定不参与比对
        content = captured["content"]
        self.assertEqual([part["type"] for part in content], ["text"])
        self.assertNotIn("兜底项", content[0]["text"])


class ReferenceImageTest(unittest.TestCase):
    """规范参考图: 轮次分配 单项与整单双重上限 下载失败跳过"""

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

    def test_all_images_of_item_are_sent(self):
        q._image_data_uri = lambda ref: f"data:uri:{ref}"
        images = q._reference_images([
            self._item("①", "rule-admin/a/1.jpg\nrule-admin/a/2.jpg"),
            self._item("②", "https://cdn.example/2.png"),
            self._item("③"),
        ])
        # 轮次分配: 先每项首图 再向后补第二张 单图项保持简洁标签 无图项跳过
        self.assertEqual(
            [label for label, _ in images], ["参考图 ①-1", "参考图 ②", "参考图 ①-2"]
        )
        self.assertEqual(images[2][1], "data:uri:rule-admin/a/2.jpg")

    def test_per_item_cap(self):
        q._image_data_uri = lambda ref: f"data:uri:{ref}"
        refs = "\n".join(f"k{idx}.jpg" for idx in range(5))
        images = q._reference_images([self._item("①", refs)])
        self.assertEqual(len(images), q._MAX_REFERENCE_IMAGES_PER_ITEM)
        self.assertEqual([label for label, _ in images], ["参考图 ①-1", "参考图 ①-2", "参考图 ①-3"])

    def test_blank_and_duplicate_refs_collapsed(self):
        self.assertEqual(self._item("①", "a.jpg\na.jpg\n\n b.jpg ").images, ["a.jpg", "b.jpg"])

    def test_capped_by_max_images(self):
        q._image_data_uri = lambda ref: f"data:uri:{ref}"
        items = [self._item(str(idx), f"k{idx}.jpg") for idx in range(20)]
        self.assertEqual(len(q._reference_images(items)), q._MAX_REFERENCE_IMAGES)

    def test_late_item_not_starved_when_capped(self):
        # 6 项各 3 张共 18 张 超整单上限: 轮次分配下每项拿到的张数一致
        # 按项顺序先到先得会让前面几项占满名额 后面的项一张参考图都没有
        q._image_data_uri = lambda ref: f"data:uri:{ref}"
        items = [
            self._item(str(n), "\n".join(f"k{n}-{idx}.jpg" for idx in range(3)))
            for n in range(1, 7)
        ]
        images = q._reference_images(items)
        self.assertEqual(len(images), q._MAX_REFERENCE_IMAGES)
        labels = [label for label, _ in images]
        per_item = q._MAX_REFERENCE_IMAGES // 6
        for n in range(1, 7):
            self.assertEqual(
                sum(1 for label in labels if label.startswith(f"参考图 {n}-")), per_item
            )

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


class SupplementFrameTest(unittest.TestCase):
    """二次复核把片段静态帧一并送模型 器械细节以静态帧为准"""

    def setUp(self):
        self._original_call_model = q._call_model

    def tearDown(self):
        q._call_model = self._original_call_model

    def _run(self, clip: dict) -> list[dict]:
        captured = {}

        def fake_call(system_prompt, content, audit_type):
            captured["content"] = content
            return '{"items":[]}'

        q._call_model = fake_call
        q.rescore_with_clips(
            "探诊检查",
            [{"number": "4", "point": "左手持口镜", "full_score": 0.15, "score": 0.0}],
            [clip],
        )
        return captured["content"]

    def test_clip_frames_are_sent_as_images(self):
        content = self._run({
            "label": " 00:17-00:30", "url": "http://example/clip.mp4",
            "transcript": "", "frames": [("00:27", "B64"), ("00:28", "B64B")],
        })
        images = [part for part in content if part["type"] == "image_url"]
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0]["image_url"]["url"], "data:image/jpeg;base64,B64")
        labels = [part.get("text") for part in content if part["type"] == "text"]
        self.assertIn("关键帧 @00:27：", labels)
        self.assertTrue(any("以这些静态帧为准" in (text or "") for text in labels))

    def test_clip_without_frames_keeps_video_only(self):
        content = self._run({
            "label": " 00:17-00:30", "url": "http://example/clip.mp4", "transcript": "",
        })
        kinds = [part["type"] for part in content]
        self.assertIn("video_url", kinds)
        self.assertNotIn("image_url", kinds)

    def test_blind_rescore_hides_previous_score(self):
        captured = {}

        def fake_call(system_prompt, content, audit_type):
            captured["text"] = system_prompt + content[0]["text"]
            return '{"items":[]}'

        q._call_model = fake_call
        q.rescore_with_clips(
            "探诊检查",
            [{"number": "4", "point": "左手持口镜", "full_score": 0.15, "score": 0.0,
              "reason": "全程未见口镜", "weight": "视觉100%"}],
            [{"label": " 00:17-00:30", "url": "http://example/clip.mp4", "transcript": ""}],
            blind=True,
        )
        self.assertNotIn("上一轮0", captured["text"])
        self.assertNotIn("全程未见口镜", captured["text"])
        # 权重决定口述满分档是否成立 复核时必须带上
        self.assertIn("口述视觉权重：视觉100%", captured["text"])


class ShrinkImageTest(unittest.TestCase):
    """参考图压缩: 阈值内原样透传 超长边才缩小 同时修正 EXIF 方向"""

    def _png(self, size, color=(200, 120, 90)) -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", size, color).save(buf, format="PNG")
        return buf.getvalue()

    def test_small_image_passes_through_untouched(self):
        raw = self._png((640, 480))
        data, mime = q._shrink_image(raw, "image/png")
        self.assertIs(data, raw)
        self.assertEqual(mime, "image/png")

    def test_large_image_is_downscaled_to_max_edge(self):
        raw = self._png((2000, 1500))
        data, mime = q._shrink_image(raw, "image/png")
        self.assertEqual(mime, "image/jpeg")
        self.assertLess(len(data), len(raw))
        with Image.open(io.BytesIO(data)) as out:
            self.assertEqual(max(out.size), q._REFERENCE_MAX_EDGE)
            self.assertEqual(out.size, (1280, 960))

    def test_exif_orientation_is_fixed(self):
        img = Image.new("RGB", (1600, 1200), (10, 20, 30))
        exif = img.getexif()
        exif[0x0112] = 6          # 顺时针旋转 90 度
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=exif)
        data, _ = q._shrink_image(buf.getvalue(), "image/jpeg")
        with Image.open(io.BytesIO(data)) as out:
            self.assertGreater(out.size[1], out.size[0])   # 已转正为竖向
            self.assertEqual(max(out.size), q._REFERENCE_MAX_EDGE)

    def test_oversized_bytes_are_recompressed_without_resize(self):
        noise = Image.effect_noise((1000, 1000), 60).convert("RGB")
        buf = io.BytesIO()
        noise.save(buf, format="PNG")
        raw = buf.getvalue()
        self.assertGreater(len(raw), q._REFERENCE_MAX_BYTES)
        data, mime = q._shrink_image(raw, "image/png")
        self.assertEqual(mime, "image/jpeg")
        self.assertLess(len(data), len(raw))
        with Image.open(io.BytesIO(data)) as out:
            self.assertEqual(out.size, (1000, 1000))

    def test_alpha_channel_flattened_to_white(self):
        buf = io.BytesIO()
        Image.new("RGBA", (1600, 1600), (255, 0, 0, 0)).save(buf, format="PNG")
        data, mime = q._shrink_image(buf.getvalue(), "image/png")
        self.assertEqual(mime, "image/jpeg")
        with Image.open(io.BytesIO(data)) as out:
            self.assertEqual(out.mode, "RGB")
            self.assertGreaterEqual(min(out.getpixel((10, 10))), 250)   # 透明区铺白底

    def test_invalid_bytes_fall_back_to_raw(self):
        data, mime = q._shrink_image(b"not-an-image", "image/png")
        self.assertEqual((data, mime), (b"not-an-image", "image/png"))


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

    def test_weight_is_optional_and_blank_means_full_score(self):
        # 权重未填时不得折算份额 符合评分标准就给满分
        self.assertIn("口述视觉权重是选填项", q._SCORING_POLICY)
        self.assertIn("就给该项满分", q._SCORING_POLICY)

    def test_policy_carries_no_default_weight_ratio(self):
        # 提示词里写死比例会被模型当默认值套到未填权重的项上
        for literal in ("视觉占八成", "口述占两成", "口述20", "视觉80"):
            self.assertNotIn(literal, q._SCORING_POLICY)

    def test_filled_weight_overrides_oral_full_mark_tier(self):
        # 「视觉100%」的项不能再被「口述完整即满分」那一档命中 否则两条规则对同一项给出相反结论
        self.assertIn("优先于第 2 条第 1 档", q._SCORING_POLICY)
        self.assertIn("未填口述视觉权重或权重全部归口述", q._SCORING_POLICY)
        self.assertIn("第 1 档的前提成立", q._EVIDENCE_NOTE)

    def test_asr_failure_is_not_silence(self):
        self.assertIn("语音转写失败", q._EVIDENCE_NOTE)
        self.assertIn("语音转写失败", q._SUPPLEMENT_POLICY)

    def test_item_rules_only_injected_when_mentioned(self):
        # 心肺复苏不该读到口镜与牙科镊的区别
        self.assertEqual(q._scoring_policy("① 胸外按压深度 5~6cm"), q._SCORING_POLICY)
        picked = q._scoring_policy("① 无名指支点稳定")
        self.assertIn("支点的认定", picked)
        self.assertNotIn("口镜与牙科镊的区别", picked)

    def test_conflict_policy_flags_same_hand_contradiction(self):
        self.assertIn("左手未持口镜", q._CONFLICT_POLICY)
        self.assertIn("两项评分点不同并不影响", q._CONFLICT_POLICY)


class TrimReasoningQuoteTest(unittest.TestCase):
    def test_quoted_question_is_kept(self):
        # 引号里的问句是考生原话 在那里截断会把后半句的结论一起丢掉
        reason = "口述“有没有不舒服？”，@00:12 探针轻触龈沟，达标"
        self.assertEqual(q._trim_reasoning(reason), reason)

    def test_bare_question_still_cuts(self):
        self.assertEqual(q._trim_reasoning("支点可见，应给满分吗？不确定"), "支点可见")


class ThinkingSwitchTest(unittest.TestCase):
    def setUp(self):
        self._client = q._client

    def tearDown(self):
        q._client = self._client

    def test_thinking_off_sends_no_extra_body(self):
        captured = {}

        class _Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                msg = type("M", (), {"content": "{}"})
                return type("R", (), {"choices": [type("C", (), {"message": msg})]})

        q._client = type("Client", (), {"chat": type("Chat", (), {"completions": _Completions()})})
        original = q.settings.QWEN_ENABLE_THINKING
        q.settings.QWEN_ENABLE_THINKING = False
        try:
            q._call_model("sys", [{"type": "text", "text": "u"}], "测试项目")
        finally:
            q.settings.QWEN_ENABLE_THINKING = original
        self.assertNotIn("extra_body", captured)


if __name__ == "__main__":
    unittest.main()
