# @Auther  : luoscorn
# @Time    : 2026/11/4
# @File    : score_grade.py
"""结果等级判定 — 按得分百分率给出 A/B/C，评分数据不完整或证据不足时给 D"""

# A 档要求百分率严格大于 80，B 档严格大于 60，其余为 C
GRADE_A_MIN_RATE = 80.0
GRADE_B_MIN_RATE = 60.0

GRADE_INCOMPLETE = "D"

# 兜底项占应有评分项的比例达到该值时不再评级
# 兜底分来自「模型没有返回该项」，本质是证据缺失，不是学员的表现
# 超过半数还按百分率给档位，等于把「模型没输出」显示成「学员做错了」
GRADE_FALLBACK_MAX_RATIO = 0.5


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _count_fallback(items) -> int:
    """统计 items 中由证据受限下限兜底产生的评分项数"""
    if not isinstance(items, list):
        return 0
    return sum(1 for item in items if isinstance(item, dict) and item.get("fallback"))


def _incomplete(reason: str, rate: float | None = None) -> dict:
    return {"score_rate": rate, "grade": GRADE_INCOMPLETE, "grade_reason": reason}


def evaluate_grade(result: dict, expected_items: int | None = None) -> dict:
    """
    按「得分 / 满分」的百分率判定等级

    A: 百分率 > 80，B: 百分率 > 60，C: 百分率 <= 60
    D: 评分数据不完整（缺满分、缺得分、评分项缺失）或过半评分项是兜底分时不参与评级

    :param result: 打分结果 dict，需含 full_score / total_score，结构化模式含 items
    :param expected_items: 结构化模式下的应有评分项数，文本模式传 None 不校验
    :return: {"score_rate": float|None, "grade": str, "grade_reason": str}
    """
    full_score = _to_float(result.get("full_score"))
    total_score = _to_float(result.get("total_score"))
    if full_score is None or full_score <= 0:
        return _incomplete("缺少满分(full_score)，无法计算百分率")
    if total_score is None:
        return _incomplete("缺少得分(total_score)")

    items = result.get("items")
    got = len(items) if isinstance(items, list) else 0
    fallback_count = _count_fallback(items)

    if expected_items and got != expected_items:
        return _incomplete(f"评分项不完整：应有 {expected_items} 项，实际 {got} 项")

    rate = round(max(0.0, total_score) / full_score * 100, 2)

    # 过半评分项是兜底分时百分率失去解释力 直接判 D 交人工复核
    # 接口层不知道规则项数 传 None 时以结果自带的 items 数为分母 老数据也能算出兜底占比
    fallback_denom = expected_items or got
    if fallback_denom and fallback_count >= fallback_denom * GRADE_FALLBACK_MAX_RATIO:
        return _incomplete(
            f"证据不足：{fallback_denom} 项中 {fallback_count} 项为模型未返回的兜底分，"
            "占比过半，需人工复核",
            rate,
        )

    if rate > GRADE_A_MIN_RATE:
        grade = "A"
    elif rate > GRADE_B_MIN_RATE:
        grade = "B"
    else:
        grade = "C"

    # 少量兜底不影响档位 但必须在理由里说清楚 否则会被当成正常得分
    reason = ""
    if fallback_count:
        reason = f"其中 {fallback_count}/{got} 项为模型未返回的兜底分，需人工复核"
    return {"score_rate": rate, "grade": grade, "grade_reason": reason}


def attach_grade(result: dict, expected_items: int | None = None) -> dict:
    """把等级判定结果就地写入 result（score_rate / grade / grade_reason）并返回"""
    result.update(evaluate_grade(result, expected_items))
    return result
