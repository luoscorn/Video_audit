# @Auther  : luoscorn
# @Time    : 2026/11/4
# @File    : score_grade.py
"""结果等级判定 — 按得分百分率给出 A/B/C，评分数据不完整时给 D"""

# A 档要求百分率严格大于 80，B 档严格大于 60，其余为 C
GRADE_A_MIN_RATE = 80.0
GRADE_B_MIN_RATE = 60.0

GRADE_INCOMPLETE = "D"


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _incomplete(reason: str) -> dict:
    return {"score_rate": None, "grade": GRADE_INCOMPLETE, "grade_reason": reason}


def evaluate_grade(result: dict, expected_items: int | None = None) -> dict:
    """
    按「得分 / 满分」的百分率判定等级

    A: 百分率 > 80，B: 百分率 > 60，C: 百分率 <= 60
    D: 评分数据不完整（缺满分、缺得分、评分项缺失）时不参与评级

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

    if expected_items:
        items = result.get("items")
        got = len(items) if isinstance(items, list) else 0
        if got != expected_items:
            return _incomplete(f"评分项不完整：应有 {expected_items} 项，实际 {got} 项")

    rate = round(max(0.0, total_score) / full_score * 100, 2)
    if rate > GRADE_A_MIN_RATE:
        grade = "A"
    elif rate > GRADE_B_MIN_RATE:
        grade = "B"
    else:
        grade = "C"
    return {"score_rate": rate, "grade": grade, "grade_reason": ""}


def attach_grade(result: dict, expected_items: int | None = None) -> dict:
    """把等级判定结果就地写入 result（score_rate / grade / grade_reason）并返回"""
    result.update(evaluate_grade(result, expected_items))
    return result
