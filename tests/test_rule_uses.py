"""T9: a slip from counting each rule's right and wrong uses in one answer (docs/design_decisions.md)."""

import pytest

from matheasy.diagnosis import RuleUse, rule_uses, uses_slip

ADD = "rule:fractions.add_numerators_denominators"
POWERS = "rule:exponents.multiply_exponents_when_multiplying_powers"


@pytest.mark.parametrize("right,wrong,verdict", [
    (0, 0, None), (1, 1, "slip"), (3, 1, "slip"), (0, 1, "misconception"), (0, 2, "misconception"),
    (5, 2, "misconception"),  # repeated: a habit, however often it was also done right
])
def test_the_verdict_table(right, wrong, verdict):
    assert RuleUse(right, wrong).verdict == verdict


def test_across_steps_the_same_rule_done_right_then_wrong_is_a_slip():
    uses = rule_uses(["1/2 + 1/3 + 1/4", "5/6 + 1/4", "6/10"])
    assert uses[ADD] == RuleUse(right=1, wrong=1) and uses[ADD].verdict == "slip"


def test_in_one_line_letter_terms_done_right_and_wrong_is_a_slip():
    uses = rule_uses(["x^2 × x^3 + x^4 × x^2", "x^5 + x^8"])
    assert uses[POWERS] == RuleUse(right=1, wrong=1)


def test_the_same_mistake_twice_is_a_misconception():
    assert rule_uses(["1/2 + 1/3 + 1/4", "2/5 + 1/4", "3/9"])[ADD] == RuleUse(right=0, wrong=2)


def test_a_mistake_with_no_chance_done_right_is_a_misconception():
    assert rule_uses(["1/2 + 1/3", "2/5"])[ADD] == RuleUse(right=0, wrong=1)


def test_a_part_copied_unchanged_is_not_counted_as_done_right():
    # 1/4 + 1/5 is carried along untouched in line 1, so only 1/2 + 1/3 was worked out there
    uses = rule_uses(["1/2 + 1/3 + (1/4 + 1/5)", "5/6 + (1/4 + 1/5)", "5/6 + 2/9"])
    assert uses[ADD] == RuleUse(right=1, wrong=1)


def test_two_separate_parts_done_right_are_two_chances():
    uses = rule_uses(["x^2 × x^3 + x^4 × x^2 + x^3 × x^3", "x^5 + x^6 + x^9"])
    assert uses[POWERS] == RuleUse(right=2, wrong=1)


def test_numbers_that_combine_in_one_line_are_not_split_yet():
    # needs the line-splitting in docs/plan_multiple_misconceptions.md; nothing is claimed meanwhile
    assert not any(u.wrong for u in rule_uses(["1/2 + 1/3 + 1/4 + 1/5", "5/6 + 2/9"]).values())


def test_typed_fractions_are_numbers_not_divisions():
    # 1/2 is a number: a division rule has no chance on it
    assert "rule:multiplication_division.divide_larger_by_smaller_always" not in rule_uses(["1/2 + 1/3", "5/6"])


def test_a_line_that_does_not_parse_ends_the_count():
    assert rule_uses(["1/2 + 1/3", "5/6", "((("]) == rule_uses(["1/2 + 1/3", "5/6"])


def test_uses_slip_says_why_and_only_for_a_rule():
    assert uses_slip(["x^2 × x^3 + x^4 × x^2", "x^5 + x^8"], POWERS) == \
        "did the same kind of step right once elsewhere in this answer, and wrong once"
    assert uses_slip(["1/2 + 1/3", "2/5"], ADD) is None
    assert uses_slip(["1/2 + 1/3", "2/5"], "slip") is None and uses_slip(["1/2 + 1/3", "2/5"], None) is None


def test_the_page_calls_it_a_slip_and_says_why(tmp_path):
    from matheasy.ui.__main__ import Tutor
    from matheasy.ui.db import Database
    db = Database(tmp_path / "t.db")
    db.add_user("anna", "anna123")
    tutor = Tutor(llm=False, eedi=False, db=db)
    slipped = tutor.analyze("1/2 + 1/3 + 1/4", "5/6 + 1/4\n6/10", "anna")
    assert slipped["kind"] == "wrong" and slipped["slip"] is True and slipped["misconception"] is None
    assert "right once elsewhere" in slipped["why_slip"]
    habit = tutor.analyze("1/2 + 1/3", "2/5", "anna")
    assert habit["slip"] is False and habit["misconception"] == ADD
