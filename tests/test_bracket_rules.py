"""Our own bracket and like-term rules (to-do T2): each rule's examples, where it does not apply,
and the whole step diagnosed with no LLM."""

import pytest

from matheasy.diagnosis import SLIP, diagnose, rule_id, run_rule
from matheasy.diagnosis.explain import rule_topic


@pytest.mark.parametrize("name, problem, answer", [
    ("brackets.multiply_first_term_only", "2(x + 3)", "2x + 3"),
    ("brackets.multiply_first_term_only", "3(a - b + 4)", "3a - b + 4"),
    ("brackets.multiply_first_term_only", "x(x + 3)", "x^2 + 3"),
    ("brackets.minus_first_term_only", "-(x + 1)", "-x + 1"),
    ("brackets.minus_first_term_only", "5 - (x - 2)", "5 - x - 2"),
    ("brackets.negative_multiplier_first_term_only", "-2(x - 3)", "-2x - 6"),
    ("brackets.negative_multiplier_first_term_only", "-3(a + b - 1)", "-3a + 3b - 3"),
    ("like_terms.add_letters_as_powers", "x + x", "x^2"),
    ("like_terms.add_letters_as_powers", "2x + 3x", "5x^2"),
    ("like_terms.combine_unlike_terms", "3x + 2", "5x"),
    ("like_terms.combine_unlike_terms", "4a + 3b", "7ab"),
])
def test_rules_predict_the_wrong_answer(name, problem, answer):
    from matheasy.diagnosis import equivalent
    assert equivalent(run_rule(name, problem), answer)


@pytest.mark.parametrize("name, problem", [
    ("brackets.multiply_first_term_only", "2 × (3 + 4)"),        # numbers only: order of operations, not expanding
    ("brackets.multiply_first_term_only", "2x + 3"),             # no bracket
    ("brackets.minus_first_term_only", "20 - (3 + 4)"),
    ("brackets.minus_first_term_only", "5 - x - 2"),             # no bracket
    ("brackets.negative_multiplier_first_term_only", "2(x - 3)"),  # the multiplier is not negative
    ("like_terms.add_letters_as_powers", "x + y"),               # not like terms
    ("like_terms.add_letters_as_powers", "x^2 + x^2"),           # powers are left alone
    ("like_terms.combine_unlike_terms", "x + x"),                # like terms
    ("like_terms.combine_unlike_terms", "2 + 3"),                # numbers only
    ("like_terms.combine_unlike_terms", "3x - 2"),               # only an addition
])
def test_rules_do_not_apply(name, problem):
    assert run_rule(name, problem) is None


@pytest.mark.parametrize("before, after, rule", [
    ("2(x + 3)", "2x + 3", "brackets.multiply_first_term_only"),
    ("-(x + 1)", "-x + 1", "brackets.minus_first_term_only"),
    ("5 - (x - 2)", "5 - x - 2", "brackets.minus_first_term_only"),
    ("-2(x - 3)", "-2x - 6", "brackets.negative_multiplier_first_term_only"),
    ("x + x", "x^2", "like_terms.add_letters_as_powers"),
    ("3x + 2", "5x", "like_terms.combine_unlike_terms"),
    ("3(x + 4) = 21", "3x + 4 = 21", "brackets.multiply_first_term_only"),  # inside a longer line
])
def test_the_step_is_diagnosed_by_the_rule_with_no_llm(before, after, rule):
    d = diagnose(before, after)
    assert (d.status, d.misconception) == ("clear", rule_id(rule))
    assert d.topics == (rule_topic(rule_id(rule)),)


def test_the_new_groups_have_their_topics():
    assert rule_topic("rule:brackets.multiply_first_term_only") == "expanding_brackets"
    assert rule_topic("rule:like_terms.combine_unlike_terms") == "collecting_like_terms"


def test_a_consistent_mistake_is_a_rule_and_an_inconsistent_one_a_slip():
    rule = diagnose("2(x + y + z)", "2x + y + z")             # no term after the first multiplied
    assert rule.misconception == rule_id("brackets.multiply_first_term_only")
    slip = diagnose("-2(x + y - z - k)", "-2x - 2y + 2z - 2k")  # -z right, -k wrong
    assert slip.misconception == SLIP


def test_two_mistakes_in_one_line_match_no_single_rule():
    assert diagnose("2(x + 3) - (x + 1)", "2x + 3 - x + 1").status == "unknown"
