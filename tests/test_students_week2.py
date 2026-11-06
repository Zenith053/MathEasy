"""D's week-2 students: a student for every rule, held-back rules, slips, several sessions."""

import pytest

from matheasy.diagnosis import HeldBackRuleError, equivalent, held_back_rules, rule_names, run_rule
from matheasy.evaluation import SLIP_ONLY_RATE, SimStudent, correct_answer, roster, rule_answers, slips
from matheasy.evaluation.held_back import held_back_names, held_back_rule

ADDS_ACROSS = "fractions.add_numerators_denominators"


def never(message, misconception):
    return False


def always(message, misconception):
    return True


def mentions_denominators(message, misconception):
    return "denominator" in message


# ---------------------------------------------------------------- held-back rules

# MalruleLib's own examples, from each rule's docstring or calculate_expected_answer().
MALRULELIB_EXAMPLES = [
    ("fractions.denominator_comparison_error", "1/3 > 1/4", "1/3 < 1/4"),
    ("fractions.denominator_comparison_error", "3/5 < 2/5", "3/5 > 2/5"),  # same denominator: correct
    ("fractions.subtract_across", "3/4 - 1/2", "1"),
    ("ratios_proportions.each_fraction_digit_is_ratio", "12/34", "11/32"),
    ("ratios_proportions.each_fraction_digit_is_ratio", "24/36", "22/33"),  # 2:3 and 4:6, right by luck
    ("decimals.right_align_decimals", "3.5 + 2.75", "31"),
    ("decimals.right_align_decimals", "0.5 + 0.25", "3"),
    ("decimals.shorter_is_larger", "0.4 < 0.45", "0.4 > 0.45"),
    ("percentages.percentage_as_index", "30% of 50", "30"),
    ("algebra.change_side_change_sign", "x + 3 = 7", "x = 10"),
    ("algebra.change_side_change_sign", "x - 3 = 7", "x = 4"),
    ("algebra.change_side_change_sign", "10 - x = 4", "x = -6"),
    ("algebra.change_side_change_sign", "2x + 3 = 11", "x = 7"),
    ("negative_numbers.multiplication_rule_for_addition", "-3 + (-5)", "8"),
    ("negative_numbers.multiplication_rule_for_addition", "-3 - (-5)", "-8"),
    ("negative_numbers.negative_swaps_operation", "5 + (-3)", "8"),
    ("negative_numbers.negative_swaps_operation", "5 - (-3)", "2"),
    ("negative_numbers.two_negatives_always_positive", "-5 - 6", "11"),
    ("negative_numbers.two_negatives_always_positive", "-5 - (-6)", "11"),
    ("negative_numbers.two_negatives_always_positive", "5 - (-6)", "1"),
]


@pytest.mark.parametrize("rule,item,expected", MALRULELIB_EXAMPLES)
def test_held_back_rule_gives_malrulelib_answer(rule, item, expected):
    assert equivalent(held_back_rule(rule).predict(item), expected)


@pytest.mark.parametrize("rule", sorted(held_back_rules()))
def test_held_back_rule_is_silent_on_other_topics(rule):
    for item in ("2x + 3", "7 × 8"):
        assert held_back_rule(rule).predict(item) is None


def test_every_held_back_rule_has_code_and_b_still_refuses_them():
    assert set(held_back_names()) == held_back_rules()
    with pytest.raises((KeyError, HeldBackRuleError)):
        run_rule("fractions.subtract_across", "3/4 - 1/2")


@pytest.mark.parametrize("rule,item,expected", [e for e in MALRULELIB_EXAMPLES if e[0] != "fractions.denominator_comparison_error"
                                                 or e[1] == "1/3 > 1/4"])
def test_held_back_student_answers_by_d_code(rule, item, expected):
    s = SimStudent("s1", rule, judge=never)
    answer, followed = s.answer(item)
    if equivalent(expected, correct_answer(item)):
        assert not followed  # the rule happens to be right here
    else:
        assert followed and equivalent(answer, expected)
    assert s.group == "unchecked"  # until D fills eedi_status
    assert s.description == held_back_rule(rule).description


def test_correct_answer_for_equations_and_comparisons():
    assert correct_answer("x + 3 = 7") == "x = 4"
    assert correct_answer("2x + 3 = 10") == "x = 7/2"
    assert correct_answer("1/3 < 1/4") == "1/3 > 1/4"  # the item's sign is ignored
    assert correct_answer("0.5 > 1/2") == "0.5 = 1/2"


# ---------------------------------------------------------------- a student for every rule B releases

SAMPLE_ITEMS = [
    "1/2 + 1/3", "2/5 × 3/5", "2/3 ÷ 4/5", "1/4 + 1/6", "3/4 - 1/2", "12/34", "3/4 > 5/8",
    "0.25 + 0.3", "0.3 × 0.2", "0.45 > 0.8", "25% of 80", "3.5 + 2.75",
    "2(x + 3)", "(a + b)^2", "2x + 4 = 10", "x + 3 = 7", "10 - 2 + 3", "12 ÷ 2 × 3", "2 + 3 × 4",
    "-3 + (-5)", "(-2)(-3)", "-5 - 6",
    # B's week-2 release: each rule's own docstring example
    "5x = 2x + 12", "(x + 3)^2", "2 × (3 + 4)", "2/3 = 6/x",
    "23 + 45", "37 + 25", "328 + 917", "8 + 12", "87 + 93", "47 + 38",
    "321 - 89", "403 - 168", "140 - 21", "403 - 208", "85 - 127", "302 - 7", "304 - 75",
    "85 - 23", "52 - 17", "62 - 37", "402 - 6",
    "68 × 46", "23 × 14", "23 × 4",
    "(x^2)^3", "(2x^3)^2", "2^3", "x^3 × x^5", "2^-2", "5^0",
    "(3 × 10^4) × (2 × 10^5)", "45000", "0.0032",
]


@pytest.mark.parametrize("rule", rule_names())
def test_student_for_every_registered_rule(rule):
    triggering = [i for i in SAMPLE_ITEMS
                  if (p := run_rule(rule, i)) is not None and not equivalent(p, correct_answer(i))]
    if not triggering:
        pytest.skip(f"D: no sample item shows {rule}; add one to SAMPLE_ITEMS")
    s = SimStudent("s1", rule, judge=never)
    for item in triggering:
        r = s.respond("Work this out.", item=item)
        assert (r.answer, r.followed_rule, r.correct, r.slipped) == (run_rule(rule, item), True, False, False)


# ---------------------------------------------------------------- slips

@pytest.mark.parametrize("item", ["1/2 + 1/3", "0.25 + 0.3", "x + 3 = 7", "2(x + 3)", "3/4 - 1/2"])
def test_slips_are_wrong_and_never_a_rule_answer(item):
    found = slips(item)
    assert found
    explained = rule_answers(item)
    for s in found:
        assert not equivalent(s, correct_answer(item))
        assert not any(equivalent(s, e) for e in explained)


def test_a_slip_that_a_rule_would_give_is_left_out():
    # Flipping 1/3 > 1/4 is exactly what denominator_comparison_error predicts.
    assert "1/3 < 1/4" in rule_answers("1/3 > 1/4")
    assert slips("1/3 > 1/4") == []


def test_slip_only_student():
    s = SimStudent("slip_only#0", None, slip_rate=1.0)
    r = s.respond("Work out 1/2 + 1/3", item="1/2 + 1/3")
    assert (r.correct, r.followed_rule, r.slipped, r.holds_misconception) == (False, False, True, False)
    assert r.answer in slips("1/2 + 1/3")
    assert s.group == "slip_only" and s.misconception_id is None


def test_slip_rate_and_seed():
    def answers(seed):
        s = SimStudent("slip_only#0", None, slip_rate=SLIP_ONLY_RATE, seed=seed)
        return [s.respond("q", item="1/2 + 1/3").slipped for _ in range(400)]

    first = answers(0)
    assert first == answers(0)  # same seed, same slips
    assert first != answers(1)
    assert 0.2 < sum(first) / len(first) < 0.4


def test_misconception_students_do_not_slip_by_default():
    s = SimStudent("s1", ADDS_ACROSS, judge=always)
    s.respond("Make the denominators the same first.")
    assert all(s.respond("q", item="1/2 + 1/3").correct for _ in range(50))


# ---------------------------------------------------------------- several sessions

def test_relapse_between_sessions():
    s = SimStudent("s1", ADDS_ACROSS, judge=mentions_denominators, relapse_rate=1.0)
    assert s.begin_session(0) is False
    s.respond("Make the denominators the same first.")
    assert not s.holds_misconception
    assert s.begin_session(1) is True and s.holds_misconception
    assert s.respond("q", item="1/2 + 1/3").followed_rule


def test_begin_session_twice_gives_the_same_answer():
    s = SimStudent("s1", ADDS_ACROSS, judge=mentions_denominators, relapse_rate=1.0)
    s.begin_session(0)
    s.respond("Make the denominators the same first.")
    assert s.begin_session(1) is True
    assert s.begin_session(1) is True and s.holds_misconception


def test_no_relapse_by_default():
    s = SimStudent("s1", ADDS_ACROSS, judge=always)
    s.respond("Make the denominators the same first.")
    assert [s.begin_session(i) for i in range(5)] == [False] * 5 and not s.holds_misconception


def test_saved_student_carries_on_exactly(tmp_path):
    def play(student, n):
        return [student.respond("q", item=f"1/{k + 2} + 1/{k + 3}").answer for k in range(n)]

    kept = SimStudent("s1", ADDS_ACROSS, judge=never, slip_rate=0.5, relapse_rate=0.5)
    saved = SimStudent("s1", ADDS_ACROSS, judge=never, slip_rate=0.5, relapse_rate=0.5)
    assert play(kept, 5) == play(saved, 5)
    kept.begin_session(1)
    saved.begin_session(1)
    path = saved.save(tmp_path)
    assert path.name == "s1.json"
    loaded = SimStudent.load(tmp_path, "s1", judge=never)
    assert loaded.to_state() == kept.to_state()
    assert play(loaded, 5) == play(kept, 5)


def test_what_cannot_be_saved(tmp_path):
    with pytest.raises(ValueError):
        SimStudent("s1", ADDS_ACROSS, judge=never, predict=lambda item: None).save(tmp_path)
    with pytest.raises(ValueError):
        SimStudent("../s1", None).save(tmp_path)


# ---------------------------------------------------------------- roster

def test_roster():
    students = roster(never, copies=2, slip_only=3)
    names = sorted(set(rule_names()) | held_back_rules())
    assert [s.student_id for s in students[:2]] == [f"{names[0]}#0", f"{names[0]}#1"]
    assert len(students) == 2 * len(names) + 3
    assert {s.misconception for s in students} == set(names) | {None}
    assert [s.slip_rate for s in students if s.misconception is None] == [SLIP_ONLY_RATE] * 3
    assert len({s.student_id for s in students}) == len(students)
