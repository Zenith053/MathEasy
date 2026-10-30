"""The rule library, module by module: MalruleLib's own examples first, then the student's own expressions.

Each rule must give MalruleLib's wrong answer, say nothing (None) where its
misconception does not apply, and be found by diagnose() from a real step.
"""

import pytest

from matheasy.diagnosis import diagnose, equivalent, load_bank, rule_id, rule_names, run_rule
from matheasy.diagnosis.rules import twin_names

# (rule, problem, the wrong answer it predicts)
PREDICTS = [
    # ---- fractions
    ("fractions.common_denominator_numerator", "1/4 + 1/6", "1/5"),
    ("fractions.common_denominator_numerator", "1/2 + 1/3 + 1/4", "1/3"),
    ("fractions.ignore_denominators", "2/3 + 1/5", "3"),
    ("fractions.ignore_denominators", "5/8 - 2/3", "3"),
    ("fractions.ignore_denominators", "2/3 × 4/5", "8"),
    ("fractions.ignore_denominators", "3/4 ÷ 2/5", "1"),          # whole-number division, as MalruleLib
    ("fractions.natural_number_bias_numerator_only", "3/4 > 5/8", "3/4 < 5/8"),
    ("fractions.natural_number_bias_numerator_only", "2/3 > 2/5", "2/3 = 2/5"),
    # ---- ratios and proportions
    ("ratios_proportions.additive_instead_of_multiplicative", "2/3 = 6/x", "x = 7"),
    ("ratios_proportions.additive_instead_of_multiplicative", "2/3 = x/9", "x = 8"),
    ("ratios_proportions.additive_instead_of_multiplicative", "x/3 = 6/9", "x = 0"),
    ("ratios_proportions.swap_ratios_or_units", "2/3 = 6/x", "x = 4"),
    ("ratios_proportions.swap_ratios_or_units", "2/3 = x/9", "x = 27/2"),
    ("ratios_proportions.ratio_as_division_only", "2/3 = 6/x", "x = 4"),
    # ---- decimals and percentages
    ("decimals.ignore_decimal_point", "0.3 × 0.2", "6"),
    ("decimals.ignore_decimal_point", "1.5 + 0.25", "40"),
    ("decimals.ignore_decimal_point", "4.8 ÷ 0.2", "24"),
    ("decimals.whole_number_thinking", "0.5 + 0.25", "30"),
    ("decimals.whole_number_thinking", "1.2 - 0.5", "7"),
    ("decimals.whole_number_thinking", "0.45 < 0.8", "0.45 > 0.8"),
    ("decimals.longer_is_larger", "0.45 < 0.8", "0.45 > 0.8"),
    ("decimals.longer_is_larger", "0.01 < 0.3", "0.01 > 0.3"),    # where whole_number_thinking is right
    ("percentages.percent_equals_decimal", "25%", "25"),
    ("percentages.percent_equals_decimal", "25% of 80", "2000"),
    ("percentages.percent_equals_decimal", "50% + 0.2", "50.2"),
    # ---- what multiplying and dividing do
    ("multiplication_division.division_makes_smaller", "4 ÷ 0.5", "2"),
    ("multiplication_division.division_makes_smaller", "3 ÷ 1/4", "3/4"),
    ("multiplication_division.divide_larger_by_smaller_always", "4 ÷ 6", "1.5"),
    ("multiplication_division.multiplication_makes_bigger", "0.5 × 4", "4.5"),
    ("multiplication_division.multiplication_makes_bigger", "1/2 × 1/3", "5/6"),
]

# (rule, a problem its misconception says nothing about)
SILENT = [
    ("fractions.common_denominator_numerator", "1/2 - 1/3"),
    ("fractions.ignore_denominators", "2/3 + 1"),
    ("fractions.ignore_denominators", "1/2 + 1/3 + 1/4"),               # MalruleLib: two fractions only
    ("fractions.natural_number_bias_numerator_only", "3/4 + 5/8"),     # not a comparison
    ("fractions.natural_number_bias_numerator_only", "0.75 > 0.5"),    # not fractions
    ("ratios_proportions.additive_instead_of_multiplicative", "2/3 = 6/9"),  # no unknown
    ("ratios_proportions.additive_instead_of_multiplicative", "x/3 = y/9"),  # two unknowns
    ("decimals.ignore_decimal_point", "3 + 4"),                       # no decimal point
    ("decimals.ignore_decimal_point", "0.5 < 0.8"),                   # a comparison
    ("decimals.longer_is_larger", "0.5 + 0.25"),                      # not a comparison
    ("percentages.percent_equals_decimal", "0.5 + 0.2"),              # no percentage
    ("multiplication_division.division_makes_smaller", "8 ÷ 2"),      # division does make it smaller
    ("multiplication_division.divide_larger_by_smaller_always", "6 ÷ 4"),
    ("multiplication_division.multiplication_makes_bigger", "3 × 4"),
]

# (step before, the student's step, a rule that must be among the suspects)
DIAGNOSED = [
    ("2/3 + 1/5", "3", "fractions.ignore_denominators"),
    ("3/4 > 5/8", "3/4 < 5/8", "fractions.natural_number_bias_numerator_only"),
    ("2/3 = 6/x", "x = 7", "ratios_proportions.additive_instead_of_multiplicative"),
    ("2/3 = 6/x", "x = 4", "ratios_proportions.swap_ratios_or_units"),
    ("0.3 × 0.2", "6", "decimals.ignore_decimal_point"),
    ("0.45 < 0.8", "0.45 > 0.8", "decimals.longer_is_larger"),
    ("25% of 80", "2000", "percentages.percent_equals_decimal"),
    ("4 ÷ 0.5", "2", "multiplication_division.division_makes_smaller"),
    ("x + 4 ÷ 6", "x + 1.5", "multiplication_division.divide_larger_by_smaller_always"),  # inside a line
]


@pytest.mark.parametrize("name, problem, answer", PREDICTS)
def test_rule_predicts_the_wrong_answer(name, problem, answer):
    predicted = run_rule(name, problem)
    assert predicted is not None and equivalent(predicted, answer), predicted


@pytest.mark.parametrize("name, problem", SILENT)
def test_rule_is_silent_where_it_does_not_apply(name, problem):
    assert run_rule(name, problem) is None


@pytest.mark.parametrize("before, after, name", DIAGNOSED)
def test_diagnose_finds_the_rule_from_a_step(before, after, name):
    assert rule_id(name) in diagnose(before, after).beliefs


def test_twins_are_one_suspect():
    # Same wrong answer on every problem: one rule stands for both, so the diagnosis stays clear.
    assert twin_names()["fractions.common_denominator_numerator"] == "fractions.add_numerators_denominators"
    assert run_rule("fractions.common_denominator_numerator", "1/2 + 1/3") == "2/5"
    d = diagnose("1/2 + 1/3", "2/5")
    assert d.status == "clear" and d.misconception == "rule:fractions.add_numerators_denominators"
    assert not set(twin_names()) & set(rule_names())


def test_every_rule_is_tested_and_in_the_bank():
    tested = {name for name, *_ in PREDICTS} | {"fractions.add_numerators_denominators",
                                                 "fractions.keep_common_denominator_for_multiplication",
                                                 "fractions.multiply_across_for_division"}  # in test_diagnosis_rules
    assert set(rule_names()) <= tested, set(rule_names()) - tested
    assert sorted(load_bank().rules()) == sorted(rule_id(n) for n in rule_names())
