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
    # ---- negative numbers
    ("negative_numbers.larger_absolute_value_always_wins", "3 × -2", "6"),
    ("negative_numbers.larger_absolute_value_always_wins", "5 - 8", "3"),
    ("negative_numbers.larger_absolute_value_always_wins", "-5 - 3", "-2"),
    ("negative_numbers.larger_absolute_value_always_wins", "6 ÷ -2", "3"),
    ("negative_numbers.larger_absolute_value_always_wins", "-9 < 4", "-9 > 4"),
    ("negative_numbers.negative_times_negative_negative", "(-2) × (-3)", "-6"),
    ("negative_numbers.negative_times_negative_negative", "-12 ÷ -4", "-3"),
    ("negative_numbers.negative_times_negative_negative", "(-2)(-3) + 4", "-2"),
    # ---- order of operations
    ("order_of_operations.strict_left_to_right", "2 + 3 × 4", "20"),
    ("order_of_operations.strict_left_to_right", "12 - 4 ÷ 2", "4"),
    ("order_of_operations.strict_left_to_right", "1/2 + 1/3 × 6", "5"),       # a typed fraction is one number
    ("order_of_operations.ignore_parentheses", "2 × (3 + 4)", "10"),
    ("order_of_operations.ignore_parentheses", "20 - (3 + 2)", "19"),
    ("order_of_operations.addition_before_subtraction_always", "10 - 3 + 2", "5"),
    ("order_of_operations.multiplication_before_division_always", "6 ÷ 2 × 3", "1"),
    ("order_of_operations.pemdas_strictly_sequential", "12 ÷ 2 × 3", "2"),
    ("order_of_operations.pemdas_strictly_sequential", "8 - 2 × 3 + 1", "1"),
    # ---- algebra
    ("algebra.cancel_across_equals", "5x = 2x + 12", "x = 12/5"),
    ("algebra.cancel_across_equals", "5x + 3 = 2x + 12", "x = 9/5"),
    ("algebra.divide_one_term_only", "2x + 4 = 10", "x = 1"),
    ("algebra.divide_one_term_only", "3x - 6 = 9", "x = 9"),
    ("algebra.divide_one_term_only", "10 = 2x + 4", "x = 1"),
    ("algebra.variable_letter_has_value", "x + 5 = 12", "x = 10"),
    ("algebra.variable_letter_has_value", "2b + 1 = 9", "b = 2"),
    ("algebra.distribute_over_non_distributive", "(x + 3)^2", "x^2 + 9"),
    ("algebra.distribute_over_non_distributive", "(x - 3)^2", "x^2 + 9"),    # MalruleLib drops the sign too
    # ---- exponents
    ("exponents.multiply_base_by_exponent", "2^3", "6"),
    ("exponents.multiply_base_by_exponent", "2^3 + 1", "7"),
    ("exponents.zero_exponent_equals_zero", "5^0", "0"),
    ("exponents.zero_exponent_equals_zero", "5x^0", "0"),
    ("exponents.zero_exponent_equals_zero", "x^3/x^3", "0"),
    ("exponents.negative_exponent_makes_negative", "2^-2", "-4"),
    ("exponents.multiply_exponents_when_multiplying_powers", "x^3 × x^5", "x^15"),
    ("exponents.multiply_exponents_when_multiplying_powers", "2^3 × 2^4", "2^12"),
    ("exponents.add_exponents_for_power_of_power", "(x^2)^3", "x^5"),
    ("exponents.add_exponents_for_power_of_power", "(2x^2)^3", "2x^5"),
    ("exponents.add_exponents_for_power_of_power", "(x^2)^3 × x", "x^6"),
    ("exponents.forget_exponent_on_coefficient", "(2x^3)^2", "2x^6"),
    ("exponents.forget_exponent_on_coefficient", "(3x^2y)^2", "3x^4y^2"),
    ("exponents.distribute_exponent_over_addition", "(2 + 3)^2", "13"),
    ("exponents.distribute_exponent_over_addition", "(x + 1 + 2)^2", "x^2 + 5"),
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
    ("negative_numbers.larger_absolute_value_always_wins", "-7 × 2"),    # the larger size has the right sign
    ("negative_numbers.negative_times_negative_negative", "-2 × 3"),     # no negative × negative
    ("order_of_operations.strict_left_to_right", "2 × 3 + 4"),           # left to right is right here
    ("order_of_operations.strict_left_to_right", "2 × (3 + 4)"),         # brackets: ignore_parentheses' case
    ("order_of_operations.ignore_parentheses", "2 + 3 × 4"),             # no brackets
    ("order_of_operations.addition_before_subtraction_always", "10 + 3 - 2"),
    ("order_of_operations.multiplication_before_division_always", "6/2 × 3"),  # typed fraction: no division
    ("algebra.cancel_across_equals", "5x = 12"),                         # the letter on one side only
    ("algebra.divide_one_term_only", "x + 4 = 10"),                      # nothing to divide by
    ("algebra.variable_letter_has_value", "x + 5 = 15"),                 # 10 happens to be right
    ("algebra.distribute_over_non_distributive", "(2 + 3)^2"),           # no letter
    ("exponents.multiply_base_by_exponent", "x^3"),                      # MalruleLib: number bases only
    ("exponents.multiply_exponents_when_multiplying_powers", "x^2 × y^3"),  # different bases
    ("exponents.forget_exponent_on_coefficient", "(x^3)^2"),             # no number in front
    ("exponents.distribute_exponent_over_addition", "(x - 3)^2"),        # MalruleLib: sums only
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
    ("3 × -2", "6", "negative_numbers.larger_absolute_value_always_wins"),
    ("(-2) × (-3)", "-6", "negative_numbers.negative_times_negative_negative"),
    ("2 + 3 × 4", "20", "order_of_operations.strict_left_to_right"),
    ("2 × (3 + 4)", "10", "order_of_operations.ignore_parentheses"),
    ("6 ÷ 2 × 3", "1", "order_of_operations.multiplication_before_division_always"),
    ("2x + 4 = 10", "x = 1", "algebra.divide_one_term_only"),
    ("2x + 4 = 10", "x + 4 = 5", "algebra.divide_one_term_only"),       # the intermediate step itself
    ("x + 5 = 12", "x = 10", "algebra.variable_letter_has_value"),
    ("(x^2)^3", "x^5", "exponents.add_exponents_for_power_of_power"),
    ("2^-2", "-4", "exponents.negative_exponent_makes_negative"),
    ("(x - 3)^2", "x^2 + 9", "algebra.distribute_over_non_distributive"),
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
