"""
Select the held-back MalruleLib rules (Role D, week 1, "do first").

These rules never enter B's misconception bank. D's simulated students use them
to test the tutor on misconceptions it has no executable rule for:
  - "known only as text": the rule matches an Eedi misconception
  - "never seen":         no Eedi misconception matches

The draw is stratified by topic and seeded, so anyone can re-run it and get the
same list. Eligibility is fixed before the draw (see EXCLUDED below).

Usage: python eval/select_held_back.py   -> writes eval/held_back_rules.json
"""

import json
import random
from pathlib import Path

SEED = 20261016  # date of the team plan, fixed before the draw
DRAW = {"fractions": 3, "decimals": 3, "expressions": 4}

# In-scope pool, by MalruleLib folder. The paper's 13 / 16 / 21 category counts
# are not published per rule, so the pool is defined by folder (49 rules).
POOL = {
    "fractions": [
        "fractions.add_numerators_denominators",
        "fractions.common_denominator_numerator",
        "fractions.denominator_comparison_error",
        "fractions.ignore_denominators",
        "fractions.keep_common_denominator_for_multiplication",
        "fractions.multiply_across_for_division",
        "fractions.natural_number_bias_numerator_only",
        "fractions.subtract_across",
        "ratios_proportions.additive_instead_of_multiplicative",
        "ratios_proportions.each_fraction_digit_is_ratio",
        "ratios_proportions.ratio_as_division_only",
        "ratios_proportions.swap_ratios_or_units",
    ],
    "decimals": [
        "decimals.ignore_decimal_point",
        "decimals.longer_is_larger",
        "decimals.right_align_decimals",
        "decimals.shorter_is_larger",
        "decimals.whole_number_thinking",
        "percentages.add_percentages_directly",
        "percentages.percent_equals_decimal",
        "percentages.percentage_as_index",
        "percentages.reverse_percentage_error",
        "rounding.decimal_places_same_as_sig_figs",
        "scientific_notation.add_coefficients_when_multiplying",
        "scientific_notation.count_all_zeros_for_exponent",
        "scientific_notation.wrong_exponent_sign",
        "measurement.inverted_conversion_factor",
        "measurement.wrong_conversion_factor",
    ],
    "expressions": [
        "algebra.cancel_across_equals",
        "algebra.change_side_change_sign",
        "algebra.distribute_over_non_distributive",
        "algebra.divide_one_term_only",
        "algebra.forget_negative_division",
        "algebra.variable_letter_has_value",
        "linear_equations.confuse_slope_and_intercept_roles",
        "linear_equations.slope_direction_confusion",
        "linear_equations.slope_is_delta_x_over_delta_y",
        "factoring.incomplete_factoring",
        "factoring.sign_errors_in_factoring",
        "factoring.sum_of_squares_factors",
        "order_of_operations.addition_before_subtraction_always",
        "order_of_operations.ignore_parentheses",
        "order_of_operations.multiplication_before_division_always",
        "order_of_operations.pemdas_strictly_sequential",
        "order_of_operations.strict_left_to_right",
        "negative_numbers.larger_absolute_value_always_wins",
        "negative_numbers.multiplication_rule_for_addition",
        "negative_numbers.negative_swaps_operation",
        "negative_numbers.negative_times_negative_negative",
        "negative_numbers.two_negatives_always_positive",
    ],
}

# Rules that stay in the bank no matter what, with the reason.
EXCLUDED = {
    # Proposal's worked examples and B's week-1 placeholder need these.
    "fractions.add_numerators_denominators": "flagship example (1/2 + 1/3 = 2/5)",
    "fractions.common_denominator_numerator": "twin of add_numerators_denominators (same answer on every addition)",
    "decimals.longer_is_larger": "flagship example (0.25 > 0.3)",
    "decimals.whole_number_thinking": "overlaps longer_is_larger on comparison items",
    "decimals.ignore_decimal_point": "overlaps whole_number_thinking on arithmetic items",
    # The tutor's items never elicit these, so a student holding one would never show it.
    "percentages.add_percentages_directly": "word problems, out of tutor scope",
    "percentages.reverse_percentage_error": "word problems, out of tutor scope",
    "rounding.decimal_places_same_as_sig_figs": "not a tutor skill",
    "scientific_notation.add_coefficients_when_multiplying": "not a tutor skill",
    "scientific_notation.count_all_zeros_for_exponent": "not a tutor skill",
    "scientific_notation.wrong_exponent_sign": "not a tutor skill",
    "measurement.inverted_conversion_factor": "not a tutor skill",
    "measurement.wrong_conversion_factor": "not a tutor skill",
    "algebra.forget_negative_division": "inequalities, out of tutor scope",
    "linear_equations.confuse_slope_and_intercept_roles": "graphs, out of tutor scope",
    "linear_equations.slope_direction_confusion": "graphs, out of tutor scope",
    "linear_equations.slope_is_delta_x_over_delta_y": "graphs, out of tutor scope",
    "factoring.incomplete_factoring": "quadratics, out of tutor scope",
    "factoring.sign_errors_in_factoring": "quadratics, out of tutor scope",
    "factoring.sum_of_squares_factors": "quadratics, out of tutor scope",
    # The order-of-operations rules reproduce each other (pemdas_strictly_sequential
    # is the union of the A-before-S and M-before-D rules), so holding one back
    # would let a sibling in the bank diagnose it.
    "order_of_operations.addition_before_subtraction_always": "overlapping family",
    "order_of_operations.ignore_parentheses": "overlapping family",
    "order_of_operations.multiplication_before_division_always": "overlapping family",
    "order_of_operations.pemdas_strictly_sequential": "overlapping family",
    "order_of_operations.strict_left_to_right": "overlapping family",
}

# Twins that give the same wrong answer: drawn as one unit, held back together.
UNITS = [
    ("negative_numbers.multiplication_rule_for_addition",
     "negative_numbers.two_negatives_always_positive"),
]


def draw():
    rng = random.Random(SEED)
    held = {}
    for topic, rules in POOL.items():
        eligible = [r for r in rules if r not in EXCLUDED]
        units = []
        for r in eligible:
            unit = next((u for u in UNITS if r in u), (r,))
            if unit not in units:
                units.append(unit)
        rng.shuffle(units)
        chosen = []
        for unit in units:
            if len(chosen) + len(unit) <= DRAW[topic]:
                chosen.extend(unit)
            if len(chosen) == DRAW[topic]:
                break
        held[topic] = sorted(chosen)
    return held


if __name__ == "__main__":
    held = draw()
    n_pool = sum(len(v) for v in POOL.values())
    n_held = sum(len(v) for v in held.values())
    out = {
        "seed": SEED,
        "pool_size": n_pool,
        "held_back": held,
        "bank_candidates": sorted(
            r for v in POOL.values() for r in v if r not in sum(held.values(), [])
        ),
        # Filled in by D after checking each rule against Eedi's misconception list:
        # "known_as_text" with the Eedi MisconceptionId, or "never_seen".
        "eedi_status": {r: None for v in held.values() for r in v},
    }
    path = Path(__file__).with_name("held_back_rules.json")
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{n_held} of {n_pool} rules held back -> {path}")
    for topic, rules in held.items():
        print(f"  {topic}: {', '.join(rules)}")
