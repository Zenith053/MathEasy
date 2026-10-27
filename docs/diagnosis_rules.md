# Diagnosis rules: which rules, in what order

Owner: B. Status: end of week 1. Items marked **needs agreement** are proposals until the named role agrees.

Week-1 check: `python -m matheasy.tutor` runs D's student holding `fractions.add_numerators_denominators` through C's orchestrator and B's `diagnose()`; the diagnosis is correct and the log passes D's checker.

Rules re-implement [MalruleLib](https://github.com/sonkar-lab/malrulelib) (Chen, Liu and Sonkar, 2026, [arXiv:2601.03217](https://arxiv.org/abs/2601.03217)), keeping its names and its wrong answers but running on the student's own expression.

## Rule count: 24, not about 40 (needs team agreement)

The team plan asks for about 40 rules. D's draw (`eval/held_back_rules.json`) leaves 39 bank candidates, but 15 of them are outside the tutor's topics: no item the tutor asks could make a student show them. Writing them would add rules that can never fire.

That leaves **24 rules** to implement, listed below. If the team wants more, the options are to widen the topics or to take rules from MalruleLib folders outside D's pool of 49. Either needs D to redo the held-back draw.

## Rules to implement

Released topic by topic in week 2. Each rule is ready once it runs on real student expressions and has tests.

Examples for the week-2 rules illustrate MalruleLib's descriptions. The exact wrong answers will be checked against MalruleLib's code as each rule is written.

### Fractions (9)

| Rule | Example | Week |
|---|---|---|
| `fractions.add_numerators_denominators` | 1/2 + 1/3 = 2/5 | **1 (done)** |
| `fractions.keep_common_denominator_for_multiplication` | 2/5 × 3/5 = 6/5 | **1 (done)** |
| `fractions.multiply_across_for_division` | 2/3 ÷ 4/5 = 8/15 | **1 (done)** |
| `fractions.common_denominator_numerator` | 1/4 + 1/6 = 2/10 = 1/5 | 2 |
| `fractions.ignore_denominators` | 2/3 + 1/5 = 3 | 2 |
| `fractions.natural_number_bias_numerator_only` | 3/4 < 5/8 because 3 < 5 | 2 |
| `ratios_proportions.additive_instead_of_multiplicative` | 2 : 3 = 6 : 7 | 2 |
| `ratios_proportions.ratio_as_division_only` | 3 : 4 read as 0.75 | 2 |
| `ratios_proportions.swap_ratios_or_units` | proportion set up inverted | 2 |

### Decimals (4)

| Rule | Example | Week |
|---|---|---|
| `decimals.longer_is_larger` | 0.45 > 0.8 | 2 |
| `decimals.whole_number_thinking` | decimals treated as whole numbers | 2 |
| `decimals.ignore_decimal_point` | 0.3 × 0.2 = 6 | 2 |
| `percentages.percent_equals_decimal` | 25% = 25 | 2 |

### Expressions (11)

| Rule | Example | Week |
|---|---|---|
| `algebra.cancel_across_equals` | cancels terms across = as if they were factors | 2 |
| `algebra.distribute_over_non_distributive` | (a + b)² = a² + b² | 2 |
| `algebra.divide_one_term_only` | 2x + 4 = 10 → x + 4 = 5 | 2 |
| `algebra.variable_letter_has_value` | letters have fixed values | 2 |
| `negative_numbers.larger_absolute_value_always_wins` | sign taken from the larger number | 2 |
| `negative_numbers.negative_times_negative_negative` | (−2)(−3) = −6 | 2 |
| `order_of_operations.addition_before_subtraction_always` | 10 − 2 + 3 = 5 | 2 |
| `order_of_operations.ignore_parentheses` | brackets ignored | 2 |
| `order_of_operations.multiplication_before_division_always` | 12 ÷ 2 × 3 = 2 | 2 |
| `order_of_operations.pemdas_strictly_sequential` | PEMDAS as six steps | 2 |
| `order_of_operations.strict_left_to_right` | 2 + 3 × 4 = 20 | 2 |

### Known twins

Some pairs give the same wrong answer on every item, so no follow-up question can separate them:
- `fractions.add_numerators_denominators` and `fractions.common_denominator_numerator`, on every addition.
- `order_of_operations.pemdas_strictly_sequential` overlaps with `addition_before_subtraction_always` and `multiplication_before_division_always`.

The hypothesis tester will report these pairs as `ambiguous` and never `clear`. D's metric should count a diagnosis of either twin as correct. **Needs agreement with D.**

## Not implemented: out of the tutor's topics (15)

`percentages.add_percentages_directly`, `percentages.reverse_percentage_error` (word problems); `rounding.decimal_places_same_as_sig_figs`, `scientific_notation.*` (3), `measurement.*` (2) (not tutor skills); `algebra.forget_negative_division` (inequalities); `linear_equations.*` (3) (graphs); `factoring.*` (3) (quadratics).

## Held back

The 10 rules in `eval/held_back_rules.json` are never implemented here. The rule registry and the bank both refuse them, and a test checks this.

## Probe and test items never overlap (agreed: D implemented it in `matheasy/evaluation/items.py`)

1. **Probes:** fractions use denominators 2–12 and numerators below the denominator. Decimals have at most two decimal places. Expression coefficients and constants are −12 to 12.
2. **B publishes every probe item** (`probe_items()` in week 3, also used for A's shortlist cache). The probe bank is finite, so this is a complete list.
3. **D's test generator rejects any item that matches a probe.** Two items match if they have the same operation on the same numbers, in any order. For example, 1/3 + 1/4 matches 1/4 + 1/3. D's checker runs this over the full test set.

This keeps tests and probes apart without forcing the tests onto harder numbers than the probes, which would change what the pre/post tests measure.
