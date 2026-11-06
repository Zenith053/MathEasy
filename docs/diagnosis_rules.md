# Diagnosis rules: which rules, and how they run

Owner: B. Items marked **needs agreement** are proposals until the named role agrees.

Rules re-implement [MalruleLib](https://github.com/sonkar-lab/malrulelib) (Chen, Liu and Sonkar, 2026, [arXiv:2601.03217](https://arxiv.org/abs/2601.03217)), keeping its names and its wrong answers but running on the student's own expression. MalruleLib's code was read, never copied or run: each rule follows its `calculate_expected_answer()`, and the column-arithmetic rules were checked against the 120 worked cases in MalruleLib's own tests (all agree).

MalruleLib's licence is non-commercial and share-alike. Our local copy lives in `Dataset/` and is git-ignored. **Needs agreement (team):** whether the repo's licence must match.

## Coverage: 60 of MalruleLib's 102 rules

58 runnable rules, plus 2 twins merged into them (see Twins). The other 42 are held back, out of scope or not expressible (see the end).

| Module (`matheasy/diagnosis/rules/`) | Rules |
|---|---|
| `fractions.py` | 4 + 1 twin |
| `ratios.py` | 2 + 1 twin |
| `decimals.py` (with percentages) | 4 |
| `multiplication_division.py` | 3 |
| `negative_numbers.py` | 2 |
| `order_of_operations.py` | 5 |
| `algebra.py` | 4 |
| `exponents.py` | 7 |
| `arithmetic.py` (whole-number column bugs) | 23 |
| `scientific_notation.py` | 3 |

Every rule has MalruleLib's example in its docstring, a bank entry, and tests in `tests/test_diagnosis_rule_library.py`: the wrong answer it predicts, where it stays silent, and a real step that `diagnose()` traces back to it.

## How a rule runs

A rule takes the part of the step the student got wrong (`LocatedError.source_node`) and returns the answer a student holding the misconception would write, or `None` if the misconception says nothing about that expression. A rule also returns `None` when the misconception happens to give the right answer (for example, no carry is needed), since a correct answer is never diagnosed.

MalruleLib poses many problems in words. In our notation they become:

| MalruleLib asks | The rule runs on | It returns |
|---|---|---|
| "Which is larger, 3/4 or 5/8?" | a comparison, as the true statement: `3/4 > 5/8` | the comparison the student would write: `3/4 < 5/8` |
| a proportion word problem | the proportion with one unknown: `2/3 = 6/x` | the student's solution: `x = 7` |
| "Solve for x" | the equation: `2x + 4 = 10` | `x = 1` |
| "Write 45000 in standard form" | the number: `45000` | `4.5 × 10^3` |
| column arithmetic | two whole numbers: `304 - 75` | `139` |

**Needs agreement (C and D):** comparison items must reach `diagnose()` with the true statement as the step before, and the student's comparison as the step after.

The parser now marks a typed fraction (`6/2`) apart from a division (`6 ÷ 2`): `BinOp.typed_fraction`. Both are equal in value and as trees, and only the order-of-operations rules read the flag. A typed fraction is one number, while a `÷` is an operation done in order.

## Twins: merged into one rule

Two names that give the same wrong answer on every problem can never be told apart by a follow-up question, so keeping both would make every such diagnosis `ambiguous`. A twin is registered on its rule with `twins=`: `run_rule()` accepts its name, it has no bank entry of its own, and the diagnoser names the rule.

| Rule | Twin |
|---|---|
| `fractions.add_numerators_denominators` | `fractions.common_denominator_numerator` |
| `ratios_proportions.swap_ratios_or_units` | `ratios_proportions.ratio_as_division_only` (MalruleLib asks it only where they agree) |

**Needs agreement with D:** a simulated student holding a twin is diagnosed as its rule, and should be counted as correct.

## Overlaps: rules that agree on some problems only

These give the same answer on some problems but not others. `diagnose()` reports `ambiguous` there, and a follow-up question can separate them:

- `fractions.multiply_across_for_division` and `multiplication_division.division_makes_smaller`, on a fraction divided by a fraction below 1 (`2/3 ÷ 4/5`); a divisor above 1 separates them.
- `order_of_operations.pemdas_strictly_sequential` with `addition_before_subtraction_always` (on lines of only + and −) and with `multiplication_before_division_always` (on lines of only × and ÷); a line that mixes both separates them.
- `algebra.distribute_over_non_distributive` and `exponents.distribute_exponent_over_addition`, on `(x + c)^2`; `(x − c)^2` and `(2 + 3)^2` separate them.
- `decimals.ignore_decimal_point` and `decimals.whole_number_thinking`, on + − ×; a comparison or a division separates them.

## Not implemented (42)

| Why | Rules |
|---|---|
| Held back by D for evaluation | the 10 in `eval/held_back_rules.json`. The registry and the bank refuse them, and a test checks this |
| Skipped (B's decision) | `rounding.decimal_places_same_as_sig_figs` |
| Word problems only (input is expressions only) | `percentages.add_percentages_directly`, `percentages.reverse_percentage_error` |
| Not an expression | `subtraction.decompose_by_place_value_label` (place-value labels such as "3 thousands") |
| No notation for it | `absolute_value.*` (3), `radicals.*` (4) |
| Outside the tutor's topics | `algebra.forget_negative_division` (inequalities), `functions.*` (4), `geometry.*` (4), `graphing.*` (1), `statistics.*` (3), `linear_equations.*` (3, graphs), `factoring.*` (3, quadratics), `measurement.*` (2) |

**Needs agreement (A and D):** the tutor's topics. A's Eedi subset includes rounding and standard form, while D's held-back draw marks them "not a tutor skill". The column-arithmetic and exponent rules also widen what the bank covers beyond D's pool of 49, which shifts the balance between known and held-back misconceptions in D's results.

## Probe and test items never overlap (agreed: D implemented it in `matheasy/evaluation/items.py`)

1. **Probes:** fractions use denominators 2–12 and numerators below the denominator. Decimals have at most two decimal places. Expression coefficients and constants are −12 to 12.
2. **B publishes every probe item** (`probe_items()`, also used for A's shortlist cache). The probe bank is finite, so this is a complete list.
3. **D's test generator rejects any item that matches a probe.** Two items match if they have the same operation on the same numbers, in any order. For example, 1/3 + 1/4 matches 1/4 + 1/3. D's checker runs this over the full test set.

This keeps tests and probes apart without forcing the tests onto harder numbers than the probes, which would change what the pre/post tests measure.

## The probe bank

`matheasy/diagnosis/probes.yaml` holds 83 questions, each with what every rule predicts on it and its topics. `eval/build_probes.py` builds it from fixed shapes in the ranges above; run it again when the rules change. A test fails if the file no longer matches the rules.

**Two jobs in the LeetCode-style tutor:**

- **Settling a tie.** When rules tie, `diagnose()` returns the question that splits them best and, once the student answers, keeps the rules that predicted that answer. A right answer rules out the rules predicting a wrong one. An answer no rule predicts changes nothing, so no rule wins by elimination.
- **Similar problems.** `similar_items(topics, seen)` gives practice questions on the student's topics, after any diagnosis.

**Coverage:** every rule but the two comparison rules below has at least 3 items, and all 26 pairs of rules that can tie have at least 3 items that tell them apart. In a simulated run, students answering as their rule predicts settled all 20 ties with one probe each. Real students will not follow a rule that faithfully.

**Gaps:**

- The two comparison rules (`decimals.longer_is_larger`, `fractions.natural_number_bias_numerator_only`) have no items until the comparison format is agreed.
- `collecting_like_terms` has no rule, so it has no items, and `similar_items()` returns nothing for it.
- Conversions to standard form use numbers such as 0.0032, beyond the two-decimal range, because the mistake needs them.
- `mode="b2"` never probes.
