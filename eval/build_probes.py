"""Build the probe bank, matheasy/diagnosis/probes.yaml. Run again when the rules change.

    PYTHONPATH=. .venv/bin/python eval/build_probes.py

Candidates come from fixed shapes in the agreed number ranges (docs/diagnosis_rules.md).
Every rule runs on every candidate, and items are picked greedily until:

  - every rule has MIN_ITEMS items on which it gives a wrong answer, and
  - every pair of rules that can tie has MIN_ITEMS items that tell them apart.

The bank stays small on purpose: D's tests may not reuse a probe item, so a large bank
would leave them no items. Comparisons are left out until their item format is agreed.
"""

import random
from collections import Counter
from itertools import combinations
from pathlib import Path

import yaml

from matheasy.diagnosis import equivalent, parse, rule_id, rule_names, run_rule
from matheasy.diagnosis.explain import rule_topic

OUT = Path(__file__).resolve().parents[1] / "matheasy" / "diagnosis" / "probes.yaml"
SEED = 20261106
PER_SHAPE = 12    # candidates tried for each shape
MIN_ITEMS = 3     # items per rule, and per pair of rules that can tie


def fraction(rng):
    d = rng.randint(2, 12)
    return f"{rng.randint(1, d - 1)}/{d}"


def decimal(rng):
    return rng.choice([f"0.{rng.randint(1, 9)}", f"0.{rng.randint(11, 99)}", f"{rng.randint(1, 9)}.{rng.randint(1, 9)}"])


def small(rng):
    return rng.randint(2, 12)


# Each shape takes a random generator and returns one item.
SHAPES = [
    # fractions
    *[lambda r, op=op: f"{fraction(r)} {op} {fraction(r)}" for op in "+-×÷"],
    lambda r: f"{fraction(r)} + {fraction(r)} + {fraction(r)}",
    lambda r: (lambda d: f"{r.randint(1, d - 1)}/{d} × {r.randint(1, d - 1)}/{d}")(r.randint(3, 12)),
    # decimals, percentages, what multiplying and dividing do
    *[lambda r, op=op: f"{decimal(r)} {op} {decimal(r)}" for op in "+-×"],
    lambda r: f"{r.choice([5, 10, 20, 25, 40, 50, 75])}%",
    lambda r: f"{r.choice([10, 20, 25, 50])}% of {r.choice([20, 40, 60, 80])}",
    lambda r: f"{r.choice([10, 20, 50])}% + {decimal(r)}",
    lambda r: f"{small(r)} ÷ 0.{r.randint(1, 9)}",
    lambda r: f"{small(r)} ÷ {fraction(r)}",
    lambda r: f"{r.randint(2, 6)} ÷ {r.randint(7, 12)}",
    lambda r: f"0.{r.randint(1, 9)} × {small(r)}",
    lambda r: f"{fraction(r)} × {small(r)}",
    # negative numbers
    lambda r: f"{small(r)} × -{small(r)}",
    lambda r: f"{r.randint(2, 6)} - {r.randint(7, 12)}",
    lambda r: f"-{small(r)} - {small(r)}",
    lambda r: f"{small(r)} ÷ -{small(r)}",
    lambda r: f"(-{small(r)}) × (-{small(r)})",
    lambda r: f"-{small(r)} ÷ -{small(r)}",
    # order of operations
    lambda r: f"{small(r)} + {small(r)} × {small(r)}",
    lambda r: f"{small(r)} - {small(r)} ÷ {small(r)}",
    lambda r: f"{small(r)} × ({small(r)} + {small(r)})",
    lambda r: f"{r.randint(20, 40)} - ({small(r)} + {small(r)})",
    lambda r: f"{r.randint(13, 20)} - {small(r)} + {small(r)}",
    lambda r: f"{r.choice([12, 18, 24])} ÷ {r.choice([2, 3])} × {small(r)}",
    lambda r: f"{r.randint(13, 20)} - {small(r)} × {r.randint(2, 3)} + {small(r)}",
    # algebra
    lambda r: f"{r.randint(5, 12)}x = {r.randint(2, 4)}x + {small(r)}",
    lambda r: f"{r.randint(5, 12)}x + {small(r)} = {r.randint(2, 4)}x + {small(r)}",
    lambda r: f"{r.randint(2, 6)}x + {small(r)} = {r.randint(10, 12)}",
    lambda r: f"{r.randint(2, 6)}x - {small(r)} = {small(r)}",
    lambda r: f"x + {small(r)} = {r.randint(10, 12)}",
    lambda r: f"(x + {small(r)})^2",
    lambda r: f"(x - {small(r)})^2",
    # exponents
    lambda r: f"{r.randint(2, 6)}^{r.randint(2, 4)}",
    lambda r: f"{small(r)}^0",
    lambda r: f"{small(r)}x^0",
    lambda r: f"x^{r.randint(2, 6)}/x^{r.randint(2, 6)}",
    lambda r: f"{r.randint(2, 5)}^-{r.randint(1, 3)}",
    lambda r: f"x^{r.randint(2, 6)} × x^{r.randint(2, 6)}",
    lambda r: f"{r.randint(2, 3)}^{r.randint(2, 4)} × {r.randint(2, 3)}^{r.randint(2, 4)}",
    lambda r: f"(x^{r.randint(2, 5)})^{r.randint(2, 4)}",
    lambda r: f"({r.randint(2, 5)}x^{r.randint(2, 4)})^{r.randint(2, 3)}",
    lambda r: f"({small(r)} + {small(r)})^2",
    # whole-number arithmetic
    lambda r: f"{r.randint(11, 99)} + {r.randint(11, 99)}",
    lambda r: f"{r.randint(101, 999)} + {r.randint(101, 999)}",
    lambda r: f"{r.randint(30, 99)} - {r.randint(11, 29)}",
    lambda r: f"{r.randint(101, 999)} - {r.randint(11, 99)}",
    lambda r: f"{r.randint(101, 999)} - {r.randint(101, 199)}",
    lambda r: f"{r.randint(2, 9)} + {r.randint(11, 99)}",
    lambda r: f"{r.randint(2, 9)}0{r.randint(1, 9)} - {r.randint(3, 9)}",
    lambda r: f"{r.randint(2, 9)}0{r.randint(1, 9)} - {r.randint(11, 99)}",
    lambda r: f"{r.randint(2, 9)}0{r.randint(1, 9)} - {r.randint(101, 199)}",
    lambda r: f"{r.randint(11, 99)} × {r.randint(11, 99)}",
    lambda r: f"{r.randint(11, 99)} × {r.randint(2, 9)}",
    # ratios
    lambda r: f"{r.randint(1, 4)}/{r.randint(5, 8)} = {small(r)}/x",
    lambda r: f"{r.randint(1, 4)}/{r.randint(5, 8)} = x/{small(r)}",
    lambda r: f"x/{r.randint(2, 5)} = {fraction(r)}",
    # standard form
    lambda r: f"({r.randint(2, 4)} × 10^{r.randint(2, 6)}) × ({r.randint(2, 4)} × 10^{r.randint(2, 6)})",
    lambda r: f"({r.choice([4, 6, 8])} × 10^{r.randint(5, 9)}) ÷ (2 × 10^{r.randint(2, 4)})",
    lambda r: f"{r.randint(11, 99)}000",
    lambda r: f"0.00{r.randint(11, 99)}",  # beyond two decimal places: standard form needs it
]


def candidates():
    rng = random.Random(SEED)
    items = {shape(rng) for shape in SHAPES for _ in range(PER_SHAPE)}
    return sorted(items)


def predictions(item, rules):
    """{rule: what it predicts} for the rules that give a wrong answer on this item."""
    found = {}
    for rule in rules:
        predicted = run_rule(rule, item)
        if predicted is not None and not equivalent(predicted, item):
            found[rule] = str(predicted)
    return found


def tied(a, b, predicted):
    return a in predicted and b in predicted and equivalent(predicted[a], predicted[b])


def separated(a, b, predicted):
    """The item tells a and b apart: at least one is wrong here, and their answers differ."""
    if a not in predicted and b not in predicted:
        return False
    return a not in predicted or b not in predicted or not equivalent(predicted[a], predicted[b])


def build():
    rules = [rule_id(name) for name in rule_names()]
    table = {item: predictions(item, rules) for item in candidates()}
    table = {item: p for item, p in table.items() if p}
    pairs = [(a, b) for a, b in combinations(rules, 2) if any(tied(a, b, p) for p in table.values())]
    separable = [pair for pair in pairs if any(separated(*pair, p) for p in table.values())]

    need = Counter({rule: MIN_ITEMS for rule in rules})
    need.update({pair: MIN_ITEMS for pair in separable})

    def gain(item):
        p = table[item]
        return (sum(need[r] > 0 for r in p) + sum(need[pair] > 0 for pair in separable if separated(*pair, p)))

    chosen, left = [], sorted(table)
    while left and any(n > 0 for n in need.values()):
        best = max(left, key=lambda item: (gain(item), item))
        if gain(best) == 0:
            break
        left.remove(best)
        chosen.append(best)
        for rule in table[best]:
            need[rule] -= 1
        for pair in separable:
            if separated(*pair, table[best]):
                need[pair] -= 1

    report(rules, pairs, separable, table, chosen, need)
    bank = [{"question": item, "topics": sorted({rule_topic(r) for r in table[item]}), "predictions": table[item]}
            for item in sorted(chosen)]
    OUT.write_text(yaml.safe_dump(bank, allow_unicode=True, sort_keys=False, width=120))
    print(f"-> {OUT}")


def report(rules, pairs, separable, table, chosen, need):
    print(f"{len(table)} candidates usable; {len(chosen)} items chosen")
    print(f"{len(pairs)} pairs of rules can tie; {len(separable)} of them can be told apart")
    print("never separable on these candidates:", sorted(set(pairs) - set(separable)))
    print("rules short of items:", {r: need[r] for r in rules if need[r] > 0})
    print("pairs short of items:", sum(need[p] > 0 for p in separable))


if __name__ == "__main__":
    parse("1/2")  # fail early if the package is not importable
    build()
