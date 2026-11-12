"""Keeping pre/post-test items apart from B's probes (owner: D).

B's proposal (docs/diagnosis_rules.md, "Probe and test items never overlap"), which
this implements: tests use the same number ranges as probes, and D rejects any test
item that matches a probe. Two items match if they have the same operations on the same numbers, in any
order: 1/3 + 1/4 matches 1/4 + 1/3, and 3/4 - 1/4 matches 1/4 - 3/4.

    from matheasy.evaluation.items import overlapping
    overlapping(test_items, probe_items())   # [] when the sets are kept apart
"""

from __future__ import annotations

from typing import Iterable

from ..diagnosis import Node, as_fraction, parse
from ..diagnosis.maths import BinOp, Mixed, Neg, Num, Rel, Sym, Word

# Items that show each rule: MalruleLib's own examples, one or more per rule. Every
# registered rule should give a wrong answer on at least one of them
# (tests/test_students_week2.py checks this). The pre/post tests (prepost.py) and C's
# runner start from these.
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
    # Our own bracket and like-term rules (T2): their docstring examples, with numbers to vary
    "5 - (x - 2)", "-2(x - 3)", "2x + 3x",
    # Held-back rules: MalruleLib examples the sample above does not already cover
    "1/3 > 1/4", "0.4 < 0.45", "30% of 50", "x - 3 = 7", "5 + (-3)", "5 - (-3)",
]

# Items for students with no misconception: one per topic family.
SLIP_ITEMS = ["1/2 + 1/3", "0.25 + 0.3", "x + 3 = 7", "2(x + 3)", "3/4 - 1/2"]


def item_key(item: str | Node) -> str:
    """A string that is equal for two items exactly when they match.

    Numbers are compared as typed fractions (2/4 is not 1/2: the student sees
    different numbers), mixed numbers as typed (3 1/2 is not 7/2), and otherwise by
    value (0.30 is 0.3). Named quantities match by name in any case (Area, area).
    """
    node = parse(item)
    fraction = as_fraction(node)
    if fraction is not None:
        return f"{fraction[0]}/{fraction[1]}"
    if isinstance(node, Num):
        return str(node.value)
    if isinstance(node, Mixed):
        return f"{node.whole} {node.num}/{node.den}"  # as typed: 3 1/2 is not 7/2
    if isinstance(node, (Sym, Word)):
        return node.name  # a word in lower case, so Area and area match
    if isinstance(node, Neg):
        return f"-({item_key(node.operand)})"
    if isinstance(node, (BinOp, Rel)):
        return f"{node.op}({','.join(sorted([item_key(node.left), item_key(node.right)]))})"
    raise TypeError(f"unexpected node {node!r}")


def overlapping(tests: Iterable[str], probes: Iterable[str]) -> list[tuple[str, str]]:
    """(test, probe) pairs that match. Empty when the sets are kept apart."""
    by_key = {}
    for probe in probes:
        by_key.setdefault(item_key(probe), probe)
    return [(t, by_key[item_key(t)]) for t in tests if item_key(t) in by_key]
