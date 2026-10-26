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
from ..diagnosis.maths import BinOp, Neg, Num, Rel, Sym


def item_key(item: str | Node) -> str:
    """A string that is equal for two items exactly when they match.

    Numbers are compared as typed fractions (2/4 is not 1/2: the student sees
    different numbers) and otherwise by value (0.30 is 0.3).
    """
    node = parse(item)
    fraction = as_fraction(node)
    if fraction is not None:
        return f"{fraction[0]}/{fraction[1]}"
    if isinstance(node, Num):
        return str(node.value)
    if isinstance(node, Sym):
        return node.name
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
