"""Fraction rules. Each docstring gives MalruleLib's example."""

from __future__ import annotations

from fractions import Fraction
from math import prod
from typing import Optional

from ..maths import BinOp, Node, Rel, as_fraction
from . import rule
from ._common import chain, fraction_node

_COMPARISONS = ("<", ">", "=", "<=", ">=")


def _fractions(node: Node, op: str) -> Optional[list[tuple[int, int]]]:
    """[(a, b), (c, d), ...] if node is a/b op c/d op ... with at least two typed fractions."""
    parts = [as_fraction(t) for t in chain(node, op)]
    if len(parts) < 2 or any(p is None or p[1] == 0 for p in parts):
        return None
    return parts


def _two_fractions(node: Node) -> Optional[tuple[str, tuple[int, int], tuple[int, int]]]:
    """(op, (a, b), (c, d)) if node is exactly a/b op c/d."""
    if not isinstance(node, BinOp) or node.op not in "+-*/":
        return None
    left, right = as_fraction(node.left), as_fraction(node.right)
    if left is None or right is None or left[1] == 0 or right[1] == 0:
        return None
    return node.op, left, right


@rule("fractions.add_numerators_denominators", twins=("fractions.common_denominator_numerator",))
def add_numerators_denominators(node: Node) -> Optional[Node]:
    """Adds the numerators and adds the denominators: 1/2 + 1/3 = 2/5.

    Twin: common_denominator_numerator, which adds the denominators to make a
    "common denominator" (1/4 + 1/6 = 2/10 = 1/5): the same answer on every addition.
    """
    parts = _fractions(node, "+")
    if parts is None:
        return None
    return fraction_node(Fraction(sum(n for n, _ in parts), sum(d for _, d in parts)))


@rule("fractions.keep_common_denominator_for_multiplication")
def keep_common_denominator_for_multiplication(node: Node) -> Optional[Node]:
    """Keeps the shared denominator when multiplying: 2/5 × 3/5 = 6/5."""
    parts = _fractions(node, "*")
    if parts is None or len({d for _, d in parts}) != 1:
        return None
    return fraction_node(Fraction(prod(n for n, _ in parts), parts[0][1]))


@rule("fractions.multiply_across_for_division")
def multiply_across_for_division(node: Node) -> Optional[Node]:
    """Multiplies straight across instead of flipping the divisor: 2/3 ÷ 4/5 = 8/15."""
    found = _two_fractions(node)
    if found is None or found[0] != "/":
        return None
    _, (a, b), (c, d) = found
    return fraction_node(Fraction(a * c, b * d))


@rule("fractions.ignore_denominators")
def ignore_denominators(node: Node) -> Optional[Node]:
    """Works on the numerators only: 2/3 + 1/5 = 3, 2/3 × 4/5 = 8, 3/4 ÷ 2/5 = 1 (whole-number division)."""
    found = _two_fractions(node)
    if found is None:
        return None
    op, (a, _), (c, _) = found
    if op == "/" and c == 0:
        return None
    result = {"+": a + c, "-": a - c, "*": a * c, "/": a // c if c else 0}[op]
    return fraction_node(Fraction(result))


@rule("fractions.natural_number_bias_numerator_only")
def natural_number_bias_numerator_only(node: Node) -> Optional[Node]:
    """Compares fractions by their numerators only: 3/4 < 5/8 because 3 < 5.

    Runs on a comparison of two fractions and returns the comparison the student would write.
    """
    if not isinstance(node, Rel) or node.op not in _COMPARISONS:
        return None
    left, right = as_fraction(node.left), as_fraction(node.right)
    if left is None or right is None:
        return None
    a, c = left[0], right[0]
    return Rel("<" if a < c else ">" if a > c else "=", node.left, node.right)

