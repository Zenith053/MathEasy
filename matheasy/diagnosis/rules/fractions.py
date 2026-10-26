"""Fraction rules. Each docstring gives MalruleLib's example."""

from __future__ import annotations

from fractions import Fraction
from math import prod
from typing import Optional

import sympy

from ..maths import BinOp, Neg, Node, Num, as_fraction
from . import rule


def _fraction_node(value: Fraction) -> Node:
    """A simplified fraction as a node: 2/4 -> 1/2, 4/2 -> 2, -1/2 -> -(1/2)."""
    if value < 0:
        return Neg(_fraction_node(-value))
    num = Num(sympy.Integer(value.numerator), str(value.numerator))
    if value.denominator == 1:
        return num
    return BinOp("/", num, Num(sympy.Integer(value.denominator), str(value.denominator)))


def _chain(node: Node, op: str) -> list[Node]:
    """a + b + c -> [a, b, c] for op '+'; a single term is a chain of one."""
    if isinstance(node, BinOp) and node.op == op and not as_fraction(node):
        return _chain(node.left, op) + _chain(node.right, op)
    return [node]


def _fractions(node: Node, op: str) -> Optional[list[tuple[int, int]]]:
    """[(a, b), (c, d), ...] if node is a/b op c/d op ... with at least two typed fractions."""
    parts = [as_fraction(t) for t in _chain(node, op)]
    if len(parts) < 2 or any(p is None or p[1] == 0 for p in parts):
        return None
    return parts


@rule("fractions.add_numerators_denominators", topic="fractions")
def add_numerators_denominators(node: Node) -> Optional[Node]:
    """Adds the numerators and adds the denominators: 1/2 + 1/3 = 2/5."""
    parts = _fractions(node, "+")
    if parts is None:
        return None
    return _fraction_node(Fraction(sum(n for n, _ in parts), sum(d for _, d in parts)))


@rule("fractions.keep_common_denominator_for_multiplication", topic="fractions")
def keep_common_denominator_for_multiplication(node: Node) -> Optional[Node]:
    """Keeps the shared denominator when multiplying: 2/5 × 3/5 = 6/5."""
    parts = _fractions(node, "*")
    if parts is None or len({d for _, d in parts}) != 1:
        return None
    return _fraction_node(Fraction(prod(n for n, _ in parts), parts[0][1]))


@rule("fractions.multiply_across_for_division", topic="fractions")
def multiply_across_for_division(node: Node) -> Optional[Node]:
    """Multiplies straight across instead of flipping the divisor: 2/3 ÷ 4/5 = 8/15."""
    if not (isinstance(node, BinOp) and node.op == "/"):
        return None
    left, right = as_fraction(node.left), as_fraction(node.right)
    if left is None or right is None or left[1] == 0 or right[1] == 0:
        return None
    (a, b), (c, d) = left, right
    return _fraction_node(Fraction(a * c, b * d))
