"""Beliefs about what multiplying and dividing do. Each docstring gives MalruleLib's example.

Each rule changes the answer only when the belief is broken by the numbers (a
factor or divisor below 1, a smaller number divided by a larger); elsewhere the
student would be right, so the rule says nothing.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Optional

from ..maths import BinOp, Node
from . import rule
from ._common import decimal_node, value_of


def _two_numbers(node: Node, op: str) -> Optional[tuple[Fraction, Fraction]]:
    """(a, b) if node is a op b with two positive numbers."""
    if not (isinstance(node, BinOp) and node.op == op):
        return None
    a, b = value_of(node.left), value_of(node.right)
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    return a, b


@rule("multiplication_division.division_makes_smaller")
def division_makes_smaller(node: Node) -> Optional[Node]:
    """Expects division to make smaller, so multiplies when it would not: 4 ÷ 0.5 = 2."""
    found = _two_numbers(node, "/")
    if found is None:
        return None
    a, b = found
    return decimal_node(a * b) if a / b > a else None


@rule("multiplication_division.divide_larger_by_smaller_always")
def divide_larger_by_smaller_always(node: Node) -> Optional[Node]:
    """Always divides the larger number by the smaller: 4 ÷ 6 = 6 ÷ 4 = 1.5."""
    found = _two_numbers(node, "/")
    if found is None:
        return None
    a, b = found
    return decimal_node(b / a) if a < b else None


@rule("multiplication_division.multiplication_makes_bigger")
def multiplication_makes_bigger(node: Node) -> Optional[Node]:
    """Expects multiplying to make bigger, so adds when it would not: 0.5 × 4 = 4.5."""
    found = _two_numbers(node, "*")
    if found is None:
        return None
    a, b = found
    return decimal_node(a + b) if a * b < a or a * b < b else None
