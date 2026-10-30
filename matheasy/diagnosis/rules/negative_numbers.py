"""Negative number rules. Each docstring gives MalruleLib's example."""

from __future__ import annotations

from fractions import Fraction
from typing import Optional

from ..maths import BinOp, Node, Rel
from . import rule
from ._common import combine, decimal_node, evaluate, value_of

_COMPARISONS = ("<", ">", "=", "<=", ">=")


def _has_negative(node: Node) -> bool:
    value = value_of(node)
    if value is not None:
        return value < 0
    return any(_has_negative(c) for c in (getattr(node, k, None) for k in ("operand", "left", "right")) if c)


@rule("negative_numbers.larger_absolute_value_always_wins")
def larger_absolute_value_always_wins(node: Node) -> Optional[Node]:
    """Gives every answer the sign of the number with the larger size: -7 × 2 = -14, but 3 × -2 = 6,
    and 5 - 8 = 3 (8 is larger, yet subtracted, so the sign is lost). In a comparison, the
    larger size is the larger number: -9 > 4.
    """
    if not isinstance(node, (BinOp, Rel)) or node.op not in "+-*/" + "".join(_COMPARISONS):
        return None
    a, b = value_of(node.left), value_of(node.right)
    if a is None or b is None:
        return None
    if isinstance(node, Rel):
        if not (a < 0 or b < 0):
            return None
        return Rel(">" if abs(a) > abs(b) else "<", node.left, node.right)  # a tie goes to the second
    right = evaluate(node)
    if right is None or not (a < 0 or b < 0 or right < 0):
        return None
    sign = 1 if (a if abs(a) >= abs(b) else b) > 0 else -1  # a tie goes to the first number
    if node.op == "+" and (a > 0) == (b > 0):
        size = abs(a) + abs(b)
    elif node.op in "+-":
        size = abs(abs(a) - abs(b))
    elif node.op == "*":
        size = abs(a) * abs(b)
    else:
        if b == 0:
            return None
        size = abs(a) / abs(b)
    wrong = sign * size
    return None if wrong == right else decimal_node(wrong)


def _negative_product(op: str, a: Fraction, b: Fraction) -> Optional[Fraction]:
    """× and ÷ give a negative answer whenever a negative number is involved."""
    if op in "*/" and (a < 0 or b < 0):
        size = combine(op, abs(a), abs(b))
        return None if size is None else -size
    return combine(op, a, b)


@rule("negative_numbers.negative_times_negative_negative")
def negative_times_negative_negative(node: Node) -> Optional[Node]:
    """Thinks a negative times a negative is negative: (-2) × (-3) = -6, and (-2)(-3) + 4 = -2."""
    if not _has_negative(node):
        return None
    wrong = evaluate(node, apply=_negative_product)
    return None if wrong is None or wrong == evaluate(node) else decimal_node(wrong)
