"""Decimal and percentage rules. Each docstring gives MalruleLib's example.

Comparisons run on a comparison of two numbers (0.45 < 0.8) and return the
comparison the student would write.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Optional

from ..maths import BinOp, Node, Num, Rel
from . import rule
from ._common import decimal_node, evaluate, fraction_node, value_of

_COMPARISONS = ("<", ">", "=", "<=", ">=")


def _decimal(node: Node) -> Optional[str]:
    """The digits of a plain decimal number as written ('0.45'), else None."""
    if isinstance(node, Num) and "%" not in node.text and node.text.replace(".", "", 1).isdigit():
        return node.text
    return None


def _two_decimals(node: Node, ops: str) -> Optional[tuple[str, str, str]]:
    """(op, '0.3', '0.2') if node is two plain numbers joined by one of `ops`, at least one with a point."""
    if not isinstance(node, (BinOp, Rel)) or node.op not in ops:
        return None
    a, b = _decimal(node.left), _decimal(node.right)
    if a is None or b is None or "." not in a + b:
        return None
    return node.op, a, b


def _as_whole(text: str) -> int:
    """0.45 -> 45: the digits with the point dropped."""
    return int(text.replace(".", "") or 0)


def _after_point(text: str) -> int:
    """0.45 -> 45, 0.8 -> 8, 3 -> 0: the digits after the point, read as a whole number."""
    return int(text.split(".")[1] or 0) if "." in text else 0


def _whole_arithmetic(op: str, a: str, b: str) -> Optional[Node]:
    x, y = _as_whole(a), _as_whole(b)
    if op == "/":
        return None if y == 0 else fraction_node(Fraction(x // y))
    return fraction_node(Fraction({"+": x + y, "-": x - y, "*": x * y}[op]))


def _says_larger(node: Rel, left_is_larger: Optional[bool]) -> Rel:
    """The comparison of node's two sides that says the left one is larger (True), smaller, or equal (None)."""
    return Rel("=" if left_is_larger is None else ">" if left_is_larger else "<", node.left, node.right)


@rule("decimals.ignore_decimal_point")
def ignore_decimal_point(node: Node) -> Optional[Node]:
    """Drops the decimal points and works with whole numbers: 0.3 × 0.2 = 6, 1.5 + 0.25 = 40."""
    found = _two_decimals(node, "+-*/")
    return None if found is None else _whole_arithmetic(*found)


@rule("decimals.whole_number_thinking")
def whole_number_thinking(node: Node) -> Optional[Node]:
    """Treats decimals as whole numbers: 0.5 + 0.25 = 30, and 0.45 > 0.8 because 45 > 8."""
    found = _two_decimals(node, "+-*" + "".join(_COMPARISONS))
    if found is None:
        return None
    op, a, b = found
    if op in _COMPARISONS:
        return _says_larger(node, _after_point(a) > _after_point(b))  # a tie goes to the second, as MalruleLib
    return _whole_arithmetic(op, a, b)


@rule("decimals.longer_is_larger")
def longer_is_larger(node: Node) -> Optional[Node]:
    """Thinks more decimal digits means larger: 0.45 > 0.8."""
    found = _two_decimals(node, "".join(_COMPARISONS))
    if found is None:
        return None
    _, a, b = found
    la, lb = len(a.partition(".")[2]), len(b.partition(".")[2])
    if la == lb:
        x, y = Fraction(a), Fraction(b)
        return _says_larger(node, None if x == y else x > y)  # same length: compares correctly
    return _says_larger(node, la > lb)


def _percent_as_number(node: Node) -> Optional[Fraction]:
    """25% read as 25."""
    if isinstance(node, Num) and node.text.endswith("%"):
        return Fraction(node.text[:-1])
    return value_of(node)


def _has_percent(node: Node) -> bool:
    if isinstance(node, Num):
        return node.text.endswith("%")
    return any(_has_percent(c) for c in (getattr(node, k, None) for k in ("operand", "left", "right")) if c)


@rule("percentages.percent_equals_decimal")
def percent_equals_decimal(node: Node) -> Optional[Node]:
    """Treats a percentage as the plain number: 25% = 25, and 25% of 80 = 25 × 80 = 2000."""
    if not _has_percent(node):
        return None
    value = evaluate(node, _percent_as_number)
    return None if value is None else decimal_node(value)
