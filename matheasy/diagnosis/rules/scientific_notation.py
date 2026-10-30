"""Standard form rules. Each docstring gives MalruleLib's example.

Converting rules run on the plain number the student was asked to convert
(45000), and return what the student would write (4.5 × 10^3).
"""

from __future__ import annotations

from fractions import Fraction
from typing import Optional

from ..maths import BinOp, Node, Num, parse
from . import rule
from ._common import decimal_text, value_of


def _standard_form(coefficient: Fraction, exponent: int) -> Node:
    sign = "-" if coefficient < 0 else ""
    return parse(f"{sign}{decimal_text(abs(coefficient))} × 10^{exponent}")


def _factors(node: Node, sign: int, out: list[tuple[int, Node]]) -> bool:
    """Every factor of a product or quotient, with -1 for those divided by."""
    if isinstance(node, BinOp) and node.op in "*/" and not node.typed_fraction:
        return _factors(node.left, sign, out) and _factors(node.right, sign if node.op == "*" else -sign, out)
    out.append((sign, node))
    return True


def _power_of_ten(node: Node) -> Optional[int]:
    if isinstance(node, BinOp) and node.op == "^" and value_of(node.left) == 10:
        k = value_of(node.right)
        return int(k) if k is not None and k.denominator == 1 else None
    return None


@rule("scientific_notation.add_coefficients_when_multiplying")
def add_coefficients_when_multiplying(node: Node) -> Optional[Node]:
    """Adds the coefficients instead of multiplying them: (3 × 10^4) × (2 × 10^5) = 5 × 10^9.
    Dividing, the exponents are subtracted but the coefficients still added."""
    factors: list[tuple[int, Node]] = []
    _factors(node, 1, factors)
    exponents = [(s, _power_of_ten(f)) for s, f in factors if _power_of_ten(f) is not None]
    coefficients = [value_of(f) for _, f in factors if _power_of_ten(f) is None]
    if len(exponents) < 2 or len(coefficients) != len(exponents) or None in coefficients:
        return None
    total = sum(coefficients, Fraction(0))
    exponent = sum(s * k for s, k in exponents)
    while abs(total) >= 10:  # written in standard form, as MalruleLib does
        total, exponent = total / 10, exponent + 1
    return _standard_form(total, exponent) if decimal_text(abs(total)) else None


def _plain_number(node: Node) -> Optional[str]:
    """The digits of a plain positive number as written (45000, 0.0032), else None."""
    if isinstance(node, Num) and node.text.replace(".", "", 1).isdigit() and node.value > 0:
        return node.text
    return None


def _correct_form(value: Fraction) -> tuple[Fraction, int]:
    exponent = 0
    while value >= 10:
        value, exponent = value / 10, exponent + 1
    while value < 1:
        value, exponent = value * 10, exponent - 1
    return value, exponent


@rule("scientific_notation.count_all_zeros_for_exponent")
def count_all_zeros_for_exponent(node: Node) -> Optional[Node]:
    """Uses the number of zeros as the exponent: 45000 = 4.5 × 10^3, 3050 = 3.05 × 10^2."""
    text = _plain_number(node)
    if text is None or "0" not in text:
        return None
    digits = text.replace(".", "").lstrip("0")
    coefficient = Fraction(f"{digits[0]}.{digits[1:] or '0'}")
    exponent = text.count("0") * (-1 if Fraction(text) < 1 else 1)
    if (coefficient, exponent) == _correct_form(Fraction(text)):
        return None
    return _standard_form(coefficient, exponent)


@rule("scientific_notation.wrong_exponent_sign")
def wrong_exponent_sign(node: Node) -> Optional[Node]:
    """Gives the exponent the wrong sign: 45000 = 4.5 × 10^-4, 0.0032 = 3.2 × 10^3."""
    text = _plain_number(node)
    if text is None:
        return None
    coefficient, exponent = _correct_form(Fraction(text))
    return None if exponent == 0 else _standard_form(coefficient, -exponent)
