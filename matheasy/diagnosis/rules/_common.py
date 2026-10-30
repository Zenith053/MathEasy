"""Helpers shared by the rule modules: reading numbers from a tree and building answers."""

from __future__ import annotations

from fractions import Fraction
from typing import Callable, Optional

import sympy

from ..maths import BinOp, Mixed, Neg, Node, Num, Rel, Sym, as_fraction


def fraction_node(value: Fraction) -> Node:
    """A simplified fraction as a node: 2/4 -> 1/2, 4/2 -> 2, -1/2 -> -(1/2)."""
    if value < 0:
        return Neg(fraction_node(-value))
    num = Num(sympy.Integer(value.numerator), str(value.numerator))
    if value.denominator == 1:
        return num
    return BinOp("/", num, Num(sympy.Integer(value.denominator), str(value.denominator)))


def decimal_node(value: Fraction) -> Node:
    """A number written as a decimal when it terminates (0.75), else as a fraction (1/3)."""
    if value < 0:
        return Neg(decimal_node(-value))
    text = decimal_text(value)
    return fraction_node(value) if text is None else Num(sympy.Rational(value.numerator, value.denominator), text)


def decimal_text(value: Fraction) -> Optional[str]:
    """0.75 for 3/4, '2' for 2, None if the decimal does not terminate. Non-negative values only."""
    rest = value.denominator
    for factor in (2, 5):
        while rest % factor == 0:
            rest //= factor
    if rest != 1:
        return None
    places = next(p for p in range(64) if (value * 10 ** p).denominator == 1)
    digits = str(int(value * 10 ** places)).rjust(places + 1, "0")
    return digits if places == 0 else f"{digits[:-places]}.{digits[-places:]}"


def chain(node: Node, op: str) -> list[Node]:
    """a + b + c -> [a, b, c] for op '+'; a single term is a chain of one. A fraction is one term."""
    if isinstance(node, BinOp) and node.op == op and not as_fraction(node):
        return chain(node.left, op) + chain(node.right, op)
    return [node]


def value_of(node: Node) -> Optional[Fraction]:
    """The exact value of a number written with digits (2, 0.5, 3/4, 3 1/2, 25%, -2), else None."""
    if isinstance(node, Neg):
        inner = value_of(node.operand)
        return None if inner is None else -inner
    if isinstance(node, Num) and node.text != "π":
        return Fraction(int(node.value.p), int(node.value.q))
    if isinstance(node, Mixed):
        return Fraction(node.whole) + Fraction(node.num, node.den)
    fraction = as_fraction(node)
    if fraction is not None and fraction[1] != 0:
        return Fraction(*fraction)
    return None


def evaluate(node: Node, number: Callable[[Node], Optional[Fraction]] = value_of) -> Optional[Fraction]:
    """The value of a tree of numbers, reading each number with `number`; None if a letter, π,
    a division by zero or a power that is not a whole number is met."""
    if isinstance(node, BinOp) and not as_fraction(node):
        a, b = evaluate(node.left, number), evaluate(node.right, number)
        if a is None or b is None:
            return None
        if node.op == "^":
            if b.denominator != 1 or abs(b) > 50 or (a == 0 and b < 0):
                return None
            return a ** int(b)
        if node.op == "/" and b == 0:
            return None
        return {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else None}[node.op]
    if isinstance(node, Neg):
        inner = evaluate(node.operand, number)
        return None if inner is None else -inner
    return number(node)


def whole(node: Node) -> Optional[int]:
    """The whole number written as plain digits (12), else None."""
    return int(node.value) if isinstance(node, Num) and node.text.isdigit() else None


def solved(letter: Sym, value: Node) -> Rel:
    """x = value."""
    return Rel("=", letter, value)
