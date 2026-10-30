"""Exponent rules. Each docstring gives MalruleLib's example.

Each rule rewrites its pattern wherever it appears in the line, and the answer is
the simplified result: (x^2)^3 · x becomes x^5 · x = x^6.
"""

from __future__ import annotations

from functools import reduce
from typing import Optional

import sympy

from ..maths import BinOp, Neg, Node, Num
from . import rule
from ._common import chain, changed_answer, power, value_of


def _number(node: Node) -> bool:
    return value_of(node) is not None


@rule("exponents.multiply_base_by_exponent")
def multiply_base_by_exponent(node: Node) -> Optional[Node]:
    """Multiplies the base by the exponent: 2^3 = 2 × 3 = 6."""
    def change(n: Node) -> Optional[Node]:
        p = power(n)
        return BinOp("*", *p) if p and _number(p[0]) and _number(p[1]) else None
    return changed_answer(node, change)


@rule("exponents.zero_exponent_equals_zero")
def zero_exponent_equals_zero(node: Node) -> Optional[Node]:
    """Thinks anything to the power 0 is 0: 5^0 = 0, 5x^0 = 0, x^3/x^3 = x^0 = 0."""
    zero = Num(sympy.Integer(0), "0")

    def change(n: Node) -> Optional[Node]:
        p = power(n)
        if p and value_of(p[1]) == 0:
            return zero
        if isinstance(n, BinOp) and n.op == "/" and power(n.left) and n.left == n.right:
            return zero  # equal powers divided: the exponents cancel to 0
        return None
    return changed_answer(node, change)


@rule("exponents.negative_exponent_makes_negative")
def negative_exponent_makes_negative(node: Node) -> Optional[Node]:
    """Thinks a negative exponent makes the answer negative: 2^-2 = -4."""
    def change(n: Node) -> Optional[Node]:
        p = power(n)
        if p and isinstance(p[1], Neg):
            return Neg(BinOp("^", p[0], p[1].operand))
        return None
    return changed_answer(node, change)


@rule("exponents.multiply_exponents_when_multiplying_powers")
def multiply_exponents_when_multiplying_powers(node: Node) -> Optional[Node]:
    """Multiplies the exponents when multiplying powers of the same base: x^3 × x^5 = x^15."""
    def change(n: Node) -> Optional[Node]:
        if not (isinstance(n, BinOp) and n.op == "*"):
            return None
        a, b = power(n.left), power(n.right)
        if a and b and a[0] == b[0]:
            return BinOp("^", a[0], BinOp("*", a[1], b[1]))
        return None
    return changed_answer(node, change)


@rule("exponents.add_exponents_for_power_of_power")
def add_exponents_for_power_of_power(node: Node) -> Optional[Node]:
    """Adds the exponents for a power of a power: (x^2)^3 = x^5. A coefficient is left as it is,
    as MalruleLib does: (2x^2)^3 = 2x^5."""
    def change(n: Node) -> Optional[Node]:
        outer = power(n)
        if not outer:
            return None
        inner, k = outer
        if power(inner):
            return BinOp("^", power(inner)[0], BinOp("+", power(inner)[1], k))
        if isinstance(inner, BinOp) and inner.op == "*" and _number(inner.left) and power(inner.right):
            base, m = power(inner.right)
            return BinOp("*", inner.left, BinOp("^", base, BinOp("+", m, k)))
        return None
    return changed_answer(node, change)


@rule("exponents.forget_exponent_on_coefficient")
def forget_exponent_on_coefficient(node: Node) -> Optional[Node]:
    """Raises the letters but not the number in front: (2x^3)^2 = 2x^6."""
    def change(n: Node) -> Optional[Node]:
        outer = power(n)
        if not outer:
            return None
        inner, k = outer
        factors = chain(inner, "*")
        numbers = [f for f in factors if _number(f)]
        letters = [f for f in factors if not _number(f)]
        if not numbers or not letters or len(factors) < 2:
            return None
        product = lambda parts: reduce(lambda a, b: BinOp("*", a, b), parts)  # noqa: E731
        return BinOp("*", product(numbers), BinOp("^", product(letters), k))
    return changed_answer(node, change)


@rule("exponents.distribute_exponent_over_addition")
def distribute_exponent_over_addition(node: Node) -> Optional[Node]:
    """Raises each term of a sum separately: (2 + 3)^2 = 4 + 9 = 13, (x + 3)^2 = x^2 + 9."""
    def change(n: Node) -> Optional[Node]:
        outer = power(n)
        if not outer or not (isinstance(outer[0], BinOp) and outer[0].op == "+"):
            return None
        total, k = outer
        return reduce(lambda a, b: BinOp("+", a, b), [BinOp("^", t, k) for t in chain(total, "+")])
    return changed_answer(node, change)
