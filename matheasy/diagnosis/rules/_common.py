"""Helpers shared by the rule modules: reading numbers from a tree and building answers."""

from __future__ import annotations

from fractions import Fraction
from typing import Callable, Optional

import sympy

from ..maths import BinOp, Mixed, Neg, Node, Num, ParseError, Rel, Sym, _sstr, as_fraction, equivalent, parse


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


Combine = Callable[[str, Fraction, Fraction], Optional[Fraction]]


def combine(op: str, a: Fraction, b: Fraction) -> Optional[Fraction]:
    """a op b done correctly; None for a division by zero or a power that is not a small whole number."""
    if op == "^":
        if b.denominator != 1 or abs(b) > 50 or (a == 0 and b < 0):
            return None
        return a ** int(b)
    if op == "/":
        return None if b == 0 else a / b
    return {"+": a + b, "-": a - b, "*": a * b}[op]


def evaluate(node: Node, number: Callable[[Node], Optional[Fraction]] = value_of,
             apply: Combine = combine) -> Optional[Fraction]:
    """The value of a tree of numbers, reading each number with `number` and doing each
    operation with `apply`; None if a letter or π is met, or `apply` gives up."""
    if isinstance(node, BinOp) and not as_fraction(node):
        a, b = evaluate(node.left, number, apply), evaluate(node.right, number, apply)
        return None if a is None or b is None else apply(node.op, a, b)
    if isinstance(node, Neg):
        inner = evaluate(node.operand, number, apply)
        return None if inner is None else -inner
    return number(node)


_LEVEL = {"+": 1, "-": 1, "*": 2, "/": 2}


def written_order(node: Node) -> Optional[tuple[list[Fraction], list[str], bool]]:
    """2 × (3 + 4) -> ([2, 3, 4], ['*', '+'], True): the numbers and operations as written,
    and whether the student used brackets. None unless the line is numbers joined by + - × ÷."""
    values: list[Fraction] = []
    ops: list[str] = []
    bracketed = False

    def walk(n: Node, parent: Optional[str], right: bool) -> bool:
        nonlocal bracketed
        if isinstance(n, BinOp) and not n.typed_fraction:  # 6 ÷ 2 is an operation, a typed 6/2 a number
            if n.op not in _LEVEL:
                return False
            if parent and (_LEVEL[n.op] < _LEVEL[parent] or (right and _LEVEL[n.op] == _LEVEL[parent])):
                bracketed = True  # it groups differently from left to right, so it was in brackets
            if not walk(n.left, n.op, False):
                return False
            ops.append(n.op)
            return walk(n.right, n.op, True)
        value = value_of(n)
        if value is None:
            return False
        values.append(value)
        return True

    return (values, ops, bracketed) if walk(node, None, False) and ops else None


def left_to_right(values: list[Fraction], ops: list[str], first: str = "") -> Optional[Fraction]:
    """Does every operation in `first` (in order of appearance, joining its neighbours), then the
    rest from left to right: first='+' does all additions before any subtraction."""
    values, ops = list(values), list(ops)
    for wanted in first:
        i = 0
        while i < len(ops):
            if ops[i] == wanted:
                result = combine(ops[i], values[i], values[i + 1])
                if result is None:
                    return None
                values[i:i + 2] = [result]
                del ops[i]
            else:
                i += 1
    result = values[0]
    for op, value in zip(ops, values[1:]):
        result = combine(op, result, value)
        if result is None:
            return None
    return result


def whole(node: Node) -> Optional[int]:
    """The whole number written as plain digits (12), else None."""
    return int(node.value) if isinstance(node, Num) and node.text.isdigit() else None


def solved(letter: Sym, value: Node) -> Rel:
    """x = value."""
    return Rel("=", letter, value)


def rewrite(node: Node, change: Callable[[Node], Optional[Node]]) -> Node:
    """The tree with `change` applied bottom-up wherever it gives a replacement."""
    if isinstance(node, BinOp) and not node.typed_fraction:
        node = BinOp(node.op, rewrite(node.left, change), rewrite(node.right, change))
    elif isinstance(node, Rel):
        node = Rel(node.op, rewrite(node.left, change), rewrite(node.right, change))
    elif isinstance(node, Neg):
        node = Neg(rewrite(node.operand, change))
    return change(node) or node


def simplified(node: Node) -> Node:
    """The value of a rewritten tree, written the way a student would (x^6, x^2 + 9, 13)."""
    try:
        value = sympy.simplify(node.to_sympy())
        if value.free_symbols:
            value = sympy.expand(value)
        return parse(_sstr(value))
    except (ParseError, TypeError, ValueError, ZeroDivisionError):
        return node


def changed_answer(node: Node, change: Callable[[Node], Optional[Node]]) -> Optional[Node]:
    """The student's answer if they apply `change` wherever it fits; None if it fits nowhere
    or does not change the value."""
    wrong = rewrite(node, change)
    if wrong == node:
        return None
    try:
        if equivalent(wrong, node):
            return None
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return simplified(wrong)


def power(node: Node) -> Optional[tuple[Node, Node]]:
    """(base, exponent) if node is base^exponent."""
    return (node.left, node.right) if isinstance(node, BinOp) and node.op == "^" else None
