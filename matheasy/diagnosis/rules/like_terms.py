"""Like-term rules: our own, not MalruleLib's (to-do T2 in docs/design_decisions.md). Each looks
at two terms added together, such as x + x or 3x + 2."""

from __future__ import annotations

from fractions import Fraction
from typing import Optional

import sympy

from ..maths import BinOp, Neg, Node, Num, Sym
from . import rule
from ._common import changed_answer, fraction_node, value_of

Monomial = tuple[Fraction, tuple[str, ...], bool]  # (coefficient, letters, coefficient written?)


def _monomial(node: Node) -> Optional[Monomial]:
    """3x -> (3, ('x',), True); x -> (1, ('x',), False); 2 -> (2, (), True); 4ab -> (4, ('a', 'b'), True).
    None for anything else, such as a power or a bracket."""
    if isinstance(node, Neg):
        inner = _monomial(node.operand)
        return None if inner is None else (-inner[0], inner[1], True)
    if isinstance(node, Sym):
        return Fraction(1), (node.name,), False
    value = value_of(node)
    if value is not None:
        return value, (), True
    if isinstance(node, BinOp) and node.op == "*" and not node.typed_fraction:
        a, b = _monomial(node.left), _monomial(node.right)
        if a is not None and b is not None:
            return a[0] * b[0], tuple(sorted(a[1] + b[1])), a[2] or b[2]
    return None


def _term(coefficient: Fraction, letters: Node) -> Node:
    return letters if coefficient == 1 else BinOp("*", fraction_node(coefficient), letters)


def _not_a_power(node: Node) -> bool:
    """Like terms are collected outside powers only: (x + 3)^2 is a bracket to expand, not 4x squared."""
    return not (isinstance(node, BinOp) and node.op == "^")


def _pair(node: Node) -> Optional[tuple[Monomial, Monomial]]:
    if not (isinstance(node, BinOp) and node.op == "+"):
        return None
    a, b = _monomial(node.left), _monomial(node.right)
    return None if a is None or b is None else (a, b)


@rule("like_terms.add_letters_as_powers")
def add_letters_as_powers(node: Node) -> Optional[Node]:
    """Adds like terms by multiplying the letters, e.g. x + x = x^2 and 2x + 3x = 5x^2."""
    def change(n: Node) -> Optional[Node]:
        pair = _pair(n)
        if pair is None:
            return None
        (a, letters_a, written_a), (b, letters_b, written_b) = pair
        if letters_a != letters_b or len(letters_a) != 1:
            return None
        coefficient = a + b if written_a or written_b else Fraction(1)  # x + x: no number is written
        if coefficient == 0:
            return None
        return _term(coefficient, BinOp("^", Sym(letters_a[0]), Num(sympy.Integer(2), "2")))
    return changed_answer(node, change, enter=_not_a_power)


@rule("like_terms.combine_unlike_terms")
def combine_unlike_terms(node: Node) -> Optional[Node]:
    """Adds terms that are not alike, e.g. 3x + 2 = 5x and 4a + 3b = 7ab."""
    def change(n: Node) -> Optional[Node]:
        pair = _pair(n)
        if pair is None:
            return None
        (a, letters_a, _), (b, letters_b, _) = pair
        if letters_a == letters_b or len(letters_a) > 1 or len(letters_b) > 1:
            return None
        if a + b == 0:
            return None
        letters = sorted(set(letters_a + letters_b))
        product: Node = Sym(letters[0])
        for letter in letters[1:]:
            product = BinOp("*", product, Sym(letter))
        return _term(a + b, product)
    return changed_answer(node, change, enter=_not_a_power)
