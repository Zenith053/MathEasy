"""Algebra rules, mostly run on a linear equation in one letter. Each docstring gives MalruleLib's example."""

from __future__ import annotations

from fractions import Fraction
from typing import Optional, Union

import sympy

from ..maths import BinOp, Node, Rel, Sym, Word
from . import rule
from ._common import changed_answer, decimal_node, power, solved, value_of

Letter = Union[Sym, Word]


def _letters(node: Node) -> dict[str, Letter]:
    if isinstance(node, (Sym, Word)):
        return {node.name: node}
    found: dict[str, Letter] = {}
    for child in (getattr(node, k, None) for k in ("operand", "left", "right")):
        if child is not None:
            found.update(_letters(child))
    return found


def _linear(node: Node) -> Optional[tuple[Letter, Fraction, Fraction, Fraction, Fraction]]:
    """(x, a, b, c, d) for an equation that reads a·x + b = c·x + d in one letter."""
    if not (isinstance(node, Rel) and node.op == "="):
        return None
    letters = _letters(node)
    if len(letters) != 1:
        return None
    letter = next(iter(letters.values()))
    x = sympy.Symbol(letter.name)
    sides = []
    for side in (node.left, node.right):
        expr = sympy.expand(side.to_sympy())
        if not expr.is_polynomial(x) or sympy.degree(expr, x) > 1:
            return None
        a, b = expr.coeff(x, 1), expr.coeff(x, 0)
        if not (a.is_Rational and b.is_Rational):
            return None
        sides += [Fraction(int(a.p), int(a.q)), Fraction(int(b.p), int(b.q))]
    return (letter, *sides)


def _solution(node: Node) -> Optional[Fraction]:
    found = _linear(node)
    if found is None or found[1] == found[3]:
        return None
    _, a, b, c, d = found
    return (d - b) / (a - c)


def _answer(node: Node, letter: Letter, wrong: Fraction) -> Optional[Node]:
    return None if wrong == _solution(node) else solved(letter, decimal_node(wrong))


@rule("algebra.cancel_across_equals")
def cancel_across_equals(node: Node) -> Optional[Node]:
    """Cancels the letter terms on the right as if they were factors: 5x = 2x + 12 gives x = 12/5,
    and 5x + 3 = 2x + 12 gives x = (12 - 3)/5."""
    found = _linear(node)
    if found is None:
        return None
    letter, a, b, c, d = found
    if a == 0 or c == 0:
        return None
    return _answer(node, letter, (d - b) / a)


@rule("algebra.divide_one_term_only")
def divide_one_term_only(node: Node) -> Optional[Node]:
    """Divides only the letter term: 2x + 4 = 10 gives x + 4 = 5, so x = 1."""
    found = _linear(node)
    if found is None:
        return None
    letter, a, b, c, d = found
    if a == 0:
        a, b, c, d = c, d, a, b  # the letter is on the right: 10 = 2x + 4
    if c != 0 or a in (0, 1) or b == 0:
        return None
    return _answer(node, letter, d / a - b)


_ASSUMED = {"x": 10, "n": 1, "y": 25}  # MalruleLib's values; any other letter is its place in the alphabet


@rule("algebra.variable_letter_has_value")
def variable_letter_has_value(node: Node) -> Optional[Node]:
    """Thinks a letter stands for a fixed number, such as its place in the alphabet: x + 5 = 12 gives x = 10."""
    found = _linear(node)
    if found is None or not isinstance(found[0], Sym) or len(found[0].name) != 1:
        return None
    letter = found[0]
    value = _ASSUMED.get(letter.name.lower(), ord(letter.name.lower()) - ord("a") + 1)
    return _answer(node, letter, Fraction(value))


@rule("algebra.distribute_over_non_distributive")
def distribute_over_non_distributive(node: Node) -> Optional[Node]:
    """Squares a bracket term by term and drops the sign: (x + 3)^2 = x^2 + 9, (x - 3)^2 = x^2 + 9."""
    def change(n: Node) -> Optional[Node]:
        p = power(n)
        if not (p and value_of(p[1]) == 2 and isinstance(p[0], BinOp) and p[0].op in "+-"):
            return None
        letter, constant = p[0].left, value_of(p[0].right)
        if not isinstance(letter, (Sym, Word)) or constant is None:
            return None
        square = Fraction(constant) ** 2
        return BinOp("+", BinOp("^", letter, p[1]), decimal_node(square))
    return changed_answer(node, change)
