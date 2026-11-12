"""Bracket rules: our own, not MalruleLib's (MalruleLib has no rule for these mistakes; to-do T2
in docs/design_decisions.md). Each is a consistent mistake: applied the same way to every
bracket. A bracket done right on every term but one is a slip, found by slips.py instead.

Each rule needs a letter inside the bracket. On numbers alone (2 × (3 + 4)) a student works
the bracket out first, and the order-of-operations rules cover what goes wrong there.
"""

from __future__ import annotations

from typing import Optional

from ..maths import BinOp, Neg, Node, Sym, Word
from . import rule
from ._common import changed_answer, value_of

Terms = list[tuple[int, Node]]  # (+1 or -1, term), in the order written


def _has_letter(node: Node) -> bool:
    if isinstance(node, (Sym, Word)):
        return True
    return any(_has_letter(child) for child in (getattr(node, k, None) for k in ("operand", "left", "right"))
               if child is not None)


def _is_sum(node: Node) -> bool:
    return isinstance(node, BinOp) and node.op in "+-" and not node.typed_fraction


def _terms(node: Node) -> Terms:
    """x - 3 + y -> [(+1, x), (-1, 3), (+1, y)]. A bracket inside the bracket stays one term."""
    if _is_sum(node):
        return _terms(node.left) + [(1 if node.op == "+" else -1, node.right)]
    return [(1, node)]


def _join(terms: Terms) -> Node:
    (sign, first), rest = terms[0], terms[1:]
    out = first if sign > 0 else Neg(first)
    for sign, term in rest:
        out = BinOp("+" if sign > 0 else "-", out, term)
    return out


def _bracket(node: Node) -> Optional[tuple[Node, Node]]:
    """(multiplier, bracket) for k(a + b) or (a + b) × k with a letter in the bracket; the
    multiplier is a single term, not a sum."""
    if not (isinstance(node, BinOp) and node.op == "*" and not node.typed_fraction):
        return None
    for multiplier, bracket in ((node.left, node.right), (node.right, node.left)):
        if _is_sum(bracket) and not _is_sum(multiplier) and _has_letter(bracket):
            return multiplier, bracket
    return None


@rule("brackets.multiply_first_term_only")
def multiply_first_term_only(node: Node) -> Optional[Node]:
    """Multiplies only the first term inside the bracket, e.g. 2(x + 3) = 2x + 3."""
    def change(n: Node) -> Optional[Node]:
        found = _bracket(n)
        if found is None:
            return None
        multiplier, bracket = found
        (sign, first), *rest = _terms(bracket)
        return _join([(sign, BinOp("*", multiplier, first)), *rest])
    return changed_answer(node, change)


@rule("brackets.minus_first_term_only")
def minus_first_term_only(node: Node) -> Optional[Node]:
    """Changes the sign of only the first term after a minus in front of a bracket,
    e.g. -(x + 1) = -x + 1 and 5 - (x - 2) = 5 - x - 2."""
    def change(n: Node) -> Optional[Node]:
        if isinstance(n, Neg) and _is_sum(n.operand) and _has_letter(n.operand):
            (sign, first), *rest = _terms(n.operand)
            return _join([(-sign, first), *rest])
        if isinstance(n, BinOp) and n.op == "-" and _is_sum(n.right) and _has_letter(n.right):
            (sign, first), *rest = _terms(n.right)
            return BinOp("+", n.left, _join([(-sign, first), *rest]))
        return None
    return changed_answer(node, change)


@rule("brackets.negative_multiplier_first_term_only")
def negative_multiplier_first_term_only(node: Node) -> Optional[Node]:
    """Multiplies every term by the size of a negative number but gives its sign only to the
    first term, e.g. -2(x - 3) = -2x - 6 (right: -2x + 6)."""
    def change(n: Node) -> Optional[Node]:
        found = _bracket(n)
        if found is None:
            return None
        multiplier, bracket = found
        value = value_of(multiplier)
        if value is None or value >= 0 or not isinstance(multiplier, Neg):
            return None
        (sign, first), *rest = _terms(bracket)
        size = multiplier.operand
        return _join([(sign, BinOp("*", multiplier, first)), *[(s, BinOp("*", size, t)) for s, t in rest]])
    return changed_answer(node, change)
