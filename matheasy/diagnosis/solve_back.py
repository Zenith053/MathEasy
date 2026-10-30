"""solve_back(): which single operation went wrong, when a student turns a whole line into one number.

The tutor checks the question against the student's final answer, so a wrong answer
is usually a whole line collapsed into one number: 7 × 8 + 6 -> 60. locate() can only
say the whole line should be 62. solve_back() assumes exactly one operation went wrong
and the rest were done correctly, and for each operation in turn works out what it must
have given for the line to come out as the student's number:

    7 × 8 + 6 -> 60      7 × 8 -> 54      (then + 6 correctly)
                         56 + 6 -> 60     (7 × 8 correctly, then the addition went wrong)

Each piece is one operation on numbers, so the rules can run on it, and is returned as
a LocatedError. Only for lines of numbers: letters and equations give no pieces.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Optional, Union

import sympy

from .interfaces import RuleShortlister
from .maths import BinOp, LocatedError, Neg, Node, Sym, _kinds, as_fraction, locate, parse
from .rules._common import decimal_node, evaluate, fraction_node

_Y = Sym("y")


def solve_back(before: Union[str, Node], after: Union[str, Node]) -> list[LocatedError]:
    """One piece per operation that, alone done wrong, explains the student's number."""
    before = parse(before) if isinstance(before, str) else before
    after = parse(after) if isinstance(after, str) else after
    target = evaluate(after)
    if target is None or evaluate(before) is None or _operations(after):
        return []  # letters, an equation, a division by zero, or not one number (9^6 - 10)
    number = decimal_node if "decimals" in _kinds(before, after) else fraction_node
    pieces = []
    for op in _operations(before):
        value = _solve(_replace(before, op), target)
        if value is None:
            continue
        piece = BinOp(op.op, _given(op.left, number), _given(op.right, number))
        error = locate(piece, after if op is before else number(value))  # the last operation gave their own line
        if error is not None:
            pieces.append(error)
    return pieces


def rules_for(piece: LocatedError, rules: Optional[RuleShortlister] = None) -> list[str]:
    """The rules that reproduce this piece exactly."""
    return [c.id for c in (rules or RuleShortlister())(piece)]


def _is_operation(node: Node) -> bool:
    return isinstance(node, BinOp) and not as_fraction(node)  # 3/4 is a number, not an operation


def _operations(node: Node) -> list[BinOp]:
    found = [node] if _is_operation(node) else []
    for child in (getattr(node, k, None) for k in ("operand", "left", "right")):
        if child is not None:
            found += _operations(child)
    return found


def _replace(node: Node, target: Node) -> Node:
    """The tree with this one occurrence of `target` replaced by y."""
    if node is target:
        return _Y
    if isinstance(node, Neg):
        return Neg(_replace(node.operand, target))
    if _is_operation(node):
        return BinOp(node.op, _replace(node.left, target), _replace(node.right, target), node.typed_fraction)
    return node


def _solve(tree: Node, target: Fraction) -> Optional[Fraction]:
    """The one rational value of y that makes the tree equal the target, else None."""
    try:
        roots = sympy.solve(sympy.Eq(tree.to_sympy(), sympy.Rational(target.numerator, target.denominator)),
                            _Y.to_sympy())
    except (NotImplementedError, TypeError):
        return None
    roots = [r for r in roots if r.is_rational]
    return Fraction(int(roots[0].p), int(roots[0].q)) if len(roots) == 1 else None


def _given(node: Node, number) -> Node:
    """An operand as the student had it: a number stays as written, an operation becomes its value."""
    return number(evaluate(node)) if _operations(node) else node
