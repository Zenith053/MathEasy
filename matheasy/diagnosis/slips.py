"""bracket_slip(): a slip spotted inside one step, from the student's own inconsistency.

A misconception is applied consistently; a slip is not. When a step spreads one operation
over a bracket of three or more letter terms (a number times the bracket, or a minus in
front of it) and the student gets every term right but one, while a term like the wrong one
(same sign, also a letter term) came out right, they evidently know the rule: it is a slip.

    -2(x + y - z - k)  ->  -2x - 2y + 2z - 2k     slip: -z came out right, -k did not
    -2(x + y - z - k)  ->  -2x - 2y - 2z - 2k     not a slip: both minus terms wrong alike
    3(x + y + z + 4)   ->  3x + 3y + 3z + 4       not a slip: the number has no term like it

diagnose() runs it after the rules and solve back, before the LLM, so such a step costs no
model call. The step is then diagnosed as SLIP, topic expanding_brackets.
"""

from __future__ import annotations

from typing import Optional, Union

import sympy

from .maths import BinOp, Neg, Node, ParseError, Rel, Sym, parse

MIN_TERMS = 3


def bracket_slip(before: Union[str, Node], after: Union[str, Node]) -> Optional[str]:
    """Why this step looks like a slip, or None."""
    try:
        before = parse(before) if isinstance(before, str) else before
        after = parse(after) if isinstance(after, str) else after
    except ParseError:
        return None
    if isinstance(before, Rel) or isinstance(after, Rel):
        if not (isinstance(before, Rel) and isinstance(after, Rel) and before.op == after.op):
            return None
        return (bracket_slip(before.left, after.left) if _same(before.right, after.right)
                else bracket_slip(before.right, after.right) if _same(before.left, after.left) else None)
    try:
        change = sympy.expand(after.to_sympy() - before.to_sympy())
    except Exception:
        return None
    if change == 0 or len(sympy.Add.make_args(change)) != 1:
        return None  # right, or more than one term off
    _coeff, letters = change.as_coeff_Mul()
    if letters == 1:
        return None  # a number term: numbers combine, so there is no single term to blame
    for bracket in _distributed_brackets(before):
        terms = _signed_terms(bracket, 1)
        if len(terms) < MIN_TERMS:
            continue
        hit = [(sign, node) for sign, node in terms if _letters(node) == letters]
        if len(hit) != 1 or _uses_elsewhere(before, bracket, letters):
            continue
        sign, wrong = hit[0]
        like = [node for s, node in terms if node is not wrong and s == sign and _letters(node) != 1]
        if like:
            return (f"spread the operation correctly to {_show(sign, like[0])} but not to {_show(sign, wrong)} "
                    f"in the same bracket")
    return None


def _same(a: Node, b: Node) -> bool:
    try:
        return sympy.simplify(a.to_sympy() - b.to_sympy()) == 0
    except Exception:
        return False


def _is_sum(node: Node) -> bool:
    return isinstance(node, BinOp) and node.op in "+-"


def _distributed_brackets(node: Node):
    """Brackets an operation is spread over: k × (…), (…) × k, -(…) and a - (…)."""
    if isinstance(node, BinOp):
        if node.op == "*":
            for side, other in ((node.left, node.right), (node.right, node.left)):
                if _is_sum(side) and not _is_sum(other):
                    yield side
        elif node.op == "-" and _is_sum(node.right):
            yield node.right
    elif isinstance(node, Neg) and _is_sum(node.operand):
        yield node.operand
    for child in (getattr(node, k, None) for k in ("operand", "left", "right")):
        if child is not None:
            yield from _distributed_brackets(child)


def _signed_terms(node: Node, sign: int) -> list[tuple[int, Node]]:
    """a - b + c -> [(+1, a), (-1, b), (+1, c)]."""
    if isinstance(node, BinOp) and node.op == "+":
        return _signed_terms(node.left, sign) + _signed_terms(node.right, sign)
    if isinstance(node, BinOp) and node.op == "-":
        return _signed_terms(node.left, sign) + _signed_terms(node.right, -sign)
    if isinstance(node, Neg):
        return _signed_terms(node.operand, -sign)
    return [(sign, node)]


def _letters(node: Node):
    try:
        return sympy.expand(node.to_sympy()).as_coeff_Mul()[1]
    except Exception:
        return None


def _uses_elsewhere(whole: Node, bracket: Node, letters) -> bool:
    """The letters also appear outside this bracket, so the change cannot be pinned on one term."""
    names = {str(s) for s in letters.free_symbols}
    count = lambda n: sum(isinstance(m, Sym) and m.name in names for m in _walk(n))
    return count(whole) > count(bracket)


def _walk(node: Node):
    yield node
    for child in (getattr(node, k, None) for k in ("operand", "left", "right")):
        if child is not None:
            yield from _walk(child)


def _show(sign: int, node: Node) -> str:
    return f"-{node}" if sign < 0 else str(node)
