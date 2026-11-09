"""question_topics(): the concepts a question visibly uses, read from its parsed form.

    question_topics("1/2 + 1/3")        -> ('fractions',)
    question_topics("4x + 2x - 3")      -> ('algebraic_notation', 'collecting_like_terms')
    question_topics("2(x + 3) = 14")    -> ('algebraic_notation', 'expanding_brackets', 'solving_equations')

The planner credits these when a student answers correctly, so a concept is credited even when
no rule covers it (no rule tests collecting like terms, yet getting 4x + 2x right is evidence of
it). Only what can be read off the question is named, from the TOPICS list in explain.py:
fractions, decimals, percentages, negative numbers, powers, standard form, order of operations,
column arithmetic on whole numbers, letters, equations, proportions, brackets to expand and like
terms. Beliefs about what multiplying or dividing does are left to the rules, which know when a
question tests them.
"""

from __future__ import annotations

from typing import Iterator, Union

from .maths import BinOp, Mixed, Neg, Node, Num, ParseError, Rel, Sym, as_fraction, parse


def question_topics(question: Union[str, Node]) -> tuple[str, ...]:
    try:
        node = parse(question) if isinstance(question, str) else question
    except ParseError:
        return ()
    found: set[str] = set()
    nodes = list(_walk(node))
    for n in nodes:
        if isinstance(n, Mixed) or _typed_fraction(n):
            found.add("fractions")
        elif isinstance(n, Num):
            if "%" in n.text:
                found.add("percentages")
            elif "." in n.text:
                found.add("decimals")
        elif isinstance(n, Sym):
            found.add("algebraic_notation")
        elif isinstance(n, Neg) and not _has_letter(n.operand):
            found.add("negative_numbers")
        elif isinstance(n, BinOp) and not as_fraction(n):
            if n.op == "^":
                found.add("scientific_notation" if _is_ten(n.left) else "exponents")
                if _sum_with_letter(n.left):
                    found.add("expanding_brackets")  # (x + 3)^2
            elif n.op == "*" and (_sum_with_letter(n.left) or _sum_with_letter(n.right)):
                found.add("expanding_brackets")      # 2(x + 3)
            elif n.op == "-" and _sum_with_letter(n.right):
                found.add("expanding_brackets")      # 5 - (x + 1): the minus reaches every term
            if n.op in "+-*" and _whole_at_least_10(n.left) and _whole_at_least_10(n.right):
                found.add("whole_number_arithmetic")
    if isinstance(node, Rel) and node.op == "=" and _has_letter(node):
        found.add("ratios_proportions" if as_fraction(node.left) or as_fraction(node.right) or _letter_fraction(node)
                  else "solving_equations")
    ops = {n.op for n in nodes if isinstance(n, BinOp) and not as_fraction(n) and n.op in "+-*/"}
    if ops & {"+", "-"} and ops & {"*", "/"} and not _has_letter(node):
        found.add("order_of_operations")
    for side in (node.left, node.right) if isinstance(node, Rel) else (node,):
        if _has_like_terms(side):
            found.add("collecting_like_terms")
    return tuple(sorted(found))


def _typed_fraction(node: Node) -> bool:
    """3/4 typed as one number; 12 ÷ 3 is a division, not a fraction."""
    return bool(as_fraction(node)) and getattr(node, "typed_fraction", False)


def _walk(node: Node) -> Iterator[Node]:
    yield node
    for child in (getattr(node, k, None) for k in ("operand", "left", "right")):
        if child is not None:
            yield from _walk(child)


def _has_letter(node: Node) -> bool:
    return any(isinstance(n, Sym) for n in _walk(node))


def _sum_with_letter(node: Node) -> bool:
    return isinstance(node, BinOp) and node.op in "+-" and _has_letter(node)


def _is_ten(node: Node) -> bool:
    return isinstance(node, Num) and node.text == "10"


def _whole_at_least_10(node: Node) -> bool:
    return isinstance(node, Num) and node.text.isdigit() and int(node.text) >= 10


def _letter_fraction(node: Rel) -> bool:
    """x/3 = 6/9: a proportion with the unknown in a fraction."""
    return any(isinstance(side, BinOp) and side.op == "/" for side in (node.left, node.right))


def _has_like_terms(node: Node) -> bool:
    """Two terms of a sum with the same letters and powers: 4x + 2x, 3a^2 - a^2."""
    if not _has_letter(node):
        return False
    seen = set()
    for term in _terms(node):
        if not _has_letter(term):
            continue
        try:
            _coeff, letters = term.to_sympy().as_coeff_Mul()
        except Exception:
            continue
        if letters in seen:
            return True
        seen.add(letters)
    return False


def _terms(node: Node) -> Iterator[Node]:
    """The terms of a sum as written, signs dropped: a - b + c -> a, b, c."""
    if isinstance(node, BinOp) and node.op in "+-":
        yield from _terms(node.left)
        yield from _terms(node.right)
    elif isinstance(node, Neg):
        yield from _terms(node.operand)
    else:
        yield node
