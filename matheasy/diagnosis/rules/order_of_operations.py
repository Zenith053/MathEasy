"""Order-of-operations rules, run on a line of numbers joined by + - × ÷.

Each docstring gives MalruleLib's example. The rules read the line as the student
wrote it, left to right, and whether it has brackets: strict_left_to_right is for
lines without brackets and ignore_parentheses for lines with them, so the two
never compete.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Optional

from ..maths import Node
from . import rule
from ._common import decimal_node, evaluate, left_to_right, written_order


def _answer(node: Node, wrong: Optional[Fraction]) -> Optional[Node]:
    """The wrong value as an answer; None when the misconception happens to give the right one."""
    if wrong is None or wrong == evaluate(node):
        return None
    return decimal_node(wrong)


def _without_brackets(node: Node, allowed: str) -> Optional[tuple[list[Fraction], list[str]]]:
    found = written_order(node)
    if found is None or found[2] or not set(found[1]) <= set(allowed):
        return None
    return found[0], found[1]


@rule("order_of_operations.strict_left_to_right")
def strict_left_to_right(node: Node) -> Optional[Node]:
    """Works strictly from left to right, ignoring precedence: 2 + 3 × 4 = 20."""
    found = _without_brackets(node, "+-*/")
    return None if found is None else _answer(node, left_to_right(*found))


@rule("order_of_operations.ignore_parentheses")
def ignore_parentheses(node: Node) -> Optional[Node]:
    """Ignores the brackets and works from left to right: 2 × (3 + 4) = 6 + 4 = 10."""
    found = written_order(node)
    if found is None or not found[2]:
        return None
    return _answer(node, left_to_right(found[0], found[1]))


@rule("order_of_operations.addition_before_subtraction_always")
def addition_before_subtraction_always(node: Node) -> Optional[Node]:
    """Always adds before subtracting: 10 - 3 + 2 = 10 - 5 = 5."""
    found = _without_brackets(node, "+-")
    return None if found is None else _answer(node, left_to_right(*found, first="+"))


@rule("order_of_operations.multiplication_before_division_always")
def multiplication_before_division_always(node: Node) -> Optional[Node]:
    """Always multiplies before dividing: 6 ÷ 2 × 3 = 6 ÷ 6 = 1."""
    found = _without_brackets(node, "*/")
    return None if found is None else _answer(node, left_to_right(*found, first="*"))


@rule("order_of_operations.pemdas_strictly_sequential")
def pemdas_strictly_sequential(node: Node) -> Optional[Node]:
    """Treats PEMDAS as four separate steps, × then ÷ then + then −: 12 ÷ 2 × 3 = 12 ÷ 6 = 2.

    On a line of only + and − it agrees with addition_before_subtraction_always, and on
    only × and ÷ with multiplication_before_division_always; a line that mixes both
    kinds tells them apart.
    """
    found = _without_brackets(node, "+-*/")
    return None if found is None else _answer(node, left_to_right(*found, first="*/+"))
