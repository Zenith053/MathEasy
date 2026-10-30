"""Ratio and proportion rules, run on a proportion with one unknown: 2/3 = 6/x.

MalruleLib poses these as word problems (recipes, maps, speeds); the same wrong
methods show on the proportion the student writes down.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Optional, Union

from ..maths import BinOp, Node, Rel, Sym
from . import rule
from ._common import fraction_node, solved, whole

Part = Union[int, Sym]


def _proportion(node: Node) -> Optional[tuple[list[Part], int]]:
    """([a, b, c, d], i) for a/b = c/d where exactly one of the four, at index i, is a letter."""
    if not (isinstance(node, Rel) and node.op == "="):
        return None
    parts: list[Part] = []
    for side in (node.left, node.right):
        if not (isinstance(side, BinOp) and side.op == "/"):
            return None
        for term in (side.left, side.right):
            n = whole(term)
            if n is not None and n != 0:
                parts.append(n)
            elif isinstance(term, Sym):
                parts.append(term)
            else:
                return None
    letters = [i for i, p in enumerate(parts) if isinstance(p, Sym)]
    return (parts, letters[0]) if len(letters) == 1 else None


# Positions: 0 = a, 1 = b, 2 = c, 3 = d in a/b = c/d.
_PARTNER = {0: 2, 1: 3, 2: 0, 3: 1}   # the part it corresponds to across the equals sign
_SAME_SIDE = {0: 1, 1: 0, 2: 3, 3: 2}  # the other part of its own ratio


@rule("ratios_proportions.additive_instead_of_multiplicative")
def additive_instead_of_multiplicative(node: Node) -> Optional[Node]:
    """Keeps the difference instead of the ratio: 2/3 = 6/x gives x = 3 + (6 - 2) = 7."""
    found = _proportion(node)
    if found is None:
        return None
    parts, i = found
    mate = _SAME_SIDE[i]                             # c, beside the unknown in c/x
    shift = parts[mate] - parts[_PARTNER[mate]]      # c - a: how much the known pair changed
    return solved(parts[i], fraction_node(Fraction(parts[_PARTNER[i]] + shift)))  # b + (c - a)


@rule("ratios_proportions.swap_ratios_or_units", twins=("ratios_proportions.ratio_as_division_only",))
def swap_ratios_or_units(node: Node) -> Optional[Node]:
    """Sets one ratio up the wrong way round: 2/3 = 6/x is solved as 3/2 = 6/x, so x = 4 (not 9).

    Twin: ratio_as_division_only, which reads a : b only as the quotient a ÷ b and scales
    by it (2/3 = 6/x gives x = 6 × 2/3 = 4). MalruleLib asks it only with the unknown at
    the bottom, where the two always agree.
    """
    found = _proportion(node)
    if found is None:
        return None
    parts, i = found
    known = [0, 1] if i in (2, 3) else [2, 3]
    ratio = Fraction(parts[known[1]], parts[known[0]])  # the complete ratio, upside down
    other = parts[_SAME_SIDE[i]]
    return solved(parts[i], fraction_node(Fraction(other / ratio if i in (1, 3) else other * ratio)))
