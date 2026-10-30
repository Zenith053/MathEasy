"""The held-back rules as code, for D's simulated students only (owner: D).

These 10 MalruleLib rules (eval/held_back_rules.json) never enter B's bank or rule
runner, so the tutor cannot run them. A student holding one still has to answer by
it, so D implements them here. SimStudent picks them up by name:

    from matheasy.evaluation import SimStudent
    SimStudent("s1", "fractions.subtract_across", judge=judge).answer("3/4 - 1/2")   # ('1', True)

Each predictor takes the item as text and returns the answer the misconception
predicts, or None if it says nothing about that item. The wrong answers follow
MalruleLib's calculate_expected_answer() (Chen, Liu and Sonkar, 2026,
https://github.com/sonkar-lab/malrulelib), run on any item of the same shape.

Comparison items (fractions.denominator_comparison_error, decimals.shorter_is_larger)
are written as a relation between two numbers, "1/3 > 1/4": the sign in the item is
ignored, and the answer is the relation the student would write. The item format
for comparisons still needs agreement with B and C (B's comparison rules need one too).

Never import this module from the tutor: the tutor must not know these rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Callable, Optional

import sympy

from ..diagnosis import ParseError, as_fraction, held_back_rules, parse
from ..diagnosis.maths import BinOp, Neg, Node, Num, Rel, Sym


@dataclass(frozen=True)
class HeldBackRule:
    name: str
    description: str  # MalruleLib's, for the feedback judge
    predict: Callable[[str], Optional[str]]


_RULES: dict[str, HeldBackRule] = {}


def _held_back(name: str, description: str):
    def register(fn: Callable[[Node], Optional[str]]):
        def predict(item: str) -> Optional[str]:
            try:
                return fn(parse(item))
            except (ParseError, ZeroDivisionError):
                return None
        _RULES[name] = HeldBackRule(name, description, predict)
        return fn
    return register


def held_back_rule(name: str) -> HeldBackRule:
    try:
        return _RULES[name]
    except KeyError:
        raise KeyError(f"no held-back rule {name}") from None


def held_back_names() -> list[str]:
    return sorted(_RULES)


# ---------------------------------------------------------------- reading items

def _int(node: Node) -> Optional[int]:
    """A whole number, possibly negative: 5, -5, (-5)."""
    if isinstance(node, Num) and node.text.isdigit():
        return int(node.text)
    if isinstance(node, Neg):
        value = _int(node.operand)
        return None if value is None else -value
    return None


def _decimal(node: Node) -> Optional[str]:
    """A non-negative number as written, if it is a plain decimal or whole number: '0.25', '3'."""
    if isinstance(node, Num) and "%" not in node.text and node.text.replace(".", "", 1).isdigit():
        return node.text
    return None


def _show(value) -> str:
    """A number the way a student writes it: 31.0 -> 31, Fraction(3, 2) -> 3/2."""
    if isinstance(value, Decimal):
        text = format(value.normalize(), "f")
        return text if text != "-0" else "0"
    return str(value)


def _relation(left: Node, right: Node, student_thinks_left_larger: Optional[bool]) -> str:
    """The relation the student writes, in the item's order. None means 'they are equal'."""
    op = "=" if student_thinks_left_larger is None else ">" if student_thinks_left_larger else "<"
    return f"{left} {op} {right}"


def _comparison(node: Node) -> Optional[tuple[Node, Node]]:
    if isinstance(node, Rel) and node.op in ("<", ">", "<=", ">=", "="):
        return node.left, node.right
    return None


def _subtraction_chain(node: Node) -> list[Node]:
    """a - b - c -> [a, b, c]; a typed fraction is one term."""
    if isinstance(node, BinOp) and node.op == "-" and as_fraction(node) is None:
        return _subtraction_chain(node.left) + [node.right]
    return [node]


# ---------------------------------------------------------------- fractions

@_held_back("fractions.denominator_comparison_error",
            "Believes a larger denominator means a larger fraction, e.g. 1/4 > 1/3 because 4 > 3.")
def _denominator_comparison_error(node: Node) -> Optional[str]:
    pair = _comparison(node)
    if pair is None:
        return None
    a, b = (as_fraction(side) for side in pair)
    if a is None or b is None:
        return None
    if a[1] != b[1]:
        return _relation(*pair, a[1] > b[1])
    return _relation(*pair, None if a[0] == b[0] else a[0] > b[0])  # same denominator: compares numerators, correctly


@_held_back("fractions.subtract_across",
            "Subtracts the numerators and subtracts the denominators, e.g. 3/4 - 1/2 = 2/2 = 1.")
def _subtract_across(node: Node) -> Optional[str]:
    parts = [as_fraction(t) for t in _subtraction_chain(node)]
    if len(parts) < 2 or any(p is None or p[1] == 0 for p in parts):
        return None
    num = parts[0][0] - sum(p[0] for p in parts[1:])
    den = parts[0][1] - sum(p[1] for p in parts[1:])
    if den == 0:
        return None  # MalruleLib writes n/0; no student answer to compare
    return _show(Fraction(num, den))


@_held_back("ratios_proportions.each_fraction_digit_is_ratio",
            "Simplifies a fraction digit by digit, treating each pair of digits as its own ratio, "
            "e.g. 12/34 becomes 1/3 and 2/4, written 11/32.")
def _each_fraction_digit_is_ratio(node: Node) -> Optional[str]:
    fraction = as_fraction(node)
    if fraction is None or max(len(str(fraction[0])), len(str(fraction[1]))) < 2:
        return None
    width = max(len(str(fraction[0])), len(str(fraction[1])))
    nums, dens = str(fraction[0]).zfill(width), str(fraction[1]).zfill(width)
    out_num, out_den = "", ""
    for n, d in zip(map(int, nums), map(int, dens)):
        d = d or 1
        if n == 0:
            n, d = 0, 1
        else:
            pair = Fraction(n, d)
            n, d = pair.numerator, pair.denominator
        out_num += str(n)
        out_den += str(d)
    return f"{int(out_num)}/{int(out_den)}"


# ---------------------------------------------------------------- decimals

@_held_back("decimals.right_align_decimals",
            "Lines decimals up by their last digit instead of by the decimal point when adding or "
            "subtracting, e.g. 3.5 + 2.75 = 31.0 (35 + 275 = 310, one decimal place).")
def _right_align_decimals(node: Node) -> Optional[str]:
    if not (isinstance(node, BinOp) and node.op in "+-"):
        return None
    a, b = _decimal(node.left), _decimal(node.right)
    if a is None or b is None or ("." not in a and "." not in b):
        return None
    digits_a, digits_b = a.replace(".", ""), b.replace(".", "")
    width = max(len(digits_a), len(digits_b))
    x, y = int(digits_a.zfill(width)), int(digits_b.zfill(width))
    result = x + y if node.op == "+" else abs(x - y)
    places = min(len(a.partition(".")[2]), len(b.partition(".")[2]))
    text = str(result)
    if places > 0 and len(text) > places:
        return _show(Decimal(text[:-places] + "." + text[-places:]))
    return _show(Decimal(result))


@_held_back("decimals.shorter_is_larger",
            "Believes a decimal with fewer decimal places is larger, e.g. 0.4 > 0.45 because tenths "
            "are bigger than hundredths.")
def _shorter_is_larger(node: Node) -> Optional[str]:
    pair = _comparison(node)
    if pair is None:
        return None
    a, b = (_decimal(side) for side in pair)
    if a is None or b is None or ("." not in a and "." not in b):
        return None
    places_a, places_b = len(a.partition(".")[2]), len(b.partition(".")[2])
    if places_a != places_b:
        return _relation(*pair, places_a < places_b)
    return _relation(*pair, None if Decimal(a) == Decimal(b) else Decimal(a) > Decimal(b))


@_held_back("percentages.percentage_as_index",
            "Treats a percentage as a position or a count rather than a proportion, e.g. 30% of 50 = 30.")
def _percentage_as_index(node: Node) -> Optional[str]:
    if not (isinstance(node, BinOp) and node.op == "*"):
        return None
    percents = [side for side in (node.left, node.right) if isinstance(side, Num) and side.text.endswith("%")]
    others = [side for side in (node.left, node.right) if _decimal(side) is not None]
    if len(percents) != 1 or len(others) != 1:
        return None
    return _show(Decimal(percents[0].text[:-1]))


# ---------------------------------------------------------------- expressions

@_held_back("algebra.change_side_change_sign",
            "Moves a term across the equals sign without changing its sign, e.g. x + 3 = 7 becomes "
            "x = 7 + 3, so x = 10.")
def _change_side_change_sign(node: Node) -> Optional[str]:
    if not (isinstance(node, Rel) and node.op == "=") or _int(node.right) is None:
        return None
    left, r = node.left, _int(node.right)
    if not isinstance(left, BinOp) or left.op not in "+-":
        return None
    x, c = left.left, _int(left.right)
    if isinstance(x, Sym) and c is not None:                       # x + c = r, x - c = r
        value = r + c if left.op == "+" else r - c
    elif isinstance(left.right, Sym) and _int(left.left) is not None:  # c + x = r, c - x = r
        x, c = left.right, _int(left.left)
        value = r + c if left.op == "+" else r - c
    elif (left.op == "+" and isinstance(x, BinOp) and x.op == "*" and _int(x.left)
          and isinstance(x.right, Sym) and c is not None):         # ax + b = r
        a, x = _int(x.left), x.right
        value = sympy.Rational(r + c, a)
    else:
        return None
    return f"{x.name} = {sympy.sstr(sympy.nsimplify(value))}"


def _two_integers(node: Node) -> Optional[tuple[str, int, int]]:
    if isinstance(node, BinOp) and node.op in "+-":
        a, b = _int(node.left), _int(node.right)
        if a is not None and b is not None:
            return node.op, a, b
    return None


@_held_back("negative_numbers.multiplication_rule_for_addition",
            "Uses 'two negatives make a positive' when adding or subtracting negative numbers, "
            "e.g. -3 + (-5) = 8.")
def _multiplication_rule_for_addition(node: Node) -> Optional[str]:
    parts = _two_integers(node)
    if parts is None:
        return None
    op, a, b = parts
    if op == "+" and (a < 0 or b < 0):
        return str(abs(a) + abs(b))
    if op == "-" and a < 0 and b < 0:
        return str(b - abs(a))
    return None


@_held_back("negative_numbers.negative_swaps_operation",
            "Thinks a negative number swaps the operation in front of it instead of being a value, "
            "e.g. 5 + (-3) = 8 and 5 - (-3) = 2.")
def _negative_swaps_operation(node: Node) -> Optional[str]:
    parts = _two_integers(node)
    if parts is None or parts[2] >= 0:
        return None
    op, a, b = parts
    return str(a + abs(b) if op == "+" else a - abs(b))


@_held_back("negative_numbers.two_negatives_always_positive",
            "Applies 'two negatives make a positive' to subtraction, e.g. -5 - 6 = 11.")
def _two_negatives_always_positive(node: Node) -> Optional[str]:
    parts = _two_integers(node)
    if parts is None:
        return None
    op, a, b = parts
    if op == "+":
        return str(abs(a) + abs(b)) if a < 0 or b < 0 else None
    if a >= 0 and b < 0:
        return str(abs(abs(a) - abs(b)))
    if a < 0:
        return str(abs(a) + abs(b))
    return None


assert set(_RULES) == held_back_rules(), "every held-back rule needs code here, and nothing else"
