"""number_fact(): a wrong basic number fact (6 + 3 -> 8), found by code with no LLM call.

Our rules describe wrong methods. A wrong number fact has no method behind it, so no rule
reproduces it and the step used to come back 'unknown': an LLM call with the LLM on, and
with it off not even a topic inside an algebra line. diagnose() now runs this check after
the rules, solve back and the bracket-slip check, before the LLM. A step is a wrong fact when

    one operation on two small whole numbers went wrong   6 + 3, 15 - 7, 7 × 8, 56 ÷ 7
    and nothing else changed                              2x + 6 + 3 -> 2x + 8 is 6 + 3 -> 8
    and the answer is close to the right one              within CLOSE, or a neighbouring
                                                          times-table entry (7 × 8 -> 48 = 6 × 8)

It is diagnosed as a slip with the reason ("6 + 3 is 9, not 8"), topic whole_number_arithmetic.
A fact that goes wrong again in a later answer is raised from a slip to a weak number fact,
fact:<a><op><b> (ids.py), by the planner: diagnose() keeps no memory between calls.

    number_fact("6 + 3", "8")             Fact(6 + 3 is 9, not 8)
    number_fact("2x + 6 + 3", "2x + 8")   Fact(6 + 3 is 9, not 8)
    number_fact("7 × 8 + 6", "60")        Fact(7 × 8 is 56, not 54), via solve back
    number_fact("47 + 38", "84")          None: not a basic fact (two-digit numbers)
    number_fact("6 + 3", "20")            None: not close to 9, so likely a method, not a fact
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Union

import sympy

from .ids import fact_id
from .maths import BinOp, LocatedError, Node, ParseError, Rel, locate, parse
from .rules._common import whole
from .slips import _signed_terms
from .solve_back import solve_back

ADD_MAX = 20  # + and -: both numbers at most 20 (number bonds and facts within 20)
TIMES_MAX = 12  # × and ÷: the times tables up to 12
CLOSE = 2  # the wrong answer is at most this far from the right one
SHOW = {"+": "+", "-": "-", "*": "×", "/": "÷"}


@dataclass(frozen=True)
class Fact:
    """One basic fact the student got wrong: a op b should be `right`, they wrote `wrong`."""
    a: int
    op: str
    b: int
    right: int
    wrong: int

    @property
    def id(self) -> str:
        return fact_id(self.a, self.op, self.b)

    @property
    def shown(self) -> str:
        return f"{self.a} {SHOW[self.op]} {self.b}"

    @property
    def reason(self) -> str:
        """For the logs and the student's memory: '6 + 3 is 9, not 8'."""
        return f"{self.shown} is {self.right}, not {self.wrong}"

    @property
    def said(self) -> str:
        """For the student, without the right result: 'wrote 8 for 6 + 3'."""
        return f"wrote {self.wrong} for {self.shown}"


def number_fact(before: Union[str, Node], after: Union[str, Node]) -> Optional[tuple[Fact, LocatedError]]:
    """The one wrong basic fact that explains before -> after, with where it is; else None."""
    try:
        before = parse(before) if isinstance(before, str) else before
        after = parse(after) if isinstance(after, str) else after
    except ParseError:
        return None
    if isinstance(before, Rel) or isinstance(after, Rel):
        if not (isinstance(before, Rel) and isinstance(after, Rel) and before.op == after.op):
            return None
        if _same(before.right, after.right):
            return number_fact(before.left, after.left)
        if _same(before.left, after.left):
            return number_fact(before.right, after.right)
        return None
    found = _from_numbers(before, after) if not _has_letters(before) else _from_terms(before, after)
    facts = {fact.id: (fact, where) for fact, where in found}
    return next(iter(facts.values())) if len(facts) == 1 else None  # two facts could explain it: say nothing


def basic(a: int, op: str, b: int) -> bool:
    """A fact a student is expected to know by heart."""
    if op == "+":
        return 0 <= a <= ADD_MAX and 0 <= b <= ADD_MAX
    if op == "-":
        return 0 <= b <= a <= ADD_MAX
    if op == "*":
        return 0 <= a <= TIMES_MAX and 0 <= b <= TIMES_MAX
    if op == "/":
        return 1 <= b <= TIMES_MAX and a % b == 0 and 0 <= a // b <= TIMES_MAX
    return False


def close(a: int, op: str, b: int, wrong: int) -> bool:
    """A near miss, not a different method: within CLOSE, or a neighbouring times-table entry."""
    right = _value(a, op, b)
    if wrong == right or wrong < 0:
        return False
    if abs(wrong - right) <= CLOSE:
        return True
    return op == "*" and wrong in {(a - 1) * b, (a + 1) * b, a * (b - 1), a * (b + 1)}


def _fact(a: int, op: str, b: int, wrong) -> Optional[Fact]:
    if a is None or b is None or wrong is None or Fraction(wrong).denominator != 1:
        return None
    wrong = int(wrong)
    if not basic(a, op, b) or not close(a, op, b, wrong):
        return None
    return Fact(a, op, b, _value(a, op, b), wrong)


def _from_numbers(before: Node, after: Node) -> list[tuple[Fact, LocatedError]]:
    """A line of numbers: solve back says which single operation went wrong and what it gave.
    A line that is one operation is checked as it stands: solve back reads 56 ÷ 7 as a number."""
    found = []
    if isinstance(before, BinOp) and before.op in SHOW and not before.typed_fraction:
        fact = _fact(whole(before.left), before.op, whole(before.right), _whole_value(after))
        where = locate(before, after) if fact else None
        if fact and where is not None:
            return [(fact, where)]
    for piece in solve_back(before, after):
        node = piece.source_node
        if not (isinstance(node, BinOp) and node.op in SHOW):
            continue
        fact = _fact(whole(node.left), node.op, whole(node.right), _whole_value(piece.wrong_node))
        if fact:
            found.append((fact, piece))
    return found


def _from_terms(before: Node, after: Node) -> list[tuple[Fact, LocatedError]]:
    """A line with letters: only a number changed, and one fact among its terms accounts for it.
    Either two whole-number terms were combined (2x + 6 + 3 -> 2x + 8), or a term that is
    itself a × or ÷ of two whole numbers was worked out (2x + 7 × 8 -> 2x + 54)."""
    try:
        change = sympy.expand(after.to_sympy() - before.to_sympy())
    except Exception:
        return []
    if change == 0 or not change.is_Rational:
        return []
    c = Fraction(int(change.p), int(change.q))
    terms = _signed_terms(before, 1)
    candidates = []
    numbers = [(sign, whole(node)) for sign, node in terms if whole(node) is not None]
    if len(numbers) == 2:
        candidates.append(_combined(*numbers[0], *numbers[1], c))
    for sign, node in terms:
        if isinstance(node, BinOp) and node.op in "*/" and not node.typed_fraction:
            a, b = whole(node.left), whole(node.right)
            if a is not None and b is not None and (node.op == "*" or (b and a % b == 0)):
                candidates.append((a, node.op, b, _value(a, node.op, b) + sign * c))
    found = []
    for a, op, b, wrong in candidates:
        fact = _fact(a, op, b, wrong)
        if fact:
            found.append((fact, locate(parse(f"{a} {SHOW[op]} {b}"), parse(str(fact.wrong)))))
    return found


def _combined(s1: int, a: int, s2: int, b: int, c: Fraction) -> tuple[int, str, int, Fraction]:
    """Two whole-number terms with signs s1, s2, combined into one number that came out c too big.
    +6 +3 -> 6 + 3; -6 -3 -> -(6 + 3); +7 -3 -> 7 - 3; -3 +7 -> 7 - 3."""
    if s1 == s2:
        return a, "+", b, a + b + s1 * c
    p, m = (a, b) if s1 > 0 else (b, a)  # the positive number, and the size of the negative one
    return (p, "-", m, p - m + c) if p >= m else (m, "-", p, m - p - c)


def _value(a: int, op: str, b: int) -> int:
    return {"+": a + b, "-": a - b, "*": a * b, "/": a // b if b else -1}[op]


def _whole_value(node: Node) -> Optional[int]:
    n = whole(node)
    if n is not None:
        return n
    try:
        v = sympy.nsimplify(node.to_sympy())
        return int(v) if v.is_Integer else None
    except Exception:
        return None


def _has_letters(node: Node) -> bool:
    try:
        return bool(node.to_sympy().free_symbols)
    except Exception:
        return True


def _same(a: Node, b: Node) -> bool:
    try:
        return sympy.simplify(a.to_sympy() - b.to_sympy()) == 0
    except Exception:
        return False


def fact_of(diagnosis) -> Optional[str]:
    """The number fact behind a diagnosis (its fact:… ID), or None."""
    return next((t["fact"] for t in diagnosis.trace if t.get("stage") == "slip" and t.get("fact")), None)
