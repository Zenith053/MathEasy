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

from dataclasses import dataclass
from typing import Optional, Sequence, Union

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


# ---------------------------------------------------------------- T9: counting each rule's right and wrong uses

@dataclass(frozen=True)
class RuleUse:
    """How one rule's mistake fared over a whole answer: chances done right, and done wrong."""
    right: int = 0
    wrong: int = 0

    @property
    def verdict(self) -> Optional[str]:
        """'slip', 'misconception', or None when the rule's mistake was never made.

        right ≥ 1, wrong = 1  slip: the student did this right elsewhere in the same answer
        right = 0, wrong ≥ 1  misconception
        any,       wrong ≥ 2  misconception (repeated)"""
        if self.wrong == 0:
            return None
        return "slip" if self.wrong == 1 and self.right >= 1 else "misconception"


def rule_uses(chain: Sequence[Union[str, Node]], bank=None) -> dict[str, RuleUse]:
    """For every rule that had a chance in this answer, how often it was done right and wrong.

    `chain` is the question then each line of the student's working. In each step, a *chance*
    for a rule is a part the student worked out (not copied unchanged to the next line) on
    which the rule gives a wrong answer. It counts as wrong when the rule reproduces what the
    student wrote for the part locate() found wrong, and as right when the step is correct or
    the part lies outside that wrong part (docs/design_decisions.md, T9):

        1/2 + 1/3 + 1/4 -> 5/6 + 1/4 -> 6/10      add_numerators_denominators: right 1, wrong 1: slip
        x^2 × x^3 + x^4 × x^2 -> x^5 + x^8         multiply_exponents_when_multiplying_powers: slip

    Each chance counts once, at the smallest part the rule fires on (rules also fire on any
    line containing their part). Typed fractions are numbers, not divisions. A line that
    does not parse ends the count. No LLM call."""
    from .bank import load_bank
    from .maths import as_fraction, equivalent, locate
    from .rules import RuleError, run_rule
    from .solve_back import rules_for, solve_back

    rules = (bank or load_bank()).rules()
    try:
        lines = [parse(line) if isinstance(line, str) else line for line in chain]
    except ParseError:
        lines = []
        for line in chain:
            try:
                lines.append(parse(line) if isinstance(line, str) else line)
            except ParseError:
                break
    right: dict[str, int] = {}
    wrong: dict[str, int] = {}

    def chances(parts: list[Node]) -> dict[str, int]:
        """{rule: chances} on these parts. A rule fires on its part and on every larger part holding
        it, so only the smallest parts it fires on count: one chance each."""
        fired: dict[str, list[Node]] = {}
        for part in parts:
            for rule in rules:
                try:
                    predicted = run_rule(rule, part)
                    if predicted is not None and not equivalent(predicted, part):
                        fired.setdefault(rule, []).append(part)
                except (RuleError, ParseError, ValueError, TypeError, ZeroDivisionError):
                    continue
        counts = {}
        for rule, found in fired.items():
            smallest = [p for p in found if not any(q is not p and any(q is t for t in _subtrees(p)) for q in found)]
            counts[rule] = len(smallest)
        return counts

    for before, after in zip(lines, lines[1:]):
        kept = set(_subtrees(after))
        worked = [p for p in _parts(before, as_fraction) if p not in kept]
        try:
            located = locate(before, after)
        except (ParseError, ValueError, TypeError, ZeroDivisionError):
            continue
        if located is None:
            for rule, n in chances(worked).items():
                right[rule] = right.get(rule, 0) + n
            continue
        reproducing = set()
        for rule in rules:
            try:
                predicted = run_rule(rule, located.source_node)
                if predicted is not None and equivalent(predicted, located.wrong_node):
                    reproducing.add(rule)
            except (RuleError, ParseError, ValueError, TypeError, ZeroDivisionError):
                continue
        if not reproducing:
            for piece in solve_back(before, after):
                reproducing.update(rules_for(piece))
        for rule in reproducing:
            wrong[rule] = wrong.get(rule, 0) + 1
        inside = set(_subtrees(located.source_node))
        outside = [p for p in worked if p not in inside and located.source_node not in set(_subtrees(p))]
        for rule, n in chances(outside).items():
            right[rule] = right.get(rule, 0) + n
    return {rule: RuleUse(right.get(rule, 0), wrong.get(rule, 0)) for rule in set(right) | set(wrong)}


def uses_slip(chain: Sequence[Union[str, Node]], misconception: Optional[str]) -> Optional[str]:
    """Why the diagnosed rule's mistake looks like a slip in this answer (T9), or None."""
    if not misconception or not misconception.startswith("rule:"):
        return None
    use = rule_uses(chain).get(misconception)
    if use is None or use.verdict != "slip":
        return None
    times = "once" if use.right == 1 else f"{use.right} times"
    return f"did the same kind of step right {times} elsewhere in this answer, and wrong once"


def _subtrees(node: Node):
    yield node
    for key in ("operand", "left", "right"):
        child = getattr(node, key, None)
        if child is not None:
            yield from _subtrees(child)


def _parts(node: Node, as_fraction) -> list[Node]:
    """The parts of a line a rule can fire on: operations (not typed fractions), -(…), and equations."""
    found = []
    for n in _subtrees(node):
        if isinstance(n, Rel):
            found.append(n)
        elif isinstance(n, BinOp) and not as_fraction(n):
            found.append(n)
        elif isinstance(n, Neg) and isinstance(n.operand, BinOp):
            found.append(n)
    return found
