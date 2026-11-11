"""The critic (Role C): the hard gate every message passes before the student sees it.

Two layers, cheapest first:

1. Code checks, on every message the tutor writes:
   - leak: any number, fraction, decimal or percentage in the text that is
     equivalent (B's equivalent()) to the problem's answer, or the answer itself
     written out, for hints and probes (repeating the question is not a leak).
     The same for every other item still open in the session (`unsolved`), as
     D's leak metric counts those too;
   - an empty or overlong hint.
2. An LLM review, for hints only (questions, probes and praise are fixed
   wording): are its maths statements correct, does it avoid doing the step for
   the student, is it aimed at the diagnosed misconception, is it right for
   students aged 11-14. The critic sees the correct answer; the hinter never does.

Without an LLM session the critic runs the code checks only.
"""

from __future__ import annotations

import re
from typing import Mapping, Optional

from ..diagnosis import ParseError, equivalent
from ..llm import Session
from .mathtext import maths_runs
from .state import CERTAINTY, HintRequest, MessageKind, Problem, Verdict

MAX_HINT_CHARS = 600

# Numbers, fractions, decimals and percentages, e.g. 5, -3, 5/6, 0.25, 25%.
_MATHS = re.compile(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?%?")

SYSTEM = """\
You review hints written by a maths tutor for students aged 11 to 14, before the student sees them.
You are given the question, its correct answer, the student's wrong answer, which part of their \
work went wrong, what they already did right (when known), the misconception the tutor diagnosed \
(if any) and how sure that diagnosis is, and the hint.

Approve the hint only if all of these hold:
1. Every mathematical statement in it is correct.
2. It does not give the final answer, and it does not work out the corrected result of the part \
that went wrong or do that step for the student. It may restate what the student already worked \
out correctly, explain an idea or a method, say what a notation means (for example that 2^2 \
means 2 × 2) without giving its value, and ask the student to work a part out themselves: a \
question is not an answer.
3. If the diagnosis is certain or likely, the hint is aimed at that misconception. If it is only \
possible, or none is given, the hint helps the student check the part that went wrong, without \
insisting on a cause.
4. Its wording suits a student aged 11 to 14.

Also say whether the diagnosed misconception fits: misconception_fits is false only if the \
student's work shows that it does not explain this mistake, and true otherwise (true when none is given).

If you reject it, list each problem as one short sentence the writer can act on."""

SCHEMA = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean"},
        "problems": {"type": "array", "items": {"type": "string"}},
        "misconception_fits": {"type": "boolean"},
    },
    "required": ["approved", "problems", "misconception_fits"],
}


# An equation's answer, e.g. "x = 4": the value alone ("4") gives it away too.
_SOLUTION = re.compile(r"\s*[A-Za-z]\s*=\s*(.+)")
_PARTS = re.compile(r"\s+(halves|half|thirds?|quarters?|fifths?|sixths?|sevenths?|eighths?|ninths?|tenths?"
                    r"|elevenths?|twelfths?|hundredths?|thousandths?)\b", re.I)


def leaks_answer(text: str, answer: str, question: Optional[str] = None) -> Optional[str]:
    """The part of `text` that gives away `answer`, or None.

    Repeating the question as written, or a number in it, is not a leak, even when the
    answer is equal to it: "Look at (x + 3)^2 again" doesn't give away (x + 3)^2, and
    "simplify 12/34" doesn't give away 6/17. A number counting fraction parts, "1 sixth" or
    "3 quarters", is not a stated answer either; "the total is 1" still is. When the answer is the question itself, unworked (an expansion the answer key
    leaves as written), only the worked-out, expanded form gives it away: (3y - 4)(3y - 4)
    restates (3y - 4)^2. This follows D's leak metric."""
    asked = _normalise(question) if question else None
    given = {_normalise(n) for n in _MATHS.findall(question)} if question else set()
    unworked = bool(question) and _equal(answer, question)
    solution = _SOLUTION.fullmatch(answer)
    if solution or re.search(r"[A-Za-z<>]", answer):
        # An expression, equation or comparison: a piece of maths in the text equal to it.
        for run in maths_runs(text):
            sides = [run] if solution or "=" not in run else [run] + [s.strip() for s in run.split("=")]
            for piece in sides:
                if _normalise(piece) == asked or (solution and not _SOLUTION.fullmatch(piece)):
                    continue  # an equation counts only in solved form: x + 3 = 7 equals x = 4 too
                if _equal(piece, answer) and (not unworked or _worked_out(piece)):
                    return piece
    value = solution.group(1).strip() if solution else answer
    if re.search(r"[A-Za-z<>=]", value):
        return None
    for match in _MATHS.finditer(text):
        token = match.group().strip()
        if _PARTS.match(text, match.end()):
            continue  # "1 sixth", "3 quarters": counting parts, not stating a result
        if _normalise(token) not in given and _equal(token, value):
            return token
    return None


def _worked_out(piece: str) -> bool:
    """True if the maths is already expanded: 9y^2 - 24y + 16, not (3y - 4)(3y - 4)."""
    try:
        from ..diagnosis import parse
        import sympy
        value = parse(piece).to_sympy()
        return sympy.expand(value) == value
    except Exception:
        return True  # unreadable: treat as a possible leak, the safe side


def _equal(a: str, b: str) -> bool:
    try:
        return equivalent(a, b)
    except (ParseError, ValueError, TypeError):
        return False


def _normalise(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


class TutorCritic:
    role = "critic"

    def review(self, text: str, kind: MessageKind, problem: Problem, *,
               request: Optional[HintRequest] = None, llm: Optional[Session] = None,
               unsolved: Mapping[str, str] = {}, **_) -> Verdict:
        reasons = self.code_checks(text, kind, problem, unsolved)
        if reasons or kind != "hint" or llm is None:
            return Verdict(not reasons, tuple(reasons))
        return self.llm_review(text, problem, request, llm)

    def code_checks(self, text: str, kind: MessageKind, problem: Problem,
                    unsolved: Mapping[str, str] = {}) -> list[str]:
        reasons = []
        if kind in ("hint", "probe"):
            leaked = leaks_answer(text, problem.correct_answer, problem.question)
            if leaked:
                reasons.append(f"It gives away the answer ({leaked}). Do not state or work out the answer.")
            for question, answer in unsolved.items():
                other = question != problem.question and leaks_answer(text, answer, question)
                if other:
                    reasons.append(f"It gives away the answer to {question} ({other}). Do not use that number.")
                    break
        if kind == "hint":
            if not text.strip():
                reasons.append("The hint is empty.")
            elif len(text) > MAX_HINT_CHARS:
                reasons.append(f"The hint is too long ({len(text)} characters); use at most three short sentences.")
        return reasons

    def llm_review(self, text: str, problem: Problem, request: Optional[HintRequest], llm: Session) -> Verdict:
        lines = [
            f"Question: {problem.question}",
            f"Correct answer: {problem.correct_answer}",
            f"Student's answer: {request.student_answer if request else 'unknown'}",
            f"What went wrong: {(request.wrong_part if request else None) or 'not located'}",
        ]
        if request and request.right:
            lines.append(f"Already right: {'; '.join(request.right)}")
        lines.append(f"Diagnosed misconception: {(request.misconception if request else None) or 'none'}")
        if request and request.misconception and request.certainty:
            lines.append(f"How sure: {CERTAINTY[request.certainty]}")
        lines.append(f"Hint:\n{text}")
        result = llm.call(self.role, [{"role": "user", "content": "\n".join(lines)}], system=SYSTEM,
                          json_schema=SCHEMA)
        fits = result.parsed.get("misconception_fits") is not False
        if result.parsed.get("approved") is True:
            return Verdict(True, misconception_fits=fits)
        problems = tuple(str(p) for p in result.parsed.get("problems") or ()) or ("The reviewer rejected the hint.",)
        return Verdict(False, problems, misconception_fits=fits)
