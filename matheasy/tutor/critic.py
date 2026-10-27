"""The critic (Role C): the hard gate every message passes before the student sees it.

Two layers, cheapest first:

1. Code checks, on every message the tutor writes:
   - leak: any number, fraction, decimal or percentage in the text that is
     equivalent (B's equivalent()) to the problem's answer, or the answer itself
     written out, for hints and probes;
   - an empty or overlong hint.
2. An LLM review, for hints only (questions, probes and praise are fixed
   wording): are its maths statements correct, does it avoid doing the step for
   the student, is it aimed at the diagnosed misconception, is it right for
   students aged 11-14. The critic sees the correct answer; the hinter never does.

Without an LLM session the critic runs the code checks only.
"""

from __future__ import annotations

import re
from typing import Optional

from ..diagnosis import ParseError, equivalent
from ..llm import Session
from .state import HintRequest, MessageKind, Problem, Verdict

MAX_HINT_CHARS = 600

# Numbers, fractions, decimals and percentages, e.g. 5, -3, 5/6, 0.25, 25%.
_MATHS = re.compile(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?%?")

SYSTEM = """\
You review hints written by a maths tutor for students aged 11 to 14, before the student sees them.
You are given the question, its correct answer, the student's wrong answer, the misconception \
the tutor diagnosed (if any), and the hint.

Approve the hint only if all of these hold:
1. Every mathematical statement in it is correct.
2. It does not give the final answer, work out the correct result of the step, or do the step \
for the student. It may explain an idea or a method.
3. It is aimed at the diagnosed misconception. If none is given, it helps the student check \
their own work without guessing a cause.
4. Its wording suits a student aged 11 to 14.

If you reject it, list each problem as one short sentence the writer can act on."""

SCHEMA = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean"},
        "problems": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["approved", "problems"],
}


def leaks_answer(text: str, answer: str) -> Optional[str]:
    """The part of `text` that gives away `answer`, or None."""
    if _normalise(answer) and _normalise(answer) in _normalise(text) and not _MATHS.fullmatch(answer.strip()):
        return answer  # an expression answer, e.g. "2x + 6", written out in full
    for token in _MATHS.findall(text):
        try:
            if equivalent(token.strip(), answer):
                return token.strip()
        except ParseError:
            continue
    return None


def _normalise(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


class TutorCritic:
    role = "critic"

    def review(self, text: str, kind: MessageKind, problem: Problem, *,
               request: Optional[HintRequest] = None, llm: Optional[Session] = None, **_) -> Verdict:
        reasons = self.code_checks(text, kind, problem)
        if reasons or kind != "hint" or llm is None:
            return Verdict(not reasons, tuple(reasons))
        return self.llm_review(text, problem, request, llm)

    def code_checks(self, text: str, kind: MessageKind, problem: Problem) -> list[str]:
        reasons = []
        if kind in ("hint", "probe"):
            leaked = leaks_answer(text, problem.correct_answer)
            if leaked:
                reasons.append(f"It gives away the answer ({leaked}). Do not state or work out the answer.")
        if kind == "hint":
            if not text.strip():
                reasons.append("The hint is empty.")
            elif len(text) > MAX_HINT_CHARS:
                reasons.append(f"The hint is too long ({len(text)} characters); use at most three short sentences.")
        return reasons

    def llm_review(self, text: str, problem: Problem, request: Optional[HintRequest], llm: Session) -> Verdict:
        prompt = "\n".join([
            f"Question: {problem.question}",
            f"Correct answer: {problem.correct_answer}",
            f"Student's answer: {request.student_answer if request else 'unknown'}",
            f"Diagnosed misconception: {(request.misconception if request else None) or 'none'}",
            f"Hint:\n{text}",
        ])
        result = llm.call(self.role, [{"role": "user", "content": prompt}], system=SYSTEM, json_schema=SCHEMA)
        if result.parsed.get("approved") is True:
            return Verdict(True)
        problems = tuple(str(p) for p in result.parsed.get("problems") or ()) or ("The reviewer rejected the hint.",)
        return Verdict(False, problems)
