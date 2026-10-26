"""Simulated students that follow a misconception rule (owner: D). Called by C's runner.

    from matheasy.evaluation import SimStudent

    student = SimStudent("student-001", "fractions.add_numerators_denominators", judge=judge)
    reply = student.respond("Try this one: 1/2 + 1/3", item="1/2 + 1/3")
    reply.answer          # '2/5': the rule decides the answer, never the LLM
    reply.to_log()        # ground truth for the run logs

The student keeps its misconception until a tutor message targets it (decided by
`judge`), then answers correctly from then on. This is what prompted LLM students
fail to do: they flip to the right answer after any feedback at all (Do et al., 2026).

`item` is the maths the tutor asks the student to work, passed by the runner as
plain text ("1/3 + 1/3"). The student never reads the maths out of the tutor's
prose, so a badly worded question cannot change what the student answers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Callable, Optional, Protocol

import sympy

from ..diagnosis import equivalent, held_back_rules, parse, rule_id, run_rule, topic_of
from ..diagnosis.rules import HELD_BACK_FILE, get_rule

GROUPS = ("has_rule", "known_as_text", "never_seen", "unchecked", "slip_only")


def student_group(rule_name: Optional[str]) -> str:
    """Which results group a student's misconception falls in (Team Plan section 7).

    has_rule        the rule is in B's bank, so the tutor can run it
    known_as_text   held back, but an Eedi misconception matches it
    never_seen      held back, and nothing in Eedi matches it
    unchecked       held back, and D has not yet checked it against Eedi
    slip_only       no misconception: only careless slips
    """
    if rule_name is None:
        return "slip_only"
    if rule_name not in held_back_rules():
        return "has_rule"
    # eedi_status: an Eedi MisconceptionId (known as text), "never_seen", or null (not checked yet).
    status = json.loads(Path(HELD_BACK_FILE).read_text())["eedi_status"].get(rule_name)
    if status is None:
        return "unchecked"
    return "never_seen" if status == "never_seen" else "known_as_text"


class FeedbackJudge(Protocol):
    def __call__(self, message: str, misconception: str) -> bool:
        """True if `message` targets the misconception described by `misconception`."""


@dataclass(frozen=True)
class StudentReply:
    text: str                       # what the student says, as the tutor sees it
    item: Optional[str]             # the maths the student was asked to work
    answer: Optional[str]           # the student's answer to `item`
    correct_answer: Optional[str]
    correct: Optional[bool]
    followed_rule: bool             # the answer came from the misconception, not from correct maths
    feedback_targeted: Optional[bool]  # did this tutor message target the misconception? None if not judged
    holds_misconception: bool       # after this turn

    def to_log(self) -> dict:
        return asdict(self)


def correct_answer(item: str) -> str:
    """The exact answer, written the way a student would: 5/6 for fractions, 0.55 for decimals."""
    node = parse(item)
    value = sympy.simplify(node.to_sympy())
    if topic_of(node) == "decimals" and value.is_Rational:
        return _decimal(value)
    return sympy.sstr(value).replace("**", "^")


def _decimal(value: sympy.Rational) -> str:
    exact = Decimal(int(value.p)) / Decimal(int(value.q))
    text = format(exact.normalize(), "f")
    return text if equivalent(text, str(value)) else sympy.sstr(value)  # non-terminating: keep the fraction


def _template(reply_to: Optional[str], answer: Optional[str]) -> str:
    return f"I got {answer}." if answer is not None else "Okay."


class SimStudent:
    """A student who holds one misconception and answers every item by its rule.

    misconception   a MalruleLib rule name, or None for a student with no misconception
    judge           decides whether a tutor message targets the misconception
    predict         the rule as code: item -> the answer it predicts, or None if it does
                    not apply. Defaults to B's run_rule(); held-back rules, which B's
                    runner refuses, pass their own implementation here.
    description     the misconception in words, for the judge. Defaults to the rule's docstring.
    hints_to_fix    how many targeted messages it takes to drop the misconception
    phrase          turns an answer into what the student says. Template for now; an
                    LLM phraser may replace it, but never changes the answer.
    """

    def __init__(
        self,
        student_id: str,
        misconception: Optional[str],
        *,
        judge: Optional[FeedbackJudge] = None,
        predict: Optional[Callable[[str], Optional[str]]] = None,
        description: Optional[str] = None,
        hints_to_fix: int = 1,
        phrase: Callable[[Optional[str], Optional[str]], str] = _template,
    ):
        if misconception is not None and judge is None:
            raise ValueError("a student with a misconception needs a judge")
        self.student_id = student_id
        self.misconception = misconception
        self.judge = judge
        self.predict = predict or (lambda item: run_rule(misconception, item))
        if description is None and misconception is not None:
            description = (get_rule(misconception).apply.__doc__ or misconception).strip()
        self.description = description
        self.hints_to_fix = hints_to_fix
        self.phrase = phrase
        self.targeted_hints = 0  # kept across sessions: the same object plays every session

    @property
    def misconception_id(self) -> Optional[str]:
        return None if self.misconception is None else rule_id(self.misconception)

    @property
    def group(self) -> str:
        return student_group(self.misconception)

    @property
    def holds_misconception(self) -> bool:
        return self.misconception is not None and self.targeted_hints < self.hints_to_fix

    def answer(self, item: str) -> tuple[str, bool]:
        """(answer, followed_rule) for one item."""
        right = correct_answer(item)
        if self.holds_misconception:
            predicted = self.predict(item)
            if predicted is not None and not equivalent(predicted, right):
                return predicted, True
        return right, False

    def respond(self, message: str, item: Optional[str] = None) -> StudentReply:
        """Read one tutor message; answer `item` if the tutor asked one.

        The message is judged before the item is answered, so a hint followed by a
        question is answered with whatever the hint taught.
        """
        targeted = None
        if self.holds_misconception:
            targeted = bool(self.judge(message, self.description))
            if targeted:
                self.targeted_hints += 1
        answer = right = None
        followed = False
        if item is not None:
            answer, followed = self.answer(item)
            right = correct_answer(item)
        return StudentReply(
            text=self.phrase(message, answer),
            item=item,
            answer=answer,
            correct_answer=right,
            correct=None if item is None else not followed,
            followed_rule=followed,
            feedback_targeted=targeted,
            holds_misconception=self.holds_misconception,
        )


_JUDGE_SYSTEM = """\
You check messages from a maths tutor in a research study. A student holds this misconception:

{misconception}

Decide whether the tutor's message targets this misconception: it explains why this way \
of working is wrong, or teaches the correct method for exactly the step the misconception \
gets wrong. These do not count: encouragement, asking the student to check their work, \
a hint about a different mistake, or only stating the answer."""

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"targets": {"type": "boolean"}},
    "required": ["targets"],
}


class LLMFeedbackJudge:
    """Asks the shared LLM client whether a tutor message targets the misconception.

    Pass a session with budget_tokens=None, so judging never spends the tutor's budget:
        LLMFeedbackJudge(llm.session(session_id, condition, budget_tokens=None))
    """

    role = "sim_student_judge"

    def __init__(self, session):
        self.session = session

    def __call__(self, message: str, misconception: str) -> bool:
        result = self.session.call(
            self.role,
            [{"role": "user", "content": f"Tutor's message:\n{message}"}],
            system=_JUDGE_SYSTEM.format(misconception=misconception),
            json_schema=_JUDGE_SCHEMA,
        )
        return bool(result.parsed.get("targets"))
