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

Any rule works by name: B's rules run through B's run_rule(), and the held-back
rules, which B's runner refuses, through D's own code (held_back.py). roster()
makes one student per rule, so each rule B releases gets a student with no new code.

Slips: `slip_rate` is the chance of a careless slip on an item the student would
otherwise get right. Students with no misconception slip at SLIP_ONLY_RATE and give
the over-claiming rate. A slip is never an answer some rule would give.

Several sessions: C's run_session() calls begin_session(index) before each session.
save() and load() keep a student between processes. `relapse_rate` is the chance
that a fixed misconception comes back at the start of a later session (0 by default).
Every random draw is seeded by (seed, student_id, what it is for), so a restored
student carries on exactly as if it had never stopped.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Callable, Optional, Protocol, Sequence, Union

import sympy

from ..diagnosis import (
    ParseError, RuleError, equivalent, held_back_rules, parse, rule_id, rule_names, run_rule, topic_of,
)
from ..diagnosis.maths import Rel
from ..diagnosis.rules import HELD_BACK_FILE, get_rule
from .held_back import held_back_names, held_back_rule

GROUPS = ("has_rule", "known_as_text", "never_seen", "unchecked", "slip_only")
SLIP_ONLY_RATE = 0.3  # share of items a slip-only student gets wrong


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
    # eedi_status: Eedi MisconceptionIds (known as text; a list, or one number), "never_seen",
    # or null (not checked yet).
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
    slipped: bool = False           # a careless slip: wrong, but not by the rule
    deliberate: bool = False        # wrong on purpose (adversarial intentional_wrong_answer), not a slip
    attack: Optional[str] = None    # the adversarial technique used in this reply (adversarial.py)

    def to_log(self) -> dict:
        return asdict(self)


def correct_answer(item: str) -> str:
    """The exact answer, written the way a student would: 5/6 for fractions, 0.55 for decimals,
    x = 4 for an equation in one letter, and for a comparison the true relation in the item's
    order, whatever sign the item was written with ("1/3 < 1/4" -> "1/3 > 1/4")."""
    node = parse(item)
    if isinstance(node, Rel):
        return _relation_answer(node)
    value = sympy.simplify(node.to_sympy())
    if topic_of(node) == "decimals" and value.is_Rational:
        return _decimal(value)
    return sympy.sstr(value).replace("**", "^")


def _relation_answer(node: Rel) -> str:
    left, right = node.left.to_sympy(), node.right.to_sympy()
    letters = sorted((left - right).free_symbols, key=str)
    if node.op == "=" and len(letters) == 1:
        solutions = sympy.solve(left - right, letters[0])
        if len(solutions) == 1:
            return f"{letters[0]} = {sympy.sstr(solutions[0])}"
    if letters:
        raise ValueError(f"no single answer for {node}")
    op = "=" if left == right else ">" if left > right else "<"
    return f"{node.left} {op} {node.right}"


def _decimal(value: sympy.Rational) -> str:
    exact = Decimal(int(value.p)) / Decimal(int(value.q))
    text = format(exact.normalize(), "f")
    return text if equivalent(text, str(value)) else sympy.sstr(value)  # non-terminating: keep the fraction


def _template(reply_to: Optional[str], answer: Optional[str]) -> str:
    return f"I got {answer}." if answer is not None else "Okay."


_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
_FLIP = {"<": ">", ">": "<"}


def rule_answers(item: str) -> list[str]:
    """Every answer some misconception rule gives for `item`: B's rules and the held-back ones."""
    answers = []
    for name in rule_names():
        try:
            answers.append(run_rule(name, item))
        except (ParseError, ValueError, ZeroDivisionError, RuleError):  # RuleError: a buggy rule, skipped
            pass
    answers += [held_back_rule(name).predict(item) for name in held_back_names()]
    return [a for a in answers if a is not None]


def slips(item: str, right: Optional[str] = None) -> list[str]:
    """Careless slips on `item`: the right answer with one number off by one in its last
    place (5/6 -> 5/7, 0.55 -> 0.54), or a comparison the wrong way round. Slips that
    match the right answer or any rule's answer are left out, so a slip never looks
    like a misconception."""
    right = right or correct_answer(item)
    node = parse(right)
    if isinstance(node, Rel) and node.op in _FLIP:
        candidates = [f"{node.left} {_FLIP[node.op]} {node.right}"]
    else:
        candidates = []
        for m in _NUMBER.finditer(right):
            step = Decimal(1).scaleb(-len(m.group().partition(".")[2]))
            for new in (Decimal(m.group()) + step, Decimal(m.group()) - step):
                if new >= 0:
                    candidates.append(right[:m.start()] + format(new, "f") + right[m.end():])
    explained = rule_answers(item)
    kept = []
    for c in candidates:
        try:
            parsed = parse(c)
            if not isinstance(parsed, Rel) and parsed.to_sympy().is_finite is False:
                continue  # 5/0
            if equivalent(c, right) or any(equivalent(c, e) for e in explained):
                continue
        except (ParseError, ValueError, TypeError, ZeroDivisionError):
            continue
        kept.append(c)
    return kept


class SimStudent:
    """A student who holds one misconception and answers every item by its rule.

    misconception   a MalruleLib rule name, or None for a student with no misconception
    judge           decides whether a tutor message targets the misconception
    predict         the rule as code: item -> the answer it predicts, or None if it does
                    not apply. Defaults to B's run_rule(), or D's code for a held-back rule.
    description     the misconception in words, for the judge. Defaults to the rule's docstring.
    hints_to_fix    how many targeted messages it takes to drop the misconception
    slip_rate       chance of a careless slip on an item it would otherwise get right
    relapse_rate    chance a fixed misconception comes back at the start of a later session
    seed            with student_id, seeds every random draw
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
        slip_rate: float = 0.0,
        relapse_rate: float = 0.0,
        seed: int = 0,
        phrase: Callable[[Optional[str], Optional[str]], str] = _template,
    ):
        if misconception is not None and judge is None:
            raise ValueError("a student with a misconception needs a judge")
        if not (0 <= slip_rate <= 1 and 0 <= relapse_rate <= 1):
            raise ValueError("slip_rate and relapse_rate are probabilities")
        self.student_id = student_id
        self.misconception = misconception
        self.judge = judge
        self._custom = predict is not None or description is not None  # can't be saved
        held = held_back_rule(misconception) if misconception in held_back_rules() else None
        if predict is None:
            predict = held.predict if held else (lambda item: run_rule(misconception, item))
        self.predict = predict
        if description is None and misconception is not None:
            description = held.description if held else (get_rule(misconception).apply.__doc__ or misconception).strip()
        self.description = description
        self.hints_to_fix = hints_to_fix
        self.slip_rate = slip_rate
        self.relapse_rate = relapse_rate
        self.seed = seed
        self.phrase = phrase
        self.targeted_hints = 0  # kept across sessions: the same object, or a saved one, plays every session
        self.items_answered = 0  # numbers the slip draws
        self.session_index: Optional[int] = None
        self.relapsed = False  # at the start of the current session

    def _random(self, purpose: str, n: int) -> random.Random:
        return random.Random(f"{self.seed}:{self.student_id}:{purpose}:{n}")

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
        """(answer, followed_rule) for one item. A wrong answer that did not follow the rule is a slip."""
        right = correct_answer(item)
        rng = self._random("slip", self.items_answered)
        self.items_answered += 1
        if self.holds_misconception:
            predicted = self.predict(item)
            if predicted is not None and not equivalent(predicted, right):
                return predicted, True
        if self.slip_rate and rng.random() < self.slip_rate:
            options = slips(item, right)
            if options:
                return rng.choice(options), False
        return right, False

    def begin_session(self, session_index: int) -> bool:
        """Called by C's run_session() before each session. From the second session on, a
        fixed misconception comes back with probability relapse_rate. Returns True if it
        came back; calling it again for the same session gives the same answer."""
        if session_index == self.session_index:
            return self.relapsed
        self.relapsed = (session_index > 0 and self.misconception is not None and not self.holds_misconception
                         and self._random("relapse", session_index).random() < self.relapse_rate)
        if self.relapsed:
            self.targeted_hints = 0
        self.session_index = session_index
        return self.relapsed

    def to_state(self) -> dict:
        """Everything needed to carry on later. The judge and phraser are passed again on load."""
        if self._custom:
            raise ValueError("a student with its own predict or description can't be saved")
        return {
            "student_id": self.student_id, "misconception": self.misconception, "hints_to_fix": self.hints_to_fix,
            "slip_rate": self.slip_rate, "relapse_rate": self.relapse_rate, "seed": self.seed,
            "targeted_hints": self.targeted_hints, "items_answered": self.items_answered,
            "session_index": self.session_index, "relapsed": self.relapsed,
        }

    @classmethod
    def from_state(cls, state: dict, *, judge: Optional[FeedbackJudge] = None,
                   phrase: Callable[[Optional[str], Optional[str]], str] = _template) -> "SimStudent":
        student = cls(state["student_id"], state["misconception"], judge=judge, hints_to_fix=state["hints_to_fix"],
                      slip_rate=state["slip_rate"], relapse_rate=state["relapse_rate"], seed=state["seed"],
                      phrase=phrase)
        student.targeted_hints = state["targeted_hints"]
        student.items_answered = state["items_answered"]
        student.session_index = state["session_index"]
        student.relapsed = state.get("relapsed", False)
        return student

    def save(self, directory: Union[str, Path]) -> Path:
        """Write the student's state to <directory>/<student_id>.json, for its next session."""
        path = _state_path(directory, self.student_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_state(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, directory: Union[str, Path], student_id: str, *, judge: Optional[FeedbackJudge] = None,
             phrase: Callable[[Optional[str], Optional[str]], str] = _template) -> "SimStudent":
        state = json.loads(_state_path(directory, student_id).read_text())
        return cls.from_state(state, judge=judge, phrase=phrase)

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
        answer = right = correct = None
        followed = False
        if item is not None:
            answer, followed = self.answer(item)
            right = correct_answer(item)
            correct = equivalent(answer, right)
        return StudentReply(
            text=self.phrase(message, answer),
            item=item,
            answer=answer,
            correct_answer=right,
            correct=correct,
            followed_rule=followed,
            feedback_targeted=targeted,
            holds_misconception=self.holds_misconception,
            slipped=correct is False and not followed,
        )


def _state_path(directory: Union[str, Path], student_id: str) -> Path:
    if not re.fullmatch(r"[\w.#-]+", student_id):
        raise ValueError(f"student_id {student_id!r} can't be a file name")
    return Path(directory) / f"{student_id}.json"


def roster(
    judge: Optional[FeedbackJudge],
    *,
    copies: int = 1,
    slip_only: int = 0,
    slip_rate: float = SLIP_ONLY_RATE,
    seed: int = 0,
    rules: Optional[Sequence[str]] = None,
    relapse_rate: float = 0.0,
) -> list[SimStudent]:
    """`copies` students for each rule (by default every rule B has registered, and every
    held-back rule), then `slip_only` students with no misconception. IDs are stable
    (fractions.subtract_across#0, slip_only#0), so shards can be split by student.
    relapse_rate > 0 lets fixed misconceptions come back in later sessions, so retention
    measures something; with 0, retention_truth is 1 by construction."""
    names = sorted(set(rule_names()) | held_back_rules()) if rules is None else list(rules)
    students = [SimStudent(f"{name}#{k}", name, judge=judge, seed=seed, relapse_rate=relapse_rate)
                for name in names for k in range(copies)]
    students += [SimStudent(f"slip_only#{k}", None, slip_rate=slip_rate, seed=seed) for k in range(slip_only)]
    return students


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
