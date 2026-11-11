"""Writing a hint that passes the critic (Role C), for the orchestrator and the analysis page.

    request = hint_request(problem, answer, diagnosis)
    written = write_hint(hinter, critic, request, problem, llm=session)
    written.hint       # the Hint that passed, or None: send the safe template
    written.attempts   # every rejected attempt, with the critic's reasons
    written.dropped    # the critic said the misconception does not fit, so later attempts went without it

The hinter and the critic are told how sure the diagnosis is, what went wrong, and what the
student already did right (docs/design_decisions.md, T6). A misconception that is only
possible is a lead, not a fact: the hint aims at the part that went wrong. When the critic
says the student's work rules the misconception out, the next attempt is written without it,
instead of being told to aim at it again.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Mapping, Optional

from ..diagnosis import SLIP, Diagnosis
from ..llm import BudgetExceeded, LLMCallError, LLMOutputError
from .analysis import mistake
from .parts import Critic, Hinter
from .state import Hint, HintRequest, Problem

MAX_REWRITES = 2
LLM_ERRORS = (BudgetExceeded, LLMCallError, LLMOutputError)


def certainty(diagnosis: Optional[Diagnosis]) -> Optional[str]:
    """How sure the diagnosis's misconception is: a key of state.CERTAINTY, or None if there is none."""
    if diagnosis is None:
        return None
    e = diagnosis.explanation
    if any(t.get("stage") == "decide" and t.get("eedi") for t in diagnosis.trace):  # the LLM's Eedi pick
        return "likely" if e is not None and e.status == "verified" and e.eedi_fit == "exact" else "possible"
    if diagnosis.status == "clear" and diagnosis.misconception and diagnosis.misconception != SLIP:
        return "possible" if diagnosis.misconception.startswith("eedi:") else "certain"
    if diagnosis.status == "unknown" and e is not None and e.status == "verified":
        return "likely"  # the explainer's own words, its working checked by code
    return None


def right_steps(diagnosis: Optional[Diagnosis]) -> tuple[str, ...]:
    """The steps before the first mistake in the explainer's working, when code verified it."""
    e = diagnosis.explanation if diagnosis is not None else None
    if e is None or e.status != "verified":
        return ()
    wrong = {w.after for w in e.wrong_steps}
    chain, right = [diagnosis.located.before, *e.steps], []
    for before, after in zip(chain, chain[1:]):
        if after in wrong:
            break
        right.append(f"{before} → {after}")
    return tuple(right)


def hint_request(problem: Problem, student_answer: str, diagnosis: Optional[Diagnosis], *, slip: bool = False,
                 right: tuple[str, ...] = ()) -> HintRequest:
    """What the hinter may see for this wrong answer. A slip gets a general hint."""
    located = diagnosis.located if diagnosis is not None else None
    words = mistake(diagnosis, problem) if diagnosis is not None and not slip else None
    return HintRequest(
        topic=problem.topic,
        question=problem.question,
        student_answer=student_answer,
        misconception=words,
        wrong_part=f"{located.source} became {located.wrong}" if located else None,
        certainty=certainty(diagnosis) if words else None,
        right=tuple(dict.fromkeys([*right, *right_steps(diagnosis)])),
    )


@dataclass
class Written:
    hint: Optional[Hint]
    request: HintRequest  # as the last attempt was written
    attempts: list[dict] = field(default_factory=list)  # {"hint", "reasons"} for each rejected one
    dropped: bool = False
    error: Optional[Exception] = None


def write_hint(hinter: Hinter, critic: Critic, request: HintRequest, problem: Problem, *, llm=None,
               unsolved: Mapping[str, str] = {}) -> Written:
    """Up to MAX_REWRITES + 1 attempts, each rewrite with the critic's reasons."""
    written = Written(None, request)
    feedback: list[str] = []
    for _ in range(MAX_REWRITES + 1):
        try:
            hint = hinter.hint(written.request, feedback or None, llm=llm)
            verdict = critic.review(hint.text, "hint", problem, request=written.request, llm=llm, unsolved=unsolved)
        except LLM_ERRORS as e:
            written.error = e
            return written
        if verdict.passed:
            written.hint = hint
            return written
        feedback = list(verdict.reasons)
        written.attempts.append({"hint": hint.text, "reasons": feedback})
        if not verdict.misconception_fits and written.request.misconception:
            written.request = dataclasses.replace(written.request, misconception=None, certainty=None)
            written.dropped = True
    return written
