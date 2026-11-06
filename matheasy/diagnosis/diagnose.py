"""diagnose(): which misconception caused a wrong step. Called by C's orchestrator.

    from matheasy.diagnosis import check_step, diagnose

    if not check_step("1/2 + 1/3", "2/5"):
        d = diagnose("1/2 + 1/3", "2/5", student_id="student-017")
        d.status         # 'clear' | 'ambiguous' | 'unknown'
        d.misconception  # 'rule:fractions.add_numerators_denominators' when clear, else None
        d.probe          # a follow-up question to ask, or None
        d.to_log()       # JSON-safe record for the run logs

The diagnoser keeps no state between calls. When it returns a probe, the
orchestrator asks it, then calls diagnose() again with the same step and the
answers so far in `follow_ups`. At most MAX_FOLLOW_UPS probes are ever returned.

Week-1 placeholder: returns the shortlister's top choice and never probes. Top
choices that tie are 'ambiguous'. The hypothesis tester replaces this in week 3,
with the same signature and return type. mode="b2" (baseline B2) will stay as it
is now: the shortlister's top choice, no hypothesis testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, Union

from .explain import Explanation, explain, rule_topic
from .interfaces import NoMemory, RuleShortlister, Shortlister, StudentStore
from .maths import LocatedError, Node, locate
from .solve_back import rules_for, solve_back

MAX_FOLLOW_UPS = 2

Status = Literal["clear", "ambiguous", "unknown"]
Mode = Literal["full", "b2"]


@dataclass(frozen=True)
class Probe:
    """A follow-up question chosen to tell the remaining suspects apart."""
    question: str                 # e.g. "1/3 + 1/3"
    predictions: dict[str, str]   # misconception ID -> the answer it predicts


@dataclass(frozen=True)
class FollowUp:
    probe: Probe
    answer: str  # what the student wrote


@dataclass(frozen=True)
class Diagnosis:
    status: Status
    misconception: Optional[str]   # set only when status is 'clear'
    beliefs: dict[str, float]      # misconception ID -> probability, most likely first
    probe: Optional[Probe]         # ask this next, or None
    located: LocatedError
    mode: Mode
    trace: tuple[dict, ...] = field(default=())  # every reasoning step, for the logs
    explanation: Optional[Explanation] = None    # the LLM's reading of an 'unknown' step, if a session was given
    topics: tuple[str, ...] = ()                 # every topic involved, whether rules or the LLM found the mistake

    def to_log(self) -> dict:
        return {
            "topics": list(self.topics),
            "explanation": None if self.explanation is None else self.explanation.to_log(),
            "status": self.status,
            "misconception": self.misconception,
            "beliefs": self.beliefs,
            "probe": None if self.probe is None else {"question": self.probe.question,
                                                      "predictions": self.probe.predictions},
            "located": self.located.to_dict(),
            "mode": self.mode,
            "trace": list(self.trace),
        }


class StepIsCorrect(ValueError):
    """diagnose() was called on a correct step. Call check_step() first."""


def diagnose(
    before: Union[str, Node],
    after: Union[str, Node],
    *,
    student_id: Optional[str] = None,
    follow_ups: Sequence[FollowUp] = (),
    shortlister: Optional[Shortlister] = None,
    store: Optional[StudentStore] = None,
    mode: Mode = "full",
    k: int = 25,
    question: str = "",
    session=None,
) -> Diagnosis:
    """Diagnose the wrong step before -> after. Raises ParseError or StepIsCorrect.

    With an LLM `session`, a step no rule reproduces is passed to explain(), so the
    result carries the model's reading of it; without one, no model is called.
    """
    if mode not in ("full", "b2"):
        raise ValueError(f"unknown mode {mode!r}")
    if len(follow_ups) > MAX_FOLLOW_UPS:
        raise ValueError(f"at most {MAX_FOLLOW_UPS} follow-ups")
    located = locate(before, after)
    if located is None:
        raise StepIsCorrect(f"{before!r} -> {after!r} is correct")
    shortlister = shortlister or RuleShortlister()
    store = store or NoMemory()  # read by the hypothesis tester in week 3

    candidates = sorted(shortlister(located, k), key=lambda c: -c.score)
    trace = [
        {"stage": "locate", **located.to_dict()},
        {"stage": "shortlist", "candidates": [[c.id, c.score] for c in candidates],
         "failed_rules": list(getattr(shortlister, "failed", []))},  # rules that crashed: bugs to fix
    ]
    total = sum(max(c.score, 0.0) for c in candidates)
    beliefs = {c.id: (max(c.score, 0.0) / total if total else 1 / len(candidates)) for c in candidates}

    if not candidates:
        status, top = "unknown", None
    elif len(candidates) > 1 and candidates[1].score == candidates[0].score:
        status, top = "ambiguous", None
    else:
        status, top = "clear", candidates[0].id
    trace.append({"stage": "decide", "status": status, "misconception": top})

    pieces = []
    if status == "unknown":  # a whole line turned into one number: find which operation went wrong
        pieces = [(p, rules_for(p)) for p in solve_back(before, after)]
        trace.append({"stage": "solve_back", "pieces": [[p.source, p.wrong, rules] for p, rules in pieces]})
        hits = {rule: p for p, rules in pieces for rule in rules}
        if hits:
            beliefs = {rule: 1 / len(hits) for rule in hits}
            status, top = ("clear", next(iter(hits))) if len(hits) == 1 else ("ambiguous", None)
            if len({id(p) for p in hits.values()}) == 1:
                located = next(iter(hits.values()))  # the wrong piece, not the whole line
            trace.append({"stage": "decide", "status": status, "misconception": top})

    explanation = None
    if status == "unknown" and session is not None:
        explanation = explain(located, question=question, session=session, pieces=[p for p, _ in pieces])
        trace.append({"stage": "explain", **explanation.to_log()})
    topics = (explanation.topics if explanation is not None
              else tuple(sorted({rule_topic(m) for m in beliefs if m.startswith("rule:")})))
    return Diagnosis(status, top, beliefs, None, located, mode, tuple(trace), explanation, topics)
