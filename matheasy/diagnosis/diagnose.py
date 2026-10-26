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

from .interfaces import NoMemory, RuleShortlister, Shortlister, StudentStore
from .maths import LocatedError, Node, locate

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

    def to_log(self) -> dict:
        return {
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
) -> Diagnosis:
    """Diagnose the wrong step before -> after. Raises ParseError or StepIsCorrect."""
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
        {"stage": "shortlist", "candidates": [[c.id, c.score] for c in candidates]},
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
    return Diagnosis(status, top, beliefs, None, located, mode, tuple(trace))
