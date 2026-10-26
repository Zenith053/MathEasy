"""The shared tutor state (Role C).

The orchestrator reads and writes a TutorState; the diagnosis types come from
B (matheasy.diagnosis). The hinter never sees a correct answer: it gets a
HintRequest, which has no field for one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from ..diagnosis import Diagnosis, FollowUp, Probe

MessageKind = Literal["question", "probe", "hint", "feedback", "safe_template"]


@dataclass(frozen=True)
class Problem:
    id: str
    topic: str  # "fractions", "decimals" or "expressions"
    question: str  # the maths the student works, e.g. "1/2 + 1/3"
    correct_answer: str  # for the critic's leak check; never given to the hinter


@dataclass(frozen=True)
class HintRequest:
    """Everything the hinter may see. No correct answer, by construction."""

    topic: str
    question: str
    student_answer: str
    misconception: Optional[str]  # the misconception in words; None means give a general hint
    wrong_part: Optional[str]  # e.g. "1/2 + 1/3 became 2/5"; never what it should have been


@dataclass(frozen=True)
class Hint:
    text: str
    citations: tuple[str, ...] = ()  # textbook section references


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass
class StudentMemory:
    """What the tutor remembers about one student across sessions."""

    student_id: str
    times_diagnosed: dict[str, int] = field(default_factory=dict)  # misconception ID -> count


@dataclass(frozen=True)
class Utterance:
    speaker: Literal["student", "tutor"]
    text: str
    kind: Optional[MessageKind] = None  # tutor messages only


@dataclass
class TutorState:
    """One session: one student working through one or more problems."""

    session_id: str
    condition: str
    student_id: str
    memory: StudentMemory
    transcript: list[Utterance] = field(default_factory=list)
    turn: int = 0  # number of tutor messages sent this session
    budget_exhausted: bool = False

    # The problem being worked now. Reset by Orchestrator.start().
    problem: Optional[Problem] = None
    probe: Optional[Probe] = None  # the follow-up question awaiting an answer
    wrong_answer: Optional[str] = None  # the wrong answer being diagnosed
    follow_ups: list[FollowUp] = field(default_factory=list)
    followups_asked: int = 0
    hints_given: int = 0
    diagnosis: Optional[Diagnosis] = None
    solved: bool = False

    @property
    def item(self) -> Optional[str]:
        """The maths the student is being asked to work right now."""
        if self.probe is not None:
            return self.probe.question
        return self.problem.question if self.problem else None


@dataclass(frozen=True)
class TutorReply:
    kind: MessageKind
    text: str
    item: Optional[str]  # the maths the student should answer, passed to D's student as `item`
    critic_passed: bool
    done: bool  # the problem is solved
