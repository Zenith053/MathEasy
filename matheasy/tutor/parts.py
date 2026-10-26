"""C's parts the orchestrator calls, and their week-1 versions in placeholders.py.

    Hinter        hint(request, feedback) -> Hint
    Critic        review(text, kind, problem) -> Verdict
    StudentStore  load(student_id), record(memory, misconception_id),
                  misconception_prior(student_id, misconception_id)  <- read by B's diagnose()

B's diagnose() and D's SimStudent are used as they are published.
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence

from .state import Hint, HintRequest, MessageKind, Problem, StudentMemory, Verdict


class Hinter(Protocol):
    def hint(self, request: HintRequest, feedback: Optional[Sequence[str]] = None) -> Hint:
        """Write a hint. `feedback` holds the critic's reasons when this is a rewrite."""


class Critic(Protocol):
    def review(self, text: str, kind: MessageKind, problem: Problem) -> Verdict:
        """Check one message before the student sees it."""


class StudentStore(Protocol):
    def load(self, student_id: str) -> StudentMemory: ...

    def record(self, memory: StudentMemory, misconception_id: str) -> None: ...

    def misconception_prior(self, student_id: str, misconception_id: str) -> Optional[float]: ...
