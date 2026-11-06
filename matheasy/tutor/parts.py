"""C's parts the orchestrator calls, and their week-1 versions in placeholders.py.

    Hinter        hint(request, feedback, *, llm) -> Hint             hinter.py
    Critic        review(text, kind, problem, *, request, llm) -> Verdict   critic.py
    StudentStore  load(student_id), begin_session(memory, index), record(memory, misconception_id),
                  observe(memory, problem, answer, correct, topics), end_session(memory),
                  misconception_prior(student_id, misconception_id)  <- read by B's diagnose()
                  store.py; the updates are the planner's (planner.py)

    Tutor         start(state, problem), respond(state, text) -> TutorReply;
                  .log and .store. The Orchestrator (full, b2) and B0Tutor (b0).

B's diagnose() and D's SimStudent are used as they are published.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence

from .state import Hint, HintRequest, MessageKind, Problem, StudentMemory, TutorReply, TutorState, Verdict


class Hinter(Protocol):
    def hint(self, request: HintRequest, feedback: Optional[Sequence[str]] = None, *, llm: Any = None) -> Hint:
        """Write a hint. `feedback` holds the critic's reasons when this is a rewrite;
        `llm` is the tutor's LLM session for this student session."""


class Critic(Protocol):
    def review(self, text: str, kind: MessageKind, problem: Problem, *,
               request: Optional[HintRequest] = None, llm: Any = None, unsolved: Any = None) -> Verdict:
        """Check one message before the student sees it. `request` is set for hints; `unsolved`
        maps the session's other open items to their answers."""


class StudentStore(Protocol):
    def load(self, student_id: str) -> StudentMemory: ...

    def begin_session(self, memory: StudentMemory, session_index: int) -> None: ...

    def record(self, memory: StudentMemory, misconception_id: str) -> None: ...

    def observe(self, memory: StudentMemory, problem: Problem, answer: str, correct: bool,
                topics: Sequence[str] = ()) -> None: ...

    def end_session(self, memory: StudentMemory) -> None: ...

    def misconception_prior(self, student_id: str, misconception_id: str) -> Optional[float]: ...


class Tutor(Protocol):
    """Anything run_session() can drive: the full tutor or a baseline."""

    log: Any
    store: StudentStore

    def start(self, state: TutorState, problem: Problem) -> TutorReply: ...

    def respond(self, state: TutorState, student_text: str) -> TutorReply: ...
