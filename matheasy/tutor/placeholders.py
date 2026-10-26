"""C's week-1 parts, so the tutor runs end to end. Each is replaced in week 2-3
with the same interface (see parts.py). Don't build on the logic in here.

    TemplateHinter   -> LLM hinter          fixed wording, optional citation from A's retrieve()
    BasicCritic      -> real critic         leak check on numbers only
    InMemoryStore    -> student store and planner
"""

from __future__ import annotations

import re
from typing import Callable, Optional, Sequence

from ..diagnosis import ParseError, equivalent
from .state import Hint, HintRequest, MessageKind, Problem, StudentMemory, Verdict

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?")


class TemplateHinter:
    """Fixed-wording hints. `retrieve` is A's retrieve(); without it, no citation."""

    def __init__(self, retrieve: Optional[Callable[[str], list[dict]]] = None):
        self.retrieve = retrieve

    def hint(self, request: HintRequest, feedback: Optional[Sequence[str]] = None) -> Hint:
        if request.misconception is None:
            text = (f"Look again at your answer {request.student_answer}. "
                    "Which step would you check first?")
        else:
            text = (f"It looks like you used this method: {request.misconception} "
                    "That method doesn't work here. Think about what each part of the question means, "
                    "then try the question again.")
        citations: tuple[str, ...] = ()
        if self.retrieve:
            passages = self.retrieve(f"{request.misconception or ''} {request.topic}")
            if passages:
                citations = (passages[0]["ref"],)
                text += f" (See {citations[0]}.)"
        return Hint(text, citations)


class BasicCritic:
    """Blocks hints and probes that state the problem's answer. Checks numbers only."""

    def review(self, text: str, kind: MessageKind, problem: Problem) -> Verdict:
        if kind in ("hint", "probe"):
            for token in _NUMBER.findall(text):
                try:
                    if equivalent(token.strip(), problem.correct_answer):
                        return Verdict(False, (f"states the answer ({token.strip()})",))
                except ParseError:
                    continue
        return Verdict(True)


class InMemoryStore:
    def __init__(self):
        self._memories: dict[str, StudentMemory] = {}

    def load(self, student_id: str) -> StudentMemory:
        return self._memories.setdefault(student_id, StudentMemory(student_id))

    def record(self, memory: StudentMemory, misconception_id: str) -> None:
        memory.times_diagnosed[misconception_id] = memory.times_diagnosed.get(misconception_id, 0) + 1

    def misconception_prior(self, student_id: str, misconception_id: str) -> Optional[float]:
        """Placeholder until the planner: 0.8 if diagnosed before, else unknown."""
        memory = self._memories.get(student_id)
        return 0.8 if memory and memory.times_diagnosed.get(misconception_id) else None
