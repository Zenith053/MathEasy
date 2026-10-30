"""C's week-1 parts, so the tutor runs end to end. Each is replaced in week 2-3
with the same interface (see parts.py). Don't build on the logic in here.

    TemplateHinter   offline hinter         fixed wording, for demos and tests without an API key

The real parts: hinter.LLMHinter, critic.TutorCritic, store.JsonStudentStore.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from .state import Hint, HintRequest


class TemplateHinter:
    """Fixed-wording hints. `retrieve` is A's retrieve(); without it, no citation."""

    def __init__(self, retrieve: Optional[Callable[[str], list[dict]]] = None):
        self.retrieve = retrieve

    def hint(self, request: HintRequest, feedback: Optional[Sequence[str]] = None, **_) -> Hint:
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
        return Hint(text, citations)
