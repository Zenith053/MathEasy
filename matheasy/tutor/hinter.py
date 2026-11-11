"""The hinter (Role C): a prompted LLM that writes one hint aimed at the diagnosed misconception.

It sees only a HintRequest (no correct answer exists in it) and the textbook
passages it retrieves with A's retrieve(). It must cite exactly one of those
passages: the JSON schema only allows their references, so it cannot invent a
citation. On a rewrite it also gets the critic's reasons.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from ..llm import LLMOutputError, Session
from .state import CERTAINTY, Hint, HintRequest

SYSTEM = """\
You write hints for a maths tutor used by students aged 11 to 14.

You are told the question, the student's wrong answer, which part of their work went wrong, \
what they already did right (when known) and, when it is known, the misconception that explains \
the mistake and how sure that is. Write one hint that helps the student see why their method \
does not work and try again.

Rules:
- If the misconception is certain or likely, aim the hint at it. If it is only possible, aim the \
hint at the part that went wrong, and mention the misconception at most as one thing to check. \
If no misconception is given, help the student find and check the step that went wrong, without \
guessing a cause.
- You may build on what the student already did right.
- Never state the final answer, never work out the correct result of the step, and never \
do the step for the student.
- Use at most three short sentences in plain words, and end with one question the student \
can answer.
- Choose the one textbook section from the list that best explains the idea behind your hint, \
and give its reference exactly as written."""

PASSAGE_CHARS = 700  # per passage, to keep prompts small


class LLMHinter:
    role = "hinter"

    def __init__(self, retrieve: Optional[Callable[..., list[dict]]] = None, k: int = 3):
        self.retrieve = retrieve
        self.k = k

    def passages(self, request: HintRequest) -> list[dict]:
        if self.retrieve is None:
            return []
        query = f"{request.misconception or request.wrong_part or ''} {request.topic}".strip()
        return self.retrieve(query, k=self.k)

    def hint(self, request: HintRequest, feedback: Optional[Sequence[str]] = None, *,
             llm: Optional[Session] = None, **_) -> Hint:
        if llm is None:
            raise ValueError("LLMHinter needs the tutor's LLM session")
        passages = self.passages(request)
        refs = [p["ref"] for p in passages]
        result = llm.call(self.role, [{"role": "user", "content": _prompt(request, passages, feedback)}],
                          system=SYSTEM, json_schema=_schema(refs))
        text = str(result.parsed.get("hint", "")).strip()
        if not text:
            raise LLMOutputError("hinter returned an empty hint")
        citation = result.parsed.get("citation")
        if refs and citation not in refs:
            raise LLMOutputError(f"hinter cited {citation!r}, which was not offered")
        return Hint(text, (citation,) if refs else ())


def _schema(refs: list[str]) -> dict:
    properties = {"hint": {"type": "string"}}
    if refs:
        properties["citation"] = {"type": "string", "enum": refs}
    return {"type": "object", "properties": properties, "required": list(properties)}


def _prompt(request: HintRequest, passages: list[dict], feedback: Optional[Sequence[str]]) -> str:
    lines = [
        f"Topic: {request.topic}",
        f"Question: {request.question}",
        f"Student's answer: {request.student_answer}",
        f"What went wrong: {request.wrong_part or 'not located'}",
        f"Misconception: {request.misconception or 'not known'}",
    ]
    if request.misconception and request.certainty:
        lines.append(f"How sure: {CERTAINTY[request.certainty]}")
    if request.right:
        lines.append(f"Already right: {'; '.join(request.right)}")
    if passages:
        lines.append("\nTextbook sections:")
        lines += [f"[{p['ref']}]\n{p['text'][:PASSAGE_CHARS]}" for p in passages]
    if feedback:
        lines.append("\nA reviewer rejected your previous hint for these reasons. "
                     "Write a new hint that fixes them:")
        lines += [f"- {reason}" for reason in feedback]
    return "\n".join(lines)
