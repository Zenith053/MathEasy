"""The analysis of a wrong answer (Role C): what the student sees, LeetCode style.

docs/design_decisions.md, "What we are trying to build": for each wrong answer
the tutor shows

    where it went wrong     the step or operation that broke ("1/2 + 1/3 became 2/5")
    what the mistake is     the rule's description, or the LLM fallback's verified reading
    which topics            B's topic labels (diagnosis.topics)
    what to practise        similar problems on those topics, from B's probe bank,
                            those the same mistake gets wrong first

All four come from B's diagnosis, in code. Every part is checked with the critic's
leak check and left out if it would give the answer away: the student sees what
went wrong, never what the right result is. The hinter's hint follows the analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence

from ..diagnosis import Diagnosis
from ..diagnosis.probes import similar_items
from .critic import leaks_answer
from .describe import misconception_text
from .planner import exposes
from .state import Problem

PRACTICE_ITEMS = 2


@dataclass(frozen=True)
class Analysis:
    where: Optional[str]
    mistake: Optional[str]
    topics: tuple[str, ...]
    practice: tuple[str, ...]

    def text(self) -> str:
        lines = []
        if self.where:
            lines.append(f"Where it went wrong: {self.where}.")
        if self.mistake:
            lines.append(f"The mistake: {self.mistake.rstrip('.')}.")
        if self.topics:
            lines.append(f"Topics: {', '.join(topic_name(t) for t in self.topics)}.")
        return "\n".join(lines)

    def practice_text(self) -> str:
        return f"Practise next: {'; '.join(self.practice)}." if self.practice else ""

    def to_log(self) -> dict:
        return {**asdict(self), "topics": list(self.topics), "practice": list(self.practice)}


def topic_name(topic: str) -> str:
    return topic.replace("_", " ")


def analyse(diagnosis: Optional[Diagnosis], problem: Problem, seen: Sequence[str] = (),
            unsolved: Mapping[str, str] = {}) -> Analysis:
    """The analysis of a wrong answer to `problem`; empty parts where the diagnosis says nothing.
    `unsolved` maps the session's other open items to their answers, which must not be given away either."""
    if diagnosis is None:
        return Analysis(None, None, (), ())
    answers = {problem.question: problem.correct_answer, **unsolved}

    def safe(text: Optional[str]) -> Optional[str]:
        ok = text and not any(leaks_answer(text, answer, question) for question, answer in answers.items())
        return text if ok else None

    located = diagnosis.located
    where = safe(f"{located.source} became {located.wrong}") if located else None
    topics = tuple(t for t in diagnosis.topics if t != "other")
    candidates = [q for q in similar_items(topics, [problem.question, *seen], n=50) if safe(q)]
    if diagnosis.misconception:  # problems the same mistake gets wrong come first
        candidates.sort(key=lambda q: exposes(diagnosis.misconception, q) is None)
    practice = tuple(candidates[:PRACTICE_ITEMS])
    return Analysis(where, safe(mistake(diagnosis, problem)), topics, practice)


def mistake(diagnosis: Diagnosis, problem: Problem) -> Optional[str]:
    """The mistake in words: a clear diagnosis's description, or for a step no rule explains,
    the LLM fallback's reading if code verified it and it is a misconception (not a slip).
    One rule reproducing the wrong step names it exactly. The explainer saw the correct
    line, so words that give the answer away are dropped."""
    if diagnosis.status == "clear":
        return misconception_text(diagnosis.misconception)
    e = diagnosis.explanation
    if diagnosis.status != "unknown" or e is None or e.status != "verified" or e.kind != "misconception":
        return None
    rules = {w.rule for w in e.wrong_steps if w.rule}
    words = [misconception_text(next(iter(rules))) if len(rules) == 1 else None,
             f"{e.mistake}: {e.explanation}" if e.explanation else None, e.mistake]
    return next((w for w in words if w and not leaks_answer(w, problem.correct_answer, problem.question)), None)
