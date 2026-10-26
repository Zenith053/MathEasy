"""What the diagnoser needs from other roles, and placeholders until the real versions land.

Shortlister (owner: A). shortlist(error, k) ranks likely misconceptions for a located
error. A builds the query from the LocatedError with build_query(), so training
and the live tutor use the same query. Candidates may be eedi: or rule: IDs; the
bank maps Eedi IDs to rules.

StudentStore (owner: C). The diagnoser only reads it: how likely is it that this
student holds this misconception, from earlier sessions? None if unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from .bank import Bank, load_bank
from .maths import LocatedError, equivalent
from .rules import run_rule


@dataclass(frozen=True)
class Candidate:
    id: str       # a misconception ID (see ids.py)
    score: float  # higher is more likely; need not sum to 1


class Shortlister(Protocol):
    def __call__(self, error: LocatedError, k: int = 25) -> list[Candidate]: ...


class StudentStore(Protocol):
    def misconception_prior(self, student_id: str, misconception_id: str) -> Optional[float]: ...


class RuleShortlister:
    """Placeholder for A's shortlist(): ranks the bank's rules that reproduce the student's answer.

    Every rule whose prediction equals what the student wrote scores 1; the rest
    are left out. Replace with A's shortlist() as soon as it exists.
    """

    def __init__(self, bank: Optional[Bank] = None):
        self.bank = bank or load_bank()

    def __call__(self, error: LocatedError, k: int = 25) -> list[Candidate]:
        hits = []
        for mid in self.bank.rules():
            predicted = run_rule(mid, error.source_node)
            if predicted is not None and equivalent(predicted, error.wrong_node):
                hits.append(Candidate(mid, 1.0))
        return hits[:k]


class NoMemory:
    """Placeholder student store: knows nothing about any student."""

    def misconception_prior(self, student_id: str, misconception_id: str) -> Optional[float]:
        return None
