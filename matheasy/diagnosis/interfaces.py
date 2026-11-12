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

import sympy

from .bank import Bank, load_bank
from .ids import eedi_id
from .maths import UNDEFINED, LocatedError, Node, Rel, equivalent, parse, topic_of, undefined
from .rules import RuleError, run_rule


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
    are left out. A rule that crashes is skipped, so one buggy rule cannot stop a
    diagnosis; `failed` lists the rules that crashed on the last call, for the logs.
    """

    def __init__(self, bank: Optional[Bank] = None):
        self.bank = bank or load_bank()
        self.failed: list[str] = []

    def __call__(self, error: LocatedError, k: int = 25) -> list[Candidate]:
        self.failed = []
        hits = self._hits(error.source_node, error.wrong_node)
        if not hits and error.source != error.before:
            # Narrowing can strip what a rule needs: it pairs off terms after multiplying out a
            # minus, so -(x + 1) -> -x + 1 becomes -1 -> 1. Then try the whole line as well.
            hits = self._hits(parse(error.before), parse(error.after))
        return hits[:k]

    def _hits(self, source: Node, wrong: Node) -> list[Candidate]:
        hits = []
        for mid in self.bank.rules():
            try:
                predicted = run_rule(mid, source)
            except RuleError as e:
                self.failed.append(str(e))
                continue
            if predicted is not None and equivalent(predicted, wrong):
                hits.append(Candidate(mid, 1.0))
        return hits


class EediShortlister:
    """A's shortlist() behind the Shortlister interface.

    Builds the query with A's build_query() from the located error, so the live
    tutor and training use the same query format, and returns eedi: IDs. Needs
    A's Eedi data (data/eedi/raw/misconception_mapping.csv); see
    matheasy/shortlister/prepare_eedi.py. Pass `shortlist` to use another ranker
    with the same signature, e.g. the fine-tuned model.
    """

    def __init__(self, shortlist=None):
        if shortlist is None:
            from matheasy.shortlister import shortlist
        self.shortlist = shortlist

    def __call__(self, error: LocatedError, k: int = 25) -> list[Candidate]:
        ranked = self.shortlist(query(error), k=k)
        return [Candidate(eedi_id(mid), float(score)) for mid, _name, score in ranked]


def query(error: LocatedError) -> str:
    """The shortlister query for a located error, built with A's build_query()."""
    from matheasy.shortlister.query import build_query, sympy_diff

    difference = None
    if undefined(error.wrong_node) or undefined(error.source_node):
        difference = UNDEFINED  # A's sympy_diff would write 'zoo'
    elif not isinstance(error.wrong_node, Rel) and not isinstance(error.source_node, Rel):
        difference = sympy_diff(error.wrong_node.to_sympy(), sympy.simplify(error.source_node.to_sympy()))
    return build_query(
        topic=topic_of(error.source_node, error.wrong_node),  # of the wrong part, not the whole line
        before=error.before, student=error.after, correct=error.correct,
        changed=f"{error.expected_term} -> {error.wrong_term}",  # A's convention: correct -> student
        difference=difference,  # None: build_query falls back to its own diff
    )


class NoMemory:
    """Placeholder student store: knows nothing about any student."""

    def misconception_prior(self, student_id: str, misconception_id: str) -> Optional[float]:
        return None
