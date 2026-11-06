"""The probe bank: a finite list of questions, each with what every rule predicts on it.

    probe_items()                        every question in the bank, for D's test check and A's cache
    choose_probe(suspects, asked)        the question that best tells the tied rules apart
    similar_items(topics, seen, n)       practice questions on a student's topics

Two jobs in the tutor. When rules tie on a step, diagnose() asks the question that
separates them and keeps the rules that predicted the student's answer. After any
diagnosis, similar_items() gives the "try a similar problem" follow-up, by topic.

The bank is probes.yaml, built by eval/build_probes.py, so choosing a probe runs no
rule. Numbers stay in the agreed ranges (docs/diagnosis_rules.md). There are no
comparison items until their format is agreed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import yaml

from .maths import equivalent

BANK = Path(__file__).with_name("probes.yaml")


@dataclass(frozen=True)
class Probe:
    """A follow-up question chosen to tell the remaining suspects apart."""
    question: str                 # e.g. "1/3 + 1/3"
    predictions: dict[str, str]   # misconception ID -> the answer it predicts


@lru_cache(maxsize=1)
def _bank() -> tuple[dict, ...]:
    return tuple(yaml.safe_load(BANK.read_text()))


def probe_items() -> list[str]:
    """Every question in the bank: the complete list."""
    return [entry["question"] for entry in _bank()]


def choose_probe(suspects: Sequence[str], asked: Sequence[str] = ()) -> Optional[Probe]:
    """The unasked question that splits the suspects into the most different answers.

    A suspect a question does not predict for is left out of its predictions: on that
    question it gives no wrong answer of its own. None if no question splits them.
    """
    best, best_groups = None, 1
    for entry in _bank():
        if entry["question"] in asked:
            continue
        predicted = {s: entry["predictions"][s] for s in suspects if s in entry["predictions"]}
        groups = _groups(predicted) + (len(predicted) < len(suspects))  # the rest answer some other way
        if predicted and groups > best_groups:
            best, best_groups = Probe(entry["question"], predicted), groups
    return best


def _groups(predicted: dict[str, str]) -> int:
    """How many different answers these predictions give."""
    answers: list[str] = []
    for answer in predicted.values():
        if not any(equivalent(answer, other) for other in answers):
            answers.append(answer)
    return len(answers)


def similar_items(topics: Sequence[str], seen: Sequence[str] = (), n: int = 3) -> list[str]:
    """Up to n bank questions on these topics that the student has not seen."""
    found = [e["question"] for e in _bank() if set(e["topics"]) & set(topics) and e["question"] not in seen]
    return found[:n]
