"""Executable misconception rules, run on the student's own expression.

    from matheasy.diagnosis import run_rule, rule_names

    run_rule("fractions.add_numerators_denominators", "1/2 + 1/3")       # '2/5'
    run_rule("rule:fractions.add_numerators_denominators", "1/2 + 1/3")  # same: the ID works too
    run_rule("fractions.add_numerators_denominators", "2x + 3")          # None: the rule does not apply

A rule takes a parsed expression (usually LocatedError.source_node) and returns the
answer a student holding that misconception would write, or None if the
misconception says nothing about that expression.

The rules re-implement MalruleLib (Chen, Liu and Sonkar, 2026,
https://github.com/sonkar-lab/malrulelib, MalruleLib Non-Commercial License): same
names, same wrong answers (simplified, as MalruleLib does), but run on any
expression the student writes instead of only on MalruleLib's generated problems.

Rules held back by D (eval/held_back_rules.json) can never be registered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

from ..ids import split_id
from ..maths import Node, parse

HELD_BACK_FILE = Path(__file__).resolve().parents[3] / "eval" / "held_back_rules.json"


def held_back_rules() -> frozenset[str]:
    """MalruleLib names D set aside. They must never enter the bank or the rule runner."""
    data = json.loads(HELD_BACK_FILE.read_text())
    return frozenset(r for rules in data["held_back"].values() for r in rules)


class HeldBackRuleError(ValueError):
    """Someone tried to register a rule that D held back for evaluation."""


@dataclass(frozen=True)
class Rule:
    name: str   # MalruleLib name, e.g. fractions.add_numerators_denominators
    topic: str  # fractions | decimals | expressions
    apply: Callable[[Node], Optional[Node]]


_REGISTRY: dict[str, Rule] = {}
_HELD_BACK = held_back_rules()


def rule(name: str, topic: str):
    """Decorator that registers a rule function under its MalruleLib name."""
    def register(fn: Callable[[Node], Optional[Node]]):
        if name in _HELD_BACK:
            raise HeldBackRuleError(f"{name} is held back for evaluation and cannot be a rule")
        if name in _REGISTRY:
            raise ValueError(f"rule {name} registered twice")
        _REGISTRY[name] = Rule(name, topic, fn)
        return fn
    return register


def _name(rule_or_id: str) -> str:
    return split_id(rule_or_id)[1] if rule_or_id.startswith("rule:") else rule_or_id


def get_rule(rule_or_id: str) -> Rule:
    try:
        return _REGISTRY[_name(rule_or_id)]
    except KeyError:
        raise KeyError(f"no such rule: {rule_or_id}") from None


def rule_names(topic: Optional[str] = None) -> list[str]:
    """Names of every registered rule, optionally only one topic's."""
    return sorted(n for n, r in _REGISTRY.items() if topic is None or r.topic == topic)


def run_rule(rule_or_id: str, expr: Union[str, Node]) -> Optional[str]:
    """The answer the misconception predicts for `expr`, or None if it does not apply."""
    result = get_rule(rule_or_id).apply(parse(expr))
    return None if result is None else str(result)


from . import fractions  # noqa: E402,F401  (registers the rules)
