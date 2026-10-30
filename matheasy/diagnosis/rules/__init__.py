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

from ..ids import rule_id, split_id
from ..maths import Node, parse

HELD_BACK_FILE = Path(__file__).resolve().parents[3] / "eval" / "held_back_rules.json"


def held_back_rules() -> frozenset[str]:
    """MalruleLib names D set aside. They must never enter the bank or the rule runner."""
    data = json.loads(HELD_BACK_FILE.read_text())
    return frozenset(r for rules in data["held_back"].values() for r in rules)


class HeldBackRuleError(ValueError):
    """Someone tried to register a rule that D held back for evaluation."""


class RuleError(RuntimeError):
    """A rule crashed: the rule has a bug. Callers skip it and carry on with the other rules."""


@dataclass(frozen=True)
class Rule:
    name: str   # MalruleLib name, e.g. fractions.add_numerators_denominators
    apply: Callable[[Node], Optional[Node]]


_REGISTRY: dict[str, Rule] = {}
_ALIASES: dict[str, str] = {}  # twin name -> the rule registered for it
_HELD_BACK = held_back_rules()


def rule(name: str, twins: tuple[str, ...] = ()):
    """Decorator that registers a rule function under its MalruleLib name.

    `twins` are MalruleLib names that give the same wrong answer on every problem the
    rule applies to. No follow-up question can tell them apart, so they are one
    suspect: run_rule() accepts their names, and the diagnoser names the rule.
    """
    for n in (name, *twins):
        rule_id(n)  # a name no misconception ID can refer to is refused here, not found missing later

    def register(fn: Callable[[Node], Optional[Node]]):
        for n in (name, *twins):
            if n in _HELD_BACK:
                raise HeldBackRuleError(f"{n} is held back for evaluation and cannot be a rule")
            if n in _REGISTRY or n in _ALIASES:
                raise ValueError(f"rule {n} registered twice")
        _REGISTRY[name] = Rule(name, fn)
        _ALIASES.update({twin: name for twin in twins})
        return fn
    return register


def _name(rule_or_id: str) -> str:
    name = split_id(rule_or_id)[1] if rule_or_id.startswith("rule:") else rule_or_id
    return _ALIASES.get(name, name)


def get_rule(rule_or_id: str) -> Rule:
    try:
        return _REGISTRY[_name(rule_or_id)]
    except KeyError:
        raise KeyError(f"no such rule: {rule_or_id}") from None


def rule_names() -> list[str]:
    """Names of every registered rule (not their twins)."""
    return sorted(_REGISTRY)


def twin_names() -> dict[str, str]:
    """Each twin's MalruleLib name -> the rule that stands for it."""
    return dict(_ALIASES)


def run_rule(rule_or_id: str, expr: Union[str, Node]) -> Optional[str]:
    """The answer the misconception predicts for `expr`, or None if it does not apply.

    Raises KeyError for an unknown rule and ParseError for an unreadable expression
    (the caller's mistakes), and RuleError if the rule itself crashes (a bug in it).
    """
    found = get_rule(rule_or_id)
    node = parse(expr)
    try:
        result = found.apply(node)
    except Exception as e:  # only errors from inside the rule function
        raise RuleError(f"rule {found.name} failed on {node}: {type(e).__name__}: {e}") from e
    return None if result is None else str(result)


from . import (  # noqa: E402,F401  (registers the rules)
    algebra, decimals, exponents, fractions, multiplication_division, negative_numbers, order_of_operations,
    ratios,
)
