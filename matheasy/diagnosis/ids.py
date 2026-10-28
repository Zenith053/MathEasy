"""The misconception ID scheme. Every role refers to misconceptions by these strings.

    rule:<MalruleLib name>    a misconception we can run as code,
                              e.g. rule:fractions.add_numerators_denominators
    eedi:<MisconceptionId>    an Eedi misconception known only as text, e.g. eedi:1287
    slip                      a careless mistake with no misconception behind it

The bank (bank.yaml) maps each rule to the Eedi misconceptions it matches, so a
shortlist ranked over Eedi labels can be turned into runnable suspects.
"""

from __future__ import annotations

import numbers
import re

SLIP = "slip"

_RULE_NAME = re.compile(r"^[a-z_]+\.[a-z0-9_]+$")
_EEDI_NUMBER = re.compile(r"^(0|[1-9][0-9]*)$")  # one spelling per number: no leading zeros


def rule_id(name: str) -> str:
    """'fractions.add_numerators_denominators' -> 'rule:fractions.add_numerators_denominators'."""
    if not _RULE_NAME.match(name):
        raise ValueError(f"not a MalruleLib rule name: {name!r}")
    return f"rule:{name}"


def eedi_id(misconception_id: int) -> str:
    """1287 -> 'eedi:1287'. Only a non-negative whole number is accepted, so a bad ID fails here."""
    if (isinstance(misconception_id, bool) or not isinstance(misconception_id, numbers.Integral)
            or misconception_id < 0):
        raise ValueError(f"not an Eedi MisconceptionId: {misconception_id!r}")
    return f"eedi:{int(misconception_id)}"


def split_id(mid: str) -> tuple[str, str]:
    """'rule:fractions.x' -> ('rule', 'fractions.x'); 'slip' -> ('slip', ''). Raises on bad IDs."""
    if mid == SLIP:
        return SLIP, ""
    kind, sep, value = mid.partition(":")
    if sep and kind == "rule" and _RULE_NAME.match(value):
        return kind, value
    if sep and kind == "eedi" and _EEDI_NUMBER.match(value):
        return kind, value
    raise ValueError(f"not a misconception ID: {mid!r} (expected rule:..., eedi:<n> or slip)")
