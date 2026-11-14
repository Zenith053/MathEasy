"""The words for a diagnosed misconception, for the hinter and critic (Role C).

    rule:<name>   B's bank text, e.g. "Adds the numerators and adds the denominators, ..."
    eedi:<n>      the Eedi misconception name, from A's Eedi data
    fact:<a><op><b>  a weak number fact: "Gets the number fact 3 + 6 wrong, more than once."
    slip          None: a careless mistake gets a general hint, not a targeted one

Returns None when the words are not available (e.g. no Eedi data on this
machine); the orchestrator then gives a general hint instead of crashing.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from ..diagnosis import load_bank, split_id
from ..diagnosis.facts import SHOW

EEDI_NAMES = Path(__file__).resolve().parents[2] / "data" / "eedi" / "raw" / "misconception_mapping.csv"


def misconception_text(mid: str) -> Optional[str]:
    try:
        kind, value = split_id(mid)
    except ValueError:
        return None
    if kind == "rule":
        bank = load_bank()
        return bank[mid].text if mid in bank else None
    if kind == "eedi":
        return _eedi_names().get(int(value))
    if kind == "fact":  # 3+6 -> "Gets the number fact 3 + 6 wrong, more than once." (no answer in it)
        a, op, b = re.match(r"^(\d+)([-+*/])(\d+)$", value).groups()
        return f"Gets the number fact {a} {SHOW[op]} {b} wrong, more than once."
    return None  # slip


@lru_cache(maxsize=1)
def _eedi_names() -> dict[int, str]:
    if not EEDI_NAMES.exists():
        return {}
    with EEDI_NAMES.open(encoding="utf-8") as f:
        return {int(row["MisconceptionId"]): row["MisconceptionName"].strip() for row in csv.DictReader(f)}
