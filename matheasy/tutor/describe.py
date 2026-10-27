"""The words for a diagnosed misconception, for the hinter and critic (Role C).

    rule:<name>   B's bank text, e.g. "Adds the numerators and adds the denominators, ..."
    eedi:<n>      the Eedi misconception name, from A's Eedi data
    slip          None: a careless mistake gets a general hint, not a targeted one

Returns None when the words are not available (e.g. no Eedi data on this
machine); the orchestrator then gives a general hint instead of crashing.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Optional

from ..diagnosis import load_bank, split_id

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
    return None  # slip


@lru_cache(maxsize=1)
def _eedi_names() -> dict[int, str]:
    if not EEDI_NAMES.exists():
        return {}
    with EEDI_NAMES.open(encoding="utf-8") as f:
        return {int(row["MisconceptionId"]): row["MisconceptionName"].strip() for row in csv.DictReader(f)}
