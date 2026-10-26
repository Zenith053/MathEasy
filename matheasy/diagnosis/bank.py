"""The misconception bank: every misconception the diagnoser can name, and how rules map to Eedi.

    from matheasy.diagnosis import load_bank

    bank = load_bank()
    bank["rule:fractions.add_numerators_denominators"].text
    bank.rules()          # IDs of every runnable misconception
    bank.for_eedi(1287)   # rule IDs that match an Eedi misconception
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional

import yaml

from .ids import split_id
from .rules import HeldBackRuleError, get_rule, held_back_rules

BANK_FILE = Path(__file__).with_name("bank.yaml")


@dataclass(frozen=True)
class Misconception:
    id: str
    topic: str
    text: str
    eedi: tuple[int, ...]  # Eedi MisconceptionIds this misconception matches

    @property
    def rule(self) -> Optional[str]:
        """The MalruleLib name if this misconception can be run as code."""
        kind, value = split_id(self.id)
        return value if kind == "rule" else None


class Bank:
    def __init__(self, misconceptions: list[Misconception]):
        self._by_id = {m.id: m for m in misconceptions}

    def __getitem__(self, mid: str) -> Misconception:
        return self._by_id[mid]

    def __contains__(self, mid: str) -> bool:
        return mid in self._by_id

    def __iter__(self) -> Iterator[Misconception]:
        return iter(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)

    def rules(self, topic: Optional[str] = None) -> list[str]:
        return [m.id for m in self if m.rule and (topic is None or m.topic == topic)]

    def for_eedi(self, misconception_id: int) -> list[str]:
        return [m.id for m in self if misconception_id in m.eedi]


def load_bank(path: Optional[Path] = None) -> Bank:
    """Read and check the bank. The default bank is read once and shared."""
    return _load(Path(path) if path else BANK_FILE)


@lru_cache(maxsize=None)
def _load(path: Path) -> Bank:
    raw = yaml.safe_load(path.read_text())["misconceptions"]
    held = held_back_rules()
    items = []
    for mid, entry in raw.items():
        m = Misconception(mid, entry["topic"], entry["text"].strip(), tuple(int(e) for e in entry.get("eedi", [])))
        if m.rule in held:
            raise HeldBackRuleError(f"{m.rule} is held back for evaluation and cannot be in the bank")
        if m.rule:
            get_rule(m.rule)  # every runnable misconception must have a rule
        items.append(m)
    return Bank(items)
