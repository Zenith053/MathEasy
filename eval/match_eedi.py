"""Fill in eedi_status for the held-back rules (owner: D).

Each held-back rule is either "known only as text" (an Eedi misconception describes the
same mistake, so the tutor can still name it through the shortlister) or "never seen"
(nothing in Eedi matches). The status decides which results group its students are in
(students.student_group) and what counts as a right diagnosis (metrics.accepted).

A person decides each match; this script only suggests candidates and records the
decision. Needs data/eedi/raw/misconception_mapping.csv (Kaggle, under the competition
terms; see matheasy/shortlister/prepare_eedi.py).

    PYTHONPATH=. .venv/bin/python eval/match_eedi.py suggest                 # top candidates for each rule
    PYTHONPATH=. .venv/bin/python eval/match_eedi.py set fractions.subtract_across 1287
    PYTHONPATH=. .venv/bin/python eval/match_eedi.py set decimals.right_align_decimals never_seen
    PYTHONPATH=. .venv/bin/python eval/match_eedi.py status                  # what is decided so far

Matching rule of thumb: the Eedi name must describe the same wrong method on the same
kind of question (not just the same topic). If two candidates both fit, pick the one
whose wording is closest to the rule's example, and note the other in the commit message.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from matheasy.diagnosis.rules import HELD_BACK_FILE  # noqa: E402
from matheasy.evaluation.held_back import held_back_names, held_back_rule  # noqa: E402

NEVER_SEEN = "never_seen"


def suggestions(name: str, k: int = 8) -> list[tuple[int, str, float]]:
    """A's BM25 shortlister over Eedi misconception names, queried with the rule's description."""
    from matheasy.shortlister.shortlist import shortlist
    return shortlist(held_back_rule(name).description, k=k)


def eedi_ids() -> set[int]:
    from matheasy.shortlister.shortlist import _index
    return set(_index()[0])


def set_status(name: str, value: str, path: Path = HELD_BACK_FILE, known_ids: Optional[set[int]] = None) -> dict:
    """Record `value` (an Eedi MisconceptionId, or never_seen) for held-back rule `name`."""
    data = json.loads(path.read_text())
    if name not in data["eedi_status"]:
        raise ValueError(f"{name} is not a held-back rule; choose from {sorted(data['eedi_status'])}")
    if value == NEVER_SEEN:
        status: object = NEVER_SEEN
    else:
        if not value.isdigit():
            raise ValueError(f"expected an Eedi MisconceptionId or {NEVER_SEEN}, got {value!r}")
        status = int(value)
        if known_ids is not None and status not in known_ids:
            raise ValueError(f"{status} is not an Eedi MisconceptionId")
    data["eedi_status"][name] = status
    path.write_text(json.dumps(data, indent=2) + "\n")
    return data["eedi_status"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("suggest")
    s.add_argument("--k", type=int, default=8)
    s.add_argument("rules", nargs="*", help="default: every held-back rule not yet decided")
    t = sub.add_parser("set")
    t.add_argument("rule")
    t.add_argument("value", help=f"an Eedi MisconceptionId, or {NEVER_SEEN}")
    sub.add_parser("status")
    args = parser.parse_args(argv)

    status = json.loads(HELD_BACK_FILE.read_text())["eedi_status"]
    if args.command == "status":
        for name in held_back_names():
            print(f"{name:52} {status.get(name) if status.get(name) is not None else '(not decided)'}")
    elif args.command == "set":
        set_status(args.rule, args.value, known_ids=eedi_ids() if args.value != NEVER_SEEN else None)
        print(f"{args.rule}: {args.value}")
    else:
        for name in args.rules or [n for n in held_back_names() if status.get(n) is None]:
            print(f"\n{name}\n  rule: {held_back_rule(name).description}")
            for mid, text, score in suggestions(name, args.k):
                print(f"  {mid:>5}  {score:5.1f}  {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
