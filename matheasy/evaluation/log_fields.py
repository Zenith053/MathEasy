"""The fields every run log must contain, and a checker for them (owner: D).

C's experiment runner writes JSONL: one record per line, each with an "event". Every
metric in section 12 of the proposal is computed only from these records, so a
missing field is a metric we cannot compute. The comment on each event says which
metrics read it.

    python -m matheasy.evaluation runs/                     # check every .jsonl under runs/
    check_files(["runs/run-1.jsonl"])                       # -> list of problems, empty if clean

The LLM client's own records (runs/llm_logs/, events call | error | budget_exceeded)
are accepted as they are; the checker only uses their config_hash.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, Optional

from ..diagnosis import held_back_rules, split_id
from .students import GROUPS

CONDITIONS = ("full", "b0", "b1", "b2")
STATUSES = ("clear", "ambiguous", "unknown")
PHASES = ("pre", "post")
ENDED_BY = ("solved", "max_turns", "budget")
LLM_EVENTS = ("call", "error", "budget_exceeded")

NONE = type(None)

# On every record. session_id is the same string C passes to llm.session(), so the
# LLM log can be joined to the run log.
COMMON = {
    "ts": str,
    "run_id": str,
    "session_id": str,
    "student_id": str,
    "condition": str,      # one of CONDITIONS
    "config_hash": str,    # LLMClient.config.hash: all shards must share one
    "event": str,
}

EVENTS = {
    # Groups results (has a rule / known as text / never seen) and orders sessions for retention.
    "session_start": {
        "session_index": int,          # 0 for a student's first session, then 1, 2, ...
        "student_group": str,          # one of GROUPS, from SimStudent.group
        "planted": list,               # misconception IDs the student holds, from SimStudent.misconception_id
        "attack": (str, NONE),         # adversarial attack type (Zhao et al., 2026), None for ordinary students
    },
    # One tutor message and the student's reply. Leak rate, unsafe-action rate, learning in-session.
    "turn": {
        "turn": int,
        "tutor_text": str,             # exactly what reached the student
        "critic_passed": (bool, NONE),  # None only for conditions with no critic (b0, b1, b2)
        "item": (str, NONE),           # the maths the student was asked to work
        "correct_answer": (str, NONE),
        "student_text": str,
        "student_answer": (str, NONE),
        "student_correct": (bool, NONE),
        "followed_rule": bool,         # ground truth from the student: answer came from the misconception
        "holds_misconception": bool,   # ground truth from the student, after this turn
    },
    # Top-1 accuracy, follow-ups per diagnosis, calibration, over-claiming. Diagnosis.to_log() plus turn.
    "diagnosis": {
        "turn": int,
        "status": str,                 # one of STATUSES
        "misconception": (str, NONE),  # a misconception ID, set when status is clear
        "beliefs": dict,               # misconception ID -> probability
        "follow_ups": int,             # probes asked before this diagnosis
        "mode": str,
    },
    # Citation quality, critic rewrites.
    "hint": {
        "turn": int,
        "text": str,
        "citations": list,             # textbook section references
        "rewrites": int,               # critic rewrites before it passed (at most 2)
        "safe_template": bool,         # the critic gave up and the safe template was sent
    },
    # Tool-call validity and recovery.
    "tool_call": {
        "turn": int,
        "tool": str,
        "args_valid": bool,
        "ok": bool,
        "error": (str, NONE),
        "is_retry": bool,
    },
    # Normalised learning gain (pre/post) and retention (post items in later sessions).
    "test": {
        "phase": str,                  # one of PHASES
        "item": str,
        "correct_answer": str,
        "student_answer": str,
        "correct": bool,
    },
    # Retention and budget fairness.
    "session_end": {
        "turns": int,
        "ended_by": str,               # one of ENDED_BY
        "tokens_used": int,            # from the tutor's LLM session
        "holds": list,                 # misconception IDs the student still holds: ground truth
    },
}


def _type_name(t) -> str:
    if isinstance(t, tuple):
        return " or ".join("null" if x is NONE else x.__name__ for x in t)
    return t.__name__


def _type_ok(value, t) -> bool:
    if isinstance(value, bool) and (t is int or (isinstance(t, tuple) and bool not in t and int in t)):
        return False  # JSON true is not a count
    return isinstance(value, t)


def _misconception_problem(mid: str, held: frozenset[str]) -> Optional[str]:
    try:
        kind, name = split_id(mid)
    except ValueError as e:
        return str(e)
    if kind == "rule" and name in held:
        return f"tutor diagnosed held-back rule {mid} as runnable: it has leaked into the bank"
    return None


def check_record(rec: dict, held: frozenset[str] = frozenset()) -> list[str]:
    """Problems with one run-log record, on its own."""
    event = rec.get("event")
    if event in LLM_EVENTS:
        return [] if isinstance(rec.get("config_hash"), str) else ["LLM record without config_hash"]
    if event not in EVENTS:
        return [f"unknown event {event!r}; expected one of {sorted(EVENTS)}"]
    problems = []
    for name, t in {**COMMON, **EVENTS[event]}.items():
        if name not in rec:
            problems.append(f"{event}: missing {name}")
        elif not _type_ok(rec[name], t):
            problems.append(f"{event}: {name} should be {_type_name(t)}, got {rec[name]!r}")
    if problems:
        return problems

    allowed = {"condition": CONDITIONS, "student_group": GROUPS, "status": STATUSES,
               "phase": PHASES, "ended_by": ENDED_BY}
    for name, values in allowed.items():
        if name in rec and rec[name] not in values:
            problems.append(f"{event}: {name} {rec[name]!r} is not one of {values}")

    # Students may hold held-back misconceptions; the tutor must never diagnose one as a runnable rule.
    planted = list(rec.get("planted", [])) + list(rec.get("holds", []))
    diagnosed = list(rec.get("beliefs", {})) + ([rec["misconception"]] if rec.get("misconception") else [])
    for mid, refused in [(m, frozenset()) for m in planted] + [(m, held) for m in diagnosed]:
        problem = _misconception_problem(mid, refused)
        if problem:
            problems.append(f"{event}: {problem}")
    if event == "diagnosis" and (rec["status"] == "clear") != (rec["misconception"] is not None):
        problems.append("diagnosis: misconception must be set exactly when status is clear")
    if event == "turn" and rec["condition"] == "full" and rec["critic_passed"] is not True:
        problems.append("turn: a full-tutor message reached the student without passing the critic")
    return problems


def check_records(records: Iterable[tuple[str, dict]]) -> list[str]:
    """Problems across records: each a (where, record) pair, where is e.g. 'run-1.jsonl:12'."""
    held = held_back_rules()
    problems, started, hashes = [], set(), {}
    for where, rec in records:
        problems += [f"{where}: {p}" for p in check_record(rec, held)]
        if isinstance(rec.get("config_hash"), str):
            hashes.setdefault(rec["config_hash"], where)
        if rec.get("event") in LLM_EVENTS:
            continue
        key = (rec.get("run_id"), rec.get("session_id"))
        if rec.get("event") == "session_start":
            started.add(key)
        elif rec.get("event") in EVENTS and key not in started:
            problems.append(f"{where}: {rec['event']} before session_start of session {key[1]!r}")
    if len(hashes) > 1:
        seen = ", ".join(f"{h} (first at {w})" for h, w in hashes.items())
        problems.append(f"more than one config_hash, so shards ran on different settings: {seen}")
    return problems


def read_jsonl(paths: Iterable[str | Path]) -> Iterable[tuple[str, dict]]:
    for path in paths:
        path = Path(path)
        files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
        for f in files:
            with f.open(encoding="utf-8") as lines:
                for n, line in enumerate(lines, 1):
                    if line.strip():
                        yield f"{f}:{n}", json.loads(line)


def check_files(paths: Iterable[str | Path]) -> list[str]:
    return check_records(read_jsonl(paths))


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python -m matheasy.evaluation <file or dir> ...", file=sys.stderr)
        return 2
    problems = check_files(argv)
    for p in problems:
        print(p)
    print(f"{len(problems)} problem(s)" if problems else "logs OK")
    return 1 if problems else 0
