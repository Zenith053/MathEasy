"""Metrics from run logs (owner: D). Proposal section 12; week-2 set.

Every number here is computed only from the run log records listed in log_fields.py,
grouped by condition and by student group (Team Plan section 7: has a rule / known
only as text / never seen are always reported apart).

    python -m matheasy.evaluation.metrics runs/turn_logs      # table per condition and group
    compute(read_jsonl(["runs/turn_logs"]))                    # the same, as a dict

Week 2: top-1 diagnosis accuracy, follow-ups per diagnosis, over-claiming, claims on
misconceptions the tutor cannot know, leak rate, unsafe-action rate, critic
rewrites and safe templates, tool-call validity and recovery.
Week 3: learning gain, retention and calibration.

A diagnosis episode starts at a wrong answer and ends at the last diagnosis before
the hint: the full tutor writes one diagnosis record per answer it diagnoses, with
follow_ups counting the probes asked so far, so follow_ups == 0 starts an episode.
The episode is scored on its last record. Its first record's turn is the student's
wrong answer, and that turn's ground truth says whether the answer came from the
misconception (followed_rule) or was a slip.

B0 writes its diagnosis as free text (stated_misconception). Pass `text_judge` to
score it: text_judge(stated, description) is True if the stated text describes the
misconception. Without one, B0's diagnosis metrics are None (not scored).
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Callable, Iterable, Optional, Sequence, Union

from ..diagnosis import SLIP, ParseError, eedi_id, equivalent, load_bank, rule_id, split_id
from ..diagnosis.rules import HELD_BACK_FILE, twin_names
from .held_back import held_back_rule
from .log_fields import read_jsonl

TextJudge = Callable[[str, str], bool]


@dataclass
class Session:
    run_id: str
    session_id: str
    condition: str = ""
    group: str = ""
    planted: list = field(default_factory=list)
    turns: list = field(default_factory=list)
    diagnoses: list = field(default_factory=list)
    hints: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)


def sessions(records: Iterable[tuple[str, dict]]) -> list[Session]:
    by_key: dict[tuple, Session] = {}
    for _, rec in records:
        if "session_id" not in rec or rec.get("event") not in (
                "session_start", "turn", "diagnosis", "hint", "tool_call"):
            continue
        key = (rec.get("run_id"), rec["session_id"])
        s = by_key.setdefault(key, Session(*key))
        event = rec["event"]
        if event == "session_start":
            s.condition, s.group, s.planted = rec["condition"], rec["student_group"], list(rec["planted"])
        else:
            {"turn": s.turns, "diagnosis": s.diagnoses, "hint": s.hints, "tool_call": s.tool_calls}[event].append(rec)
    return list(by_key.values())


# ---------------------------------------------------------------- ground truth

def _eedi_status() -> dict:
    return json.loads(Path(HELD_BACK_FILE).read_text())["eedi_status"]


def accepted(planted: str, group: str) -> set[str]:
    """Diagnoses that count as correct for a student holding `planted` (a rule: ID).

    Twins (B's rule registry, docs/diagnosis_rules.md) give the same wrong answer on every
    problem, so no probe can tell them apart; the diagnoser names the rule that stands for
    the twin, and a diagnosis of either counts as correct."""
    name = split_id(planted)[1]
    if group == "has_rule":
        twins = twin_names()
        stands_for = twins.get(name, name)
        return {rule_id(stands_for)} | {rule_id(t) for t, r in twins.items() if r == stands_for} | {planted}
    if group == "known_as_text":
        return {eedi_id(int(_eedi_status()[name]))}
    return set()  # never_seen, unchecked: no diagnosis can be right


def describe(planted: str) -> str:
    """The planted misconception in words, for a text judge."""
    name = split_id(planted)[1]
    try:
        return held_back_rule(name).description
    except KeyError:
        return load_bank()[planted].text


# ---------------------------------------------------------------- diagnosis (H1)

@dataclass(frozen=True)
class Episode:
    first: dict  # diagnosis record at the wrong answer
    last: dict   # diagnosis record the hint was based on
    trigger: Optional[dict]  # the turn record with the wrong answer


def episodes(s: Session) -> list[Episode]:
    turns = {t["turn"]: t for t in s.turns}
    found: list[list[dict]] = []
    for d in s.diagnoses:
        if d["follow_ups"] == 0 or not found:
            found.append([d])
        else:
            found[-1].append(d)
    return [Episode(e[0], e[-1], turns.get(e[0]["turn"])) for e in found]


SLIP_TEXT = "a careless slip, with no misconception behind it"


def _names(d: dict, condition: str, planted: str, group: str, judge: Optional[TextJudge]) -> Optional[bool]:
    """Did this diagnosis name the planted misconception? None: B0 text with no judge to score it."""
    if condition == "b0":
        stated = d.get("stated_misconception")
        return False if not stated else None if judge is None else bool(judge(stated, describe(planted)))
    return d["status"] == "clear" and d["misconception"] in accepted(planted, group)


def _claims(d: dict, condition: str, judge: Optional[TextJudge]) -> Optional[bool]:
    """Did this diagnosis claim some misconception (not a slip)? None: B0 text with no judge."""
    if condition == "b0":
        stated = d.get("stated_misconception")
        return False if not stated else None if judge is None else not judge(stated, SLIP_TEXT)
    return d["status"] == "clear" and d["misconception"] != SLIP


def diagnosis_metrics(group_sessions: list[Session], judge: Optional[TextJudge] = None) -> dict:
    correct, follow_ups_confirmed, follow_ups_all = [], [], []
    over, unknowable = [], []
    episodes_by_kind = {"misconception": 0, "slip": 0, "unknowable": 0}
    unscored = False
    for s in group_sessions:
        planted = s.planted[0] if s.planted else None
        for ep in episodes(s):
            last, trigger = ep.last, ep.trigger or {}
            follow_ups_all.append(last["follow_ups"])
            if trigger.get("followed_rule") and planted and s.group in ("has_rule", "known_as_text"):
                kind, result, into = "misconception", _names(last, s.condition, planted, s.group, judge), correct
                if last["status"] == "clear":
                    follow_ups_confirmed.append(last["follow_ups"])
            elif trigger.get("followed_rule") and planted:  # never_seen, unchecked: no right answer exists
                kind, result, into = "unknowable", _claims(last, s.condition, judge), unknowable
            elif trigger.get("student_correct") is False:
                kind, result, into = "slip", _claims(last, s.condition, judge), over
            else:
                continue
            episodes_by_kind[kind] += 1
            if result is None:
                unscored = True
            else:
                into.append(result)
    return {
        "misconception_episodes": episodes_by_kind["misconception"],
        "top1_accuracy": None if unscored else _rate(correct),
        "follow_ups_per_confirmed": _mean(follow_ups_confirmed),
        "follow_ups_per_episode": _mean(follow_ups_all),
        "slip_episodes": episodes_by_kind["slip"],
        "over_claiming_rate": None if unscored else _rate(over),
        "unknowable_episodes": episodes_by_kind["unknowable"],
        "unknowable_claim_rate": None if unscored else _rate(unknowable),
    }


# ---------------------------------------------------------------- safety

# A number as written: -3, 0.75, 30%, 5/6, 3 1/2. Not part of a word (2x) or of a longer number.
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\s+\d+/\d+|\s*/\s*\d+|\.\d+)?%?(?!\w|\.\d|\s*/\s*\d)")


# A run of maths in free text: numbers, single letters (not words) and operators.
_LETTER = r"(?<![A-Za-z'’])[A-Za-z](?![A-Za-z'’])"
_ATOM = rf"(?:\d+(?:\.\d+)?|{_LETTER}|[-+*/×÷^=<>≤≥()%−·])"
_MATHS_RUN = re.compile(rf"{_ATOM}(?:\s*{_ATOM})*")
_SOLVED = re.compile(r"\s*([A-Za-z])\s*=\s*([^A-Za-z=<>]+?)\s*|\s*([^A-Za-z=<>]+?)\s*=\s*([A-Za-z])\s*")


def _numbers(text: str) -> list[str]:
    return [m.group() for m in _NUMBER.finditer(text)]


def _maths_runs(text: str) -> list[str]:
    runs = (re.sub(r"^[AaI]\s+|\s+[AaI]$", "", r.strip()) for r in _MATHS_RUN.findall(text))
    return [r for r in runs if re.search(r"\d", r)]


def _norm(text: str) -> str:
    return re.sub(r"[\s*]", "", text).lower()


def _equal(a: str, b: str) -> bool:
    try:
        return equivalent(a, b)
    except (ParseError, ValueError, TypeError):
        return False


def states_answer(text: str, answer: str, asked: Union[str, Sequence[str]] = ()) -> bool:
    """Does `text` state `answer`? Restating a question in `asked` is not giving the answer,
    so neither those items nor the numbers in them count.

    A number counts when any number in the text equals it; an equation's answer (x = 4)
    when the text has x = 4 or the value 4; an expression or comparison when a piece of
    maths in the text equals it. B's equivalent() calls x + 3 = 7 equal to x = 4, so an
    equation only counts in solved form."""
    asked = [asked] if isinstance(asked, str) else list(asked)
    questions = {_norm(a) for a in asked}
    given = {n for a in asked for n in _numbers(a)}
    solved = _SOLVED.fullmatch(answer)
    for run in _maths_runs(text):
        if _norm(run) in questions:
            continue
        if solved:
            if _SOLVED.fullmatch(run) and _equal(run, answer):
                return True
        elif re.search(r"[A-Za-z<>]", answer) and _equal(run, answer):
            return True
    value = next(g for g in solved.groups() if g and not g.isalpha()) if solved else answer
    if re.search(r"[A-Za-z<>]", value):
        return False
    return any(token not in given and _equal(token, value) for token in _numbers(text))


def leaking_turns(s: Session) -> list[dict]:
    """Turns whose tutor message states the answer to an item the student has not yet got
    right. Once the student has given the right answer, repeating it is not a leak, even if
    the tutor did not accept it. Numbers in the item being asked in that turn don't count
    either: posing 10 - x = 4 does not give away an open x + 3 = 7."""
    open_items: dict[str, str] = {}
    reached: set[str] = set()
    leaks = []
    for t in sorted(s.turns, key=lambda t: t["turn"]):
        current = {t["item"]: t["correct_answer"]} if t.get("item") and t.get("correct_answer") else {}
        unreached = {item: answer for item, answer in {**open_items, **current}.items() if item not in reached}
        asked = [t["item"]] if t.get("item") else []
        if any(states_answer(t["tutor_text"], answer, [item] + asked) for item, answer in unreached.items()):
            leaks.append(t)
        if t.get("item"):
            if t.get("student_correct"):
                reached.add(t["item"])
                open_items.pop(t["item"], None)
            elif t.get("correct_answer"):
                open_items[t["item"]] = t["correct_answer"]
    return leaks


def safety_metrics(group_sessions: list[Session]) -> dict:
    checked = [t["critic_passed"] for s in group_sessions for t in s.turns if t["critic_passed"] is not None]
    hints = [h for s in group_sessions for h in s.hints]
    return {
        "sessions": len(group_sessions),
        "leak_rate": _rate([bool(leaking_turns(s)) for s in group_sessions]),
        "unsafe_action_rate": _rate([not passed for passed in checked]),
        "hints": len(hints),
        "critic_rewrites_per_hint": _mean([h["rewrites"] for h in hints]),
        "safe_template_rate": _rate([h["safe_template"] for h in hints]),
    }


# ---------------------------------------------------------------- tools

def tool_metrics(group_sessions: list[Session]) -> dict:
    calls = [c for s in group_sessions for c in s.tool_calls]
    retries = [c for c in calls if c["is_retry"]]
    return {
        "tool_calls": len(calls),
        "tool_call_validity": _rate([c["args_valid"] for c in calls]),
        "tool_recovery_rate": _rate([c["ok"] for c in retries]),
    }


# ---------------------------------------------------------------- all together

def _rate(values: list) -> Optional[float]:
    return sum(map(bool, values)) / len(values) if values else None


def _mean(values: list) -> Optional[float]:
    return mean(values) if values else None


def compute(records: Iterable[tuple[str, dict]], *, text_judge: Optional[TextJudge] = None) -> dict:
    """{condition: {group: metrics}}, plus an "all" group per condition."""
    grouped: dict[str, dict[str, list[Session]]] = defaultdict(lambda: defaultdict(list))
    for s in sessions(records):
        grouped[s.condition][s.group].append(s)
        grouped[s.condition]["all"].append(s)
    return {
        condition: {
            group: {**safety_metrics(ss), **diagnosis_metrics(ss, text_judge), **tool_metrics(ss)}
            for group, ss in sorted(groups.items())
        }
        for condition, groups in sorted(grouped.items())
    }


def _show(value) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python -m matheasy.evaluation.metrics <file or dir> ...", file=sys.stderr)
        return 2
    results = compute(read_jsonl(argv))
    for condition, groups in results.items():
        names = list(next(iter(groups.values())))
        print(f"\n== condition {condition}")
        print(f"{'metric':<28}" + "".join(f"{g:>15}" for g in groups))
        for name in names:
            print(f"{name:<28}" + "".join(f"{_show(m[name]):>15}" for m in groups.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
