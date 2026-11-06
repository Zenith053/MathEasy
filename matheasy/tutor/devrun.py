"""Offline development run (Role C): the full tutor and emulated B2 on every roster student.

    python -m matheasy.tutor.devrun                   # logs in runs/dev/
    python -m matheasy.evaluation.metrics runs/dev    # D's metrics on them

No LLM: the template hinter, the critic's code checks and the keyword judge,
as in python -m matheasy.tutor. The numbers show the pipeline works and how
the two conditions differ; they are not evidence for the hypotheses.

    full  B's diagnose() as it is today. When rules tie, the diagnosis is
          "ambiguous" and the tutor gives a general hint (no probe bank yet).
    b2    The same, but a tie commits to the shortlister's first candidate:
          B2's "take the top choice, no follow-ups", emulated here until B's
          mode="b2" lands in week 3.

Each student gets one session: up to 2 problems on which its rule gives a wrong
answer, with up to 4 such items as pre- and post-test. Slip-only students get
fixed items. The week-3 runner replaces this with isomorphic tests and shards.
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

from ..diagnosis import diagnose
from ..evaluation import SimStudent, check_files, roster
from ..evaluation.metrics import accepted
from ..llm import load_config
from .__main__ import keyword_judge
from .critic import TutorCritic
from .orchestrator import Orchestrator
from .placeholders import TemplateHinter
from .session import make_problem, run_session
from .store import InMemoryStore
from .turnlog import REPO_ROOT, TurnLogger

OUT = REPO_ROOT / "runs" / "dev"

# D's sample items (tests/test_students_week2.py): each rule's own example.
ITEMS = [
    "1/2 + 1/3", "2/5 × 3/5", "2/3 ÷ 4/5", "1/4 + 1/6", "3/4 - 1/2", "12/34", "3/4 > 5/8",
    "0.25 + 0.3", "0.3 × 0.2", "0.45 > 0.8", "25% of 80", "3.5 + 2.75",
    "2(x + 3)", "(a + b)^2", "2x + 4 = 10", "x + 3 = 7", "10 - 2 + 3", "12 ÷ 2 × 3", "2 + 3 × 4",
    "-3 + (-5)", "(-2)(-3)", "-5 - 6",
    "5x = 2x + 12", "(x + 3)^2", "2 × (3 + 4)", "2/3 = 6/x",
    "23 + 45", "37 + 25", "328 + 917", "8 + 12", "87 + 93", "47 + 38",
    "321 - 89", "403 - 168", "140 - 21", "403 - 208", "85 - 127", "302 - 7", "304 - 75",
    "85 - 23", "52 - 17", "62 - 37", "402 - 6",
    "68 × 46", "23 × 14", "23 × 4",
    "(x^2)^3", "(2x^3)^2", "2^3", "x^3 × x^5", "2^-2", "5^0",
    "(3 × 10^4) × (2 × 10^5)", "45000", "0.0032",
]
SLIP_ITEMS = ["1/2 + 1/3", "0.25 + 0.3", "x + 3 = 7", "2(x + 3)", "3/4 - 1/2"]
SLIP_ONLY = 3


def wrong_items(rule: str) -> list[str]:
    """The items on which a student holding `rule` follows it to a wrong answer."""
    probe = SimStudent("probe", rule, judge=lambda message, misconception: False)
    found = []
    for item in ITEMS:
        try:
            reply = probe.respond("q", item=item)
        except Exception:  # a rule that can't read this item
            continue
        if reply.followed_rule and not reply.correct:
            found.append(item)
    return found


def b2_diagnose(*args, **kwargs):
    """Emulated B2: never stay ambiguous; commit to the shortlister's first candidate."""
    d = diagnose(*args, **kwargs)
    if d.status != "ambiguous":
        return d
    shortlist = next(t for t in d.trace if t["stage"] == "shortlist")["candidates"]
    return dataclasses.replace(d, status="clear", misconception=shortlist[0][0])


def plan():
    """(student, problems, test items) for every roster student with an item to work."""
    plans, skipped = [], []
    for student in roster(keyword_judge, slip_only=SLIP_ONLY):
        if student.misconception_id is None:
            plans.append((student, SLIP_ITEMS[:3], SLIP_ITEMS[3:]))
            continue
        found = wrong_items(student.misconception_id.split(":", 1)[1])
        if found:
            plans.append((student, found[:2], found[:4]))
        else:
            skipped.append(student.student_id)
    return plans, skipped


def run(condition: str, config_hash: str) -> dict:
    log = TurnLogger(f"dev-{condition}", config_hash, OUT)
    tutor = Orchestrator(TemplateHinter(), TutorCritic(), InMemoryStore(), log, mode=condition,
                         diagnose_fn=diagnose if condition == "full" else b2_diagnose)
    plans, skipped = plan()
    first = {"right": 0, "wrong": 0, "ambiguous": 0, "unknown": 0, "none": 0}
    students = fixed = post = tests = 0
    for student, problems, test_items in plans:
        result = run_session(tutor, student, [make_problem(f"p{i}", q) for i, q in enumerate(problems)],
                             session_id=f"{condition}-{student.student_id}", condition=condition,
                             session_index=0, pre_items=test_items, post_items=test_items)
        if student.group != "has_rule":
            continue
        students += 1
        fixed += not student.holds_misconception
        post += result.post_correct
        tests += len(test_items)
        d = next((r for r in _diagnoses(log.path, f"{condition}-{student.student_id}")), None)
        if d is None:
            first["none"] += 1
        elif d["status"] == "clear":
            first["right" if d["misconception"] in accepted(student.misconception_id, "has_rule") else "wrong"] += 1
        else:
            first[d["status"]] += 1
    return {"log": log.path, "checker": check_files([log.path]), "students": students, "first": first,
            "fixed": fixed, "post": post, "tests": tests, "skipped": skipped}


def _diagnoses(path: Path, session_id: str):
    import json
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["event"] == "diagnosis" and r["session_id"] == session_id:
                yield r


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    config_hash = load_config().hash
    results = {c: run(c, config_hash) for c in ("full", "b2")}
    print("Students with a rule (has_rule group); pre-test was 0 for both.\n")
    print(f"{'':34}{'full':>8}{'b2':>8}")
    rows = [("students", lambda r: r["students"])]
    rows += [(f"first diagnosis: {k}", lambda r, k=k: r["first"][k]) for k in results["full"]["first"]]
    rows += [("misconception fixed by the end", lambda r: r["fixed"]),
             ("post-test correct", lambda r: f"{r['post']}/{r['tests']}")]
    for name, value in rows:
        print(f"{name:34}{value(results['full']):>8}{value(results['b2']):>8}")
    for c, r in results.items():
        print(f"\n{c}: {r['log']}  D's checker: {'logs OK' if not r['checker'] else r['checker'][:3]}")
    print(f"skipped (no sample item shows the rule): {', '.join(results['full']['skipped']) or 'none'}")
    print(f"\nD's metrics: python -m matheasy.evaluation.metrics {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
