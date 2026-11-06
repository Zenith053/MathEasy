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
fixed items (runner.materials()). For several sessions, shards and the other
conditions, use the experiment runner: python -m matheasy.tutor.runner.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..diagnosis import diagnose
from ..evaluation import check_files, roster
from ..evaluation.metrics import accepted
from ..llm import load_config
from .__main__ import keyword_judge
from .critic import TutorCritic
from .orchestrator import Orchestrator
from .placeholders import TemplateHinter
from .runner import PROBLEMS_PER_SESSION, SLIP_ONLY, b2_diagnose, materials
from .session import make_problem, run_session
from .store import InMemoryStore
from .turnlog import REPO_ROOT, TurnLogger

OUT = REPO_ROOT / "runs" / "dev"


def plan():
    """(student, problems, test items) for every roster student with an item to work."""
    plans, skipped = [], []
    for student in roster(keyword_judge, slip_only=SLIP_ONLY):
        pool, tests = materials(student)
        if pool:
            plans.append((student, pool[:PROBLEMS_PER_SESSION], tests))
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
