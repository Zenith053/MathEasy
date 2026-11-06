"""The experiment runner (Role C): any condition, on any shard of students, over several sessions.

    python -m matheasy.tutor.runner trial --conditions full,b2 --sessions 2            # offline
    python -m matheasy.tutor.runner final --llm --shard 0/4 --conditions full,b0,b1,b2  # one member's shard
    python -m matheasy.evaluation runs/final                                            # D's checker
    python -m matheasy.evaluation.metrics runs/final/turns                              # D's metrics

Students are split by student, never by condition (team plan, section 7): shard
i/n takes every n-th student of D's roster, and runs it under every condition
given. Each condition has its own copy of each student and its own memory.

    full  the tutor: B's diagnose(), the hinter and critic, memory and the planner
    b2    the same, with one-shot diagnosis (emulated until B's mode="b2" lands; see b2_diagnose)
    b0    one prompted tutor (needs --llm)
    b1    one tool-using agent with the same tools, no memory (needs --llm)

Offline (no --llm) the hinter is the template hinter, the critic runs its code
checks only, and students judge feedback by keyword: fast and free, for
checking the pipeline, not for results.

Everything goes under runs/<run>/: turns/ (run logs), llm/ (LLM call logs),
students/ and memory/ (state between sessions). Each is saved when a session
finishes, so a stopped run carries on where it stopped: run the same command
again. Records of a session that never finished are dropped from its log first.

Until D's pre/post tests land, each student works the sample items on which its
rule gives a wrong answer, and up to TEST_ITEMS of them are the pre- and
post-test (slip-only students get fixed items); see materials().
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Callable, Optional, Sequence

from ..diagnosis import diagnose
from ..evaluation import LLMFeedbackJudge, SimStudent, check_files, roster
from ..llm import LLMClient, load_config
from . import planner
from .__main__ import keyword_judge, textbook_retrieve
from .baselines import B0Tutor, B1Agent
from .critic import TutorCritic
from .hinter import LLMHinter
from .orchestrator import Orchestrator
from .placeholders import TemplateHinter
from .session import make_problem, run_session
from .store import JsonStudentStore, NoMemoryStore
from .tools import ToolBox, standard_tools
from .turnlog import REPO_ROOT, TurnLogger

RUNS = REPO_ROOT / "runs"
CONDITIONS = ("full", "b2", "b0", "b1")
NEEDS_LLM = ("b0", "b1")
WITH_MEMORY = ("full", "b2")  # B0 and B1 have no memory, so no planner

PROBLEMS_PER_SESSION = 2
TEST_ITEMS = 4
SLIP_ONLY = 3

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


def materials(student: SimStudent) -> tuple[list[str], list[str]]:
    """(problem pool, test items) for a student; an empty pool means nothing shows its rule."""
    if student.misconception_id is None:
        return SLIP_ITEMS[:3], SLIP_ITEMS[3:]
    found = wrong_items(student.misconception_id.split(":", 1)[1])
    return found, found[:TEST_ITEMS]


def b2_diagnose(*args, **kwargs):
    """Emulated B2: never stay ambiguous; commit to the shortlister's first candidate.
    A no-op once B's diagnose(mode="b2") returns clear diagnoses itself."""
    d = diagnose(*args, **kwargs)
    if d.status != "ambiguous":
        return d
    shortlist = next(t for t in d.trace if t["stage"] == "shortlist")["candidates"]
    return dataclasses.replace(d, status="clear", misconception=shortlist[0][0])


def make_tutor(condition: str, log: TurnLogger, store, *, llm: bool, retrieve: Optional[Callable]):
    if condition == "b0":
        return B0Tutor(log)
    if condition == "b1":
        return B1Agent(log, ToolBox(standard_tools(retrieve), log))
    hinter = LLMHinter(retrieve) if llm else TemplateHinter()
    return Orchestrator(hinter, TutorCritic(), store, log, mode=condition,
                        diagnose_fn=diagnose if condition == "full" else b2_diagnose)


def finished_sessions(path: Path) -> set[str]:
    """Session IDs that reached session_end. Drops the records of any that did not."""
    if not path.exists():
        return set()
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    done = {r["session_id"] for r in records if r["event"] == "session_end"}
    kept = [r for r in records if r["session_id"] in done]
    if len(kept) < len(records):
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    return done


def run_condition(condition: str, students: Sequence[SimStudent], out: Path, *, sessions: int, shard: str,
                  client: Optional[LLMClient], config_hash: str, retrieve: Optional[Callable]) -> dict:
    log = TurnLogger(f"{condition}-shard{shard}", config_hash, out / "turns")
    done = finished_sessions(log.path)
    store = JsonStudentStore(out / "memory" / f"{condition}-shard{shard}.json", autosave=False) \
        if condition in WITH_MEMORY else NoMemoryStore()
    tutor = make_tutor(condition, log, store, llm=client is not None, retrieve=retrieve)
    state_dir = out / "students" / condition
    ran = skipped = 0
    for fresh in students:
        pool, tests = materials(fresh)
        if not pool:
            continue
        saved = (state_dir / f"{fresh.student_id}.json").exists()
        student = SimStudent.load(state_dir, fresh.student_id, judge=keyword_judge) if saved else fresh
        for index in range(sessions):
            session_id = f"{condition}-{student.student_id}-s{index}"
            if session_id in done:
                skipped += 1
                continue
            if condition in WITH_MEMORY:
                problems, rechecks = planner.plan(store.load(student.student_id), index, pool,
                                                  PROBLEMS_PER_SESSION)
            else:
                problems, rechecks = pool[:PROBLEMS_PER_SESSION], []
            tutor_llm = client.session(session_id, condition) if client else None
            student.judge = LLMFeedbackJudge(client.session(session_id, condition, budget_tokens=None)) \
                if client else keyword_judge
            run_session(tutor, student, [make_problem(f"p{i}", q) for i, q in enumerate(problems)],
                        session_id=session_id, condition=condition, session_index=index,
                        pre_items=tests, post_items=tests, llm_session=tutor_llm, rechecks=rechecks)
            student.save(state_dir)
            if isinstance(store, JsonStudentStore):
                store.save()
            ran += 1
    return {"log": log.path, "ran": ran, "skipped": skipped}


def parse_shard(text: str) -> tuple[int, int]:
    i, n = (int(x) for x in text.split("/"))
    if not 0 <= i < n:
        raise argparse.ArgumentTypeError(f"shard must be i/n with 0 <= i < n, got {text}")
    return i, n


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m matheasy.tutor.runner", description=__doc__.split("\n")[0])
    parser.add_argument("run", help="run name: everything goes under runs/<run>/")
    parser.add_argument("--conditions", default="full,b2", help=f"comma-separated, from {','.join(CONDITIONS)}")
    parser.add_argument("--shard", type=parse_shard, default=(0, 1), help="i/n: every n-th student from the i-th")
    parser.add_argument("--sessions", type=int, default=2, help="sessions per student")
    parser.add_argument("--students", type=int, default=None, help="only the first N students of the shard")
    parser.add_argument("--llm", action="store_true", help="LLM hinter, critic and judge (needs GEMINI_API_KEY)")
    args = parser.parse_args(argv)

    conditions = args.conditions.split(",")
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        parser.error(f"unknown condition(s): {', '.join(unknown)}")
    if not args.llm and set(conditions) & set(NEEDS_LLM):
        parser.error(f"{', '.join(sorted(set(conditions) & set(NEEDS_LLM)))} need --llm")

    i, n = args.shard
    shard = f"{i}of{n}"
    out = RUNS / args.run
    config = dataclasses.replace(load_config(), log_dir=out / "llm")
    client = LLMClient(config, run_id=f"shard{shard}") if args.llm else None
    retrieve = textbook_retrieve() if args.llm else None
    students = roster(keyword_judge, slip_only=SLIP_ONLY)[i::n][:args.students]  # judge replaced per session

    for condition in conditions:
        # A fresh copy of each student per condition: no condition sees another's effect on it.
        copies = [SimStudent.from_state(s.to_state(), judge=keyword_judge) for s in students]
        result = run_condition(condition, copies, out, sessions=args.sessions, shard=shard, client=client,
                               config_hash=config.hash, retrieve=retrieve)
        problems = check_files([result["log"]])
        print(f"{condition}: {result['ran']} sessions run, {result['skipped']} already done; "
              f"{_shown(result['log'])}  D's checker: {'logs OK' if not problems else problems[:3]}")
    if args.llm:
        print(f"LLM calls: {_shown(client.log_path)}")


def _shown(path: Path) -> Path:
    return path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path


if __name__ == "__main__":
    main()
