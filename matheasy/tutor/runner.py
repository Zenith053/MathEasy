"""The experiment runner (Role C): any condition, on any shard of students, over several sessions.

    python -m matheasy.tutor.runner trial --conditions full,b2 --sessions 2            # offline
    python -m matheasy.tutor.runner final --llm --shard 0/4 --conditions full,b0,b1,b2  # one member's shard
    python -m matheasy.evaluation runs/final                                            # D's checker
    python -m matheasy.evaluation.metrics runs/final/turns                              # D's metrics

Students are split by student, never by condition (team plan, section 7): shard
i/n takes every n-th student of D's roster and every n-th of D's adversarial
students, and runs them under every condition given. Each condition has its own
copy of each student and its own memory. Students relapse between sessions
(--relapse, D's suggested 0.3), so retention is measured, not given.

    full  the tutor: B's diagnose(), the hinter and critic, memory and the planner
    b2    the same, with one-shot diagnosis: B's mode="b2", no probes (see b2_diagnose)
    b0    one prompted tutor (needs --llm)
    b1    one tool-using agent with the same tools, no memory (needs --llm)

Offline (no --llm) the hinter is the template hinter, the critic runs its code
checks only, and students judge feedback by keyword: fast and free, for
checking the pipeline, not for results.

Everything goes under runs/<run>/: turns/ (run logs), llm/ (LLM call logs),
students/ and memory/ (state between sessions). Each is saved when a session
finishes, so a stopped run carries on where it stopped: run the same command
again. Records of a session that never finished are dropped from its log first.

Each student works D's sample items on which its rule gives a wrong answer (the
pool; slip-only students get fixed items), and its pre- and post-tests are D's
prepost_forms(): the same ideas with new numbers, never a pool item or a probe.
Session k's pre-test is also the retention check for session k - 1. See materials().
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
from pathlib import Path
from typing import Callable, Optional, Sequence

from ..diagnosis import EediRanker, diagnose, load_ranker
from ..evaluation import (
    SAMPLE_ITEMS, SLIP_ITEMS, LLMAttackLine, LLMFeedbackJudge, SimStudent, TestForm, adversarial_roster,
    check_files, prepost_forms, roster,
)
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
SLIP_ONLY = 3

ITEMS = SAMPLE_ITEMS  # D's sample items: each rule's own example
RELAPSE_RATE = 0.3  # D's suggestion for multi-session runs
PER_ATTACK = 6  # adversarial students per attack technique


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


def materials(student: SimStudent, sessions: int = 1) -> tuple[list[str], list[TestForm]]:
    """(problem pool, one test form per session) for a student; an empty pool means
    nothing shows its rule. D's forms never match a pool item, a probe or each other."""
    if student.misconception_id is None:
        pool = list(SLIP_ITEMS[:3])
    else:
        pool = wrong_items(student.misconception_id.split(":", 1)[1])
    if not pool:
        return [], []
    try:
        forms = prepost_forms(student.misconception_id, sessions=sessions, student_id=student.student_id,
                              avoid=pool)
    except ValueError:  # no sample item shows the misconception
        return [], []
    return pool, forms


def b2_diagnose(*args, **kwargs):
    """B2 takes the top choice with no follow-ups. B's mode="b2" skips the probes but
    leaves tied rules ambiguous, so this commits to the shortlister's first candidate."""
    d = diagnose(*args, **kwargs)
    if d.status != "ambiguous":
        return d
    shortlist = next(t for t in d.trace if t["stage"] == "shortlist")["candidates"]
    return dataclasses.replace(d, status="clear", misconception=shortlist[0][0])


def make_tutor(condition: str, log: TurnLogger, store, *, llm: bool, retrieve: Optional[Callable],
               eedi_ranker: Optional[EediRanker] = None):
    if condition == "b0":
        return B0Tutor(log)
    if condition == "b1":
        return B1Agent(log, ToolBox(standard_tools(retrieve), log))
    hinter = LLMHinter(retrieve) if llm else TemplateHinter()
    diagnose_fn = diagnose if condition == "full" else b2_diagnose
    if eedi_ranker is not None:  # B's Eedi link: a step no rule explains may get an eedi: misconception
        diagnose_fn = functools.partial(diagnose_fn, eedi_ranker=eedi_ranker)
    return Orchestrator(hinter, TutorCritic(), store, log, mode=condition, diagnose_fn=diagnose_fn)


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
                  client: Optional[LLMClient], config_hash: str, retrieve: Optional[Callable],
                  eedi_ranker: Optional[EediRanker] = None) -> dict:
    log = TurnLogger(f"{condition}-shard{shard}", config_hash, out / "turns")
    done = finished_sessions(log.path)
    store = JsonStudentStore(out / "memory" / f"{condition}-shard{shard}.json", autosave=False) \
        if condition in WITH_MEMORY else NoMemoryStore()
    tutor = make_tutor(condition, log, store, llm=client is not None, retrieve=retrieve, eedi_ranker=eedi_ranker)
    state_dir = out / "students" / condition
    ran = skipped = 0
    for fresh in students:
        pool, forms = materials(fresh, sessions)
        if not pool:
            continue
        saved = (state_dir / f"{fresh.student_id}.json").exists()
        student = type(fresh).load(state_dir, fresh.student_id, judge=keyword_judge) if saved else fresh
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
            if client and hasattr(student, "attack_line"):  # an adversarial student: the LLM writes its attacks
                student.attack_line = LLMAttackLine(client.session(session_id, condition, budget_tokens=None))
            run_session(tutor, student, [make_problem(f"p{i}", q) for i, q in enumerate(problems)],
                        session_id=session_id, condition=condition, session_index=index,
                        pre_items=forms[index].pre, post_items=forms[index].post, llm_session=tutor_llm,
                        rechecks=rechecks)
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
    parser.add_argument("--per-attack", type=int, default=PER_ATTACK,
                        help="adversarial students per attack technique (0 for none)")
    parser.add_argument("--relapse", type=float, default=RELAPSE_RATE,
                        help="chance a fixed misconception comes back at a later session")
    parser.add_argument("--llm", action="store_true", help="LLM hinter, critic and judge (needs GEMINI_API_KEY)")
    parser.add_argument("--eedi", action="store_true",
                        help="name Eedi misconceptions with A's shortlister (needs --llm, a GPU and data/)")
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
    eedi_ranker = None
    if args.eedi:
        if not args.llm:
            parser.error("--eedi needs --llm: the LLM fallback picks the Eedi misconception")
        eedi_ranker = load_ranker()
        if eedi_ranker is None:
            parser.error("--eedi: the shortlister did not load (needs torch, a GPU and data/shortlister/adapter-3B-v2)")
    # The judge is replaced per session. --students keeps runs small: the first N of each kind.
    students = roster(keyword_judge, slip_only=SLIP_ONLY, relapse_rate=args.relapse)[i::n][:args.students]
    if args.per_attack:
        students += adversarial_roster(keyword_judge, per_attack=args.per_attack)[i::n][:args.students]

    for condition in conditions:
        # A fresh copy of each student per condition: no condition sees another's effect on it.
        copies = [type(s).from_state(s.to_state(), judge=keyword_judge) for s in students]
        result = run_condition(condition, copies, out, sessions=args.sessions, shard=shard, client=client,
                               config_hash=config.hash, retrieve=retrieve, eedi_ranker=eedi_ranker)
        problems = check_files([result["log"]])
        print(f"{condition}: {result['ran']} sessions run, {result['skipped']} already done; "
              f"{_shown(result['log'])}  D's checker: {'logs OK' if not problems else problems[:3]}")
    if args.llm:
        print(f"LLM calls: {_shown(client.log_path)}")


def _shown(path: Path) -> Path:
    return path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path


if __name__ == "__main__":
    main()
