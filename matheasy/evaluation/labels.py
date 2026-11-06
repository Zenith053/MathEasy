"""Human labelling (owner: D): blind samples for labellers, agreement, and checking judges.

Protocol: docs/labelling_protocol.md. In short:

    python -m matheasy.evaluation.labels sample runs/final/turns --out labels/   # make the sheets
    # each member fills in the label column of labels/<name>.csv, without the key
    python -m matheasy.evaluation.labels agree labels/                          # agreement between members
    python -m matheasy.evaluation.labels check labels/ --judge leak             # the leak code vs humans

What a task is. One yes/no question (judges.QUESTIONS) about one piece of tutor
output, with only the context needed to answer it. Nothing in a task says which
condition produced it: the condition, session and turn go to labels/key.jsonl,
which labellers do not open. Tasks are shuffled together across conditions.

Overlap. About OVERLAP of the tasks go to two labellers, so agreement can be
measured (Cohen's kappa). Everything else goes to one labeller, in turn.

Sampling (per condition and kind, seeded):
    leak            tutor messages with an open question; half from turns the leak code
                    flags, half from the rest, so there are enough positives to check it
    targets         hints to students who hold a misconception
    diagnosis_text  B0's stated diagnoses for students whose misconception it could know
    citation        hints that cite a textbook section
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from .judges import QUESTIONS
from .log_fields import read_jsonl
from .metrics import describe, leaking_turns, open_questions, sessions

LABELLERS = ("A", "B", "C", "D")
OVERLAP = 0.2
PER_KIND = {"leak": 40, "targets": 40, "diagnosis_text": 30, "citation": 20}  # per condition
JUDGE_KAPPA_MIN = 0.7  # a judge whose kappa with humans is lower is not used for a reported metric
FIELDS = ("task_id", "kind", "question", "context", "label", "note")


@dataclass(frozen=True)
class Task:
    task_id: str
    kind: str
    context: str        # what the labeller sees: no condition, no names of systems
    condition: str      # key only
    session_id: str     # key only
    turn: Optional[int]  # key only
    auto: Optional[bool]  # what the code says, where code decides (leak); key only
    inputs: tuple = ()    # key only: (name, text) pairs a judge is given, e.g. message, misconception

    def sheet_row(self) -> dict:
        return {"task_id": self.task_id, "kind": self.kind, "question": QUESTIONS[self.kind],
                "context": self.context, "label": "", "note": ""}

    def key_row(self) -> dict:
        row = asdict(self)
        row.pop("context")
        row["inputs"] = dict(self.inputs)
        return row


def _task_id(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:10]


# ---------------------------------------------------------------- making tasks

def candidate_tasks(records: Iterable[tuple[str, dict]]) -> dict[tuple[str, str], list[Task]]:
    """Every task that could be labelled, by (condition, kind). Adversarial sessions are included
    for leak only."""
    found: dict[tuple[str, str], list[Task]] = defaultdict(list)
    for s in sessions(records):
        planted = s.planted[0] if s.planted else None
        flagged = {t["turn"] for t in leaking_turns(s)}
        for t, unreached, _ in open_questions(s):
            if not unreached:
                continue
            open_text = "\n".join(f"Open question: {item}   (correct answer: {answer})"
                                  for item, answer in unreached.items())
            context = f"{open_text}\n\nTutor's message:\n{t['tutor_text']}"
            found[(s.condition, "leak")].append(Task(_task_id("leak", s.session_id, t["turn"]), "leak", context,
                                                     s.condition, s.session_id, t["turn"], t["turn"] in flagged,
                                                     (("message", t["tutor_text"]),)))
        if s.attack:
            continue
        if planted and s.group in ("has_rule", "known_as_text", "never_seen", "unchecked"):
            for h in s.hints:
                context = f"Planted misconception:\n{describe(planted)}\n\nTutor's message:\n{h['text']}"
                found[(s.condition, "targets")].append(Task(
                    _task_id("targets", s.session_id, h["turn"]), "targets", context, s.condition, s.session_id,
                    h["turn"], None, (("message", h["text"]), ("misconception", describe(planted)))))
        if planted and s.group in ("has_rule", "known_as_text"):
            for d in s.diagnoses:
                if d.get("stated_misconception"):
                    context = (f"Planted misconception:\n{describe(planted)}\n\n"
                               f"Tutor's stated diagnosis:\n{d['stated_misconception']}")
                    found[(s.condition, "diagnosis_text")].append(Task(
                        _task_id("diagnosis_text", s.session_id, d["turn"]), "diagnosis_text", context,
                        s.condition, s.session_id, d["turn"], None,
                        (("stated", d["stated_misconception"]), ("misconception", describe(planted)))))
        for h in s.hints:
            if h.get("citations"):
                context = f"Hint:\n{h['text']}\n\nCited section: {', '.join(h['citations'])}"
                found[(s.condition, "citation")].append(Task(_task_id("citation", s.session_id, h["turn"]),
                                                             "citation", context, s.condition, s.session_id,
                                                             h["turn"], None))
    return found


def sample_tasks(records: Iterable[tuple[str, dict]], *, per_kind: Optional[dict] = None, seed: int = 0) -> list[Task]:
    per_kind = {**PER_KIND, **(per_kind or {})}
    rng = random.Random(seed)
    chosen = []
    for (condition, kind), tasks in sorted(candidate_tasks(records).items()):
        n = per_kind.get(kind, 0)
        tasks = sorted(tasks, key=lambda t: t.task_id)
        if kind == "leak":
            flagged = [t for t in tasks if t.auto]
            clean = [t for t in tasks if not t.auto]
            take = min(len(flagged), n // 2)
            chosen += rng.sample(flagged, take) + rng.sample(clean, min(len(clean), n - take))
        else:
            chosen += rng.sample(tasks, min(len(tasks), n))
    rng.shuffle(chosen)
    return chosen


def assign(tasks: Sequence[Task], labellers: Sequence[str] = LABELLERS, *, overlap: float = OVERLAP,
           seed: int = 0) -> dict[str, list[Task]]:
    """Each task to one labeller in turn; the first `overlap` share also to the next labeller."""
    rng = random.Random(seed)
    order = list(tasks)
    rng.shuffle(order)
    doubled = int(round(overlap * len(order)))
    sheets: dict[str, list[Task]] = {name: [] for name in labellers}
    for i, task in enumerate(order):
        first = labellers[i % len(labellers)]
        sheets[first].append(task)
        if i < doubled:
            sheets[labellers[(i + 1) % len(labellers)]].append(task)
    for sheet in sheets.values():
        rng.shuffle(sheet)
    return sheets


def write_sheets(tasks: Sequence[Task], out: Path, labellers: Sequence[str] = LABELLERS, *,
                 overlap: float = OVERLAP, seed: int = 0) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, sheet in assign(tasks, labellers, overlap=overlap, seed=seed).items():
        path = out / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(t.sheet_row() for t in sheet)
        paths[name] = path
    with (out / "key.jsonl").open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t.key_row()) + "\n")
    return paths


# ---------------------------------------------------------------- reading labels back

_YES = {"y", "yes", "1", "true", "t"}
_NO = {"n", "no", "0", "false", "f"}


def read_labels(directory: Path) -> dict[str, dict[str, bool]]:
    """{task_id: {labeller: label}} from every <labeller>.csv; blank or unclear labels are skipped."""
    labels: dict[str, dict[str, bool]] = defaultdict(dict)
    for path in sorted(Path(directory).glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                value = (row.get("label") or "").strip().lower()
                if value in _YES or value in _NO:
                    labels[row["task_id"]][path.stem] = value in _YES
    return dict(labels)


def read_key(directory: Path) -> dict[str, dict]:
    with (Path(directory) / "key.jsonl").open(encoding="utf-8") as f:
        return {row["task_id"]: row for row in map(json.loads, f) if row}


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> Optional[float]:
    """Agreement beyond chance between two yes/no raters. None if undefined (fewer than 2
    items, or both raters gave one answer only and agree completely: kappa is 0/0)."""
    if len(a) != len(b) or len(a) < 2:
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def agreement(labels: dict[str, dict[str, bool]], key: dict[str, dict]) -> dict[str, dict]:
    """Per kind: tasks labelled twice, raw agreement and Cohen's kappa between the two labels."""
    pairs: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for task_id, by in labels.items():
        if len(by) >= 2 and task_id in key:
            x, y = list(by.values())[:2]
            pairs[key[task_id]["kind"]].append((x, y))
    return {kind: {"double_labelled": len(p), "agreement": sum(x == y for x, y in p) / len(p),
                   "kappa": cohen_kappa([x for x, _ in p], [y for _, y in p])}
            for kind, p in sorted(pairs.items())}


def human_label(by: dict[str, bool]) -> Optional[bool]:
    """The human answer for a task: the label, or the agreed label if labelled twice; None if they differ."""
    values = set(by.values())
    return values.pop() if len(values) == 1 else None


def check_judge(labels: dict[str, dict[str, bool]], key: dict[str, dict], kind: str,
                judge: Callable[[dict], Optional[bool]]) -> dict:
    """Compare a judge with the human labels on tasks of one kind. `judge(key_row)` returns
    the judge's answer for a task (None: no answer). Tasks where two humans disagree are left out."""
    rows = []
    for task_id, by in labels.items():
        row = key.get(task_id)
        human = human_label(by)
        if row is None or row["kind"] != kind or human is None:
            continue
        auto = judge(row)
        if auto is not None:
            rows.append((human, bool(auto)))
    if not rows:
        return {"kind": kind, "tasks": 0}
    h, j = [x for x, _ in rows], [y for _, y in rows]
    tp = sum(x and y for x, y in rows)
    kappa = cohen_kappa(h, j)
    return {
        "kind": kind,
        "tasks": len(rows),
        "accuracy": sum(x == y for x, y in rows) / len(rows),
        "kappa": kappa,
        "precision": tp / sum(j) if sum(j) else None,
        "recall": tp / sum(h) if sum(h) else None,
        "usable": kappa is not None and kappa >= JUDGE_KAPPA_MIN,
    }


def leak_code_judge(row: dict) -> Optional[bool]:
    """The leak metric's own decision for a task (stored in the key when the sheets were made)."""
    return row.get("auto")


def feedback_judge_check(judge) -> Callable[[dict], Optional[bool]]:
    """Adapts students.LLMFeedbackJudge (or any judge(message, misconception)) to check_judge()."""
    return lambda row: judge(row["inputs"]["message"], row["inputs"]["misconception"])


def text_judge_check(judge) -> Callable[[dict], Optional[bool]]:
    """Adapts judges.LLMTextJudge (or any text_judge(stated, description)) to check_judge()."""
    return lambda row: judge(row["inputs"]["stated"], row["inputs"]["misconception"])


# ---------------------------------------------------------------- command line

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m matheasy.evaluation.labels")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample", help="make blind labelling sheets from run logs")
    s.add_argument("paths", nargs="+")
    s.add_argument("--out", type=Path, default=Path("labels"))
    s.add_argument("--labellers", default=",".join(LABELLERS))
    s.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("agree", help="agreement between labellers on double-labelled tasks")
    a.add_argument("directory", type=Path)
    c = sub.add_parser("check", help="compare a judge with the human labels")
    c.add_argument("directory", type=Path)
    c.add_argument("--judge", choices=["leak", "targets", "diagnosis_text"], default="leak",
                   help="leak checks the leak code; targets and diagnosis_text call the LLM judges")
    args = parser.parse_args(argv)

    if args.command == "sample":
        tasks = sample_tasks(read_jsonl(args.paths), seed=args.seed)
        paths = write_sheets(tasks, args.out, args.labellers.split(","), seed=args.seed)
        counts = defaultdict(int)
        for t in tasks:
            counts[t.kind] += 1
        print(f"{len(tasks)} tasks ({', '.join(f'{k}: {n}' for k, n in sorted(counts.items()))})")
        for name, path in paths.items():
            print(f"  {name}: {path}")
        print(f"  key (do not give to labellers): {args.out / 'key.jsonl'}")
    elif args.command == "agree":
        for kind, row in agreement(read_labels(args.directory), read_key(args.directory)).items():
            print(kind, row)
    else:
        if args.judge == "leak":
            judge = leak_code_judge
        else:
            from ..llm import LLMClient, load_config
            from .judges import LLMTextJudge
            from .students import LLMFeedbackJudge
            session = LLMClient(load_config(), run_id="label-check").session(
                f"label-check-{args.judge}", "full", budget_tokens=None)
            judge = feedback_judge_check(LLMFeedbackJudge(session)) if args.judge == "targets" \
                else text_judge_check(LLMTextJudge(session))
        print(check_judge(read_labels(args.directory), read_key(args.directory), args.judge, judge))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Task", "agreement", "assign", "candidate_tasks", "check_judge", "cohen_kappa", "feedback_judge_check",
           "leak_code_judge", "read_key", "read_labels", "sample_tasks", "text_judge_check", "write_sheets"]
