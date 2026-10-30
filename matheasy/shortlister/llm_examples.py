"""
LLM-written training examples for misconceptions with no training questions (Role A, run 3).

31% of val/test wrong answers have a misconception that never appears in the
training split. For each such misconception, the backbone model (C's client,
config/llm.yaml) writes one wrong step a student holding it would make. The step
goes through build_query(), so it looks like a real training query. Misconceptions
outside the tutor's topics come back as "other" and are skipped.

Misconceptions are sent BATCH at a time and calls are spaced to stay under the
free-tier limit (15 requests a minute). Only misconception names are sent to the
model, never questions from val or test. Output is Eedi-derived, so it stays in
data/ (git-ignored) and is never uploaded.

Usage: python -m matheasy.shortlister.llm_examples [--limit N] [--fake]
  -> data/eedi/llm_examples.csv  (MisconceptionId, Topic, Query); reruns skip IDs already done
"""

import argparse
import csv
import time
from pathlib import Path

import pandas as pd

from .query import build_query

DATA = Path(__file__).resolve().parents[2] / "data" / "eedi"
OUT = DATA / "llm_examples.csv"
TOPICS = ["fractions", "decimals", "expressions"]
BATCH = 20        # misconceptions per call
PAUSE_S = 4.5     # between calls: 60 / 4.5 < 15 requests a minute
WAIT_S = 60       # after a failed call (usually the rate limit), then try again
MAX_TOKENS = 4096  # 20 examples do not fit in the config's 1024 output tokens

SYSTEM = """You write training data for a maths tutor. For EACH numbered misconception, write ONE
short calculation step that a 11-14 year old student holding exactly this misconception would get
wrong. Use plain maths text such as 1/2 + 1/3, 0.25, 3(x + 2). topic is one of fractions, decimals,
expressions (algebraic expressions and linear equations); use "other" if the misconception is about
something else (geometry, statistics, graphs, word problems, factorising quadratics, quadratic
equations, ...). For "other", leave before, student and correct empty. Return one item per
misconception, with its number as id."""

SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "topic": {"type": "string", "enum": TOPICS + ["other"]},
            "before": {"type": "string", "description": "the expression or question before the step"},
            "student": {"type": "string", "description": "the student's wrong answer"},
            "correct": {"type": "string", "description": "the correct answer"},
        },
        "required": ["id", "topic", "before", "student", "correct"],
    }}},
    "required": ["items"],
}


def targets(queries, names):
    """Misconceptions with no training question, in ID order."""
    seen = set(queries[queries.Split == "train"].MisconceptionId)
    return names[~names.MisconceptionId.isin(seen)].sort_values("MisconceptionId")


def write_examples(session, batch):
    """{misconception ID: {"Topic", "Query"} or None} for a batch of (ID, name) pairs.

    None means outside the tutor's topics; IDs the model left out are missing,
    so a rerun asks for them again.
    """
    numbered = "\n".join(f"{i}. {name}" for i, name in batch)
    out = session.call("shortlister_data", [{"role": "user", "content": numbered}],
                       system=SYSTEM, json_schema=SCHEMA, max_output_tokens=MAX_TOKENS).parsed or {}
    asked = {i for i, _ in batch}
    examples = {}
    for item in out.get("items", []):
        if item.get("id") not in asked:
            continue
        usable = item.get("topic") in TOPICS and all(item.get(f) for f in ("before", "student", "correct"))
        # A "wrong" answer identical to the correct one teaches nothing.
        if usable and item["student"].replace(" ", "") != item["correct"].replace(" ", ""):
            examples[item["id"]] = {"Topic": item["topic"], "Query": build_query(
                item["topic"], item["before"], item["student"], item["correct"])}
        else:
            examples[item["id"]] = None
    return examples


def main(limit=None, fake=False):
    from matheasy.llm import FakeBackend, LLMCallError, LLMClient, LLMOutputError

    queries = pd.read_csv(DATA / "queries.csv")
    names = pd.read_csv(DATA / "raw" / "misconception_mapping.csv")
    todo = targets(queries, names)
    done = set(pd.read_csv(OUT).MisconceptionId) if OUT.exists() else set()
    todo = todo[~todo.MisconceptionId.isin(done)].head(limit)
    print(f"{len(done)} done, {len(todo)} to write")

    fake_reply = lambda req: '{"items": [{"id": %s, "topic": "fractions", "before": "1/2 + 1/3", ' \
                             '"student": "2/5", "correct": "5/6"}]}' % req.messages[0].content.split(".")[0]
    llm = LLMClient(backend=FakeBackend(fake_reply)) if fake else LLMClient()
    session = llm.session("shortlister-llm-examples", condition="data", budget_tokens=None)
    pairs = [(int(r.MisconceptionId), r.MisconceptionName.strip()) for r in todo.itertuples()]
    kept = written = 0
    with OUT.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["MisconceptionId", "Topic", "Query"])
        if not done:
            w.writeheader()
        for start in range(0, len(pairs), BATCH):
            try:
                examples = write_examples(session, pairs[start:start + BATCH])
            except LLMCallError as e:
                print(f"call failed ({str(e)[:80]}...), waiting {WAIT_S}s; rerun to fill gaps")
                time.sleep(WAIT_S)
                continue
            except LLMOutputError as e:
                print(f"bad reply ({str(e)[:80]}...), skipped; rerun to fill gaps")
                continue
            for mid, ex in examples.items():
                # "other" is written too (empty Query), so reruns do not ask again.
                w.writerow({"MisconceptionId": mid, **(ex or {"Topic": "other", "Query": ""})})
                kept += ex is not None
                written += 1
            f.flush()
            print(f"{start + BATCH}/{len(pairs)}: kept {kept} of {written}")
            if not fake:
                time.sleep(PAUSE_S)
    print(f"kept {kept} of {written} (rest outside the tutor's topics), "
          f"{session.used_tokens} tokens -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--fake", action="store_true", help="offline backend, to test the pipeline")
    a = ap.parse_args()
    main(a.limit, a.fake)
