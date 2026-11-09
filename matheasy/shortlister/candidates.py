"""
The Eedi misconceptions the shortlister may rank: only those in the tutor's topics.

Ranking all 2,587 names lets geometry, statistics and the like crowd the top 10. A
name is a candidate if either

  - it labels a wrong answer of a training question in our topics (pairs.csv), or
  - it has no training question, and the LLM put it in one of our topics
    (llm_examples.csv, whose Topic is "other" for everything else).

Val and test labels are never used, so the list cannot leak test answers. About 90%
of val/test wrong answers have their misconception in the list: all of those seen in
training, about 70% of the unseen ones.

The list holds IDs only, no Eedi text, so it is kept in the repo like eedi_split.json.

Usage: python -m matheasy.shortlister.candidates  -> matheasy/shortlister/candidates.json
"""

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "eedi"
FILE = Path(__file__).with_name("candidates.json")


def build():
    pairs = pd.read_csv(DATA / "pairs.csv")
    llm = pd.read_csv(DATA / "llm_examples.csv")
    trained = set(pairs[pairs.Split == "train"].MisconceptionId)
    tagged = set(llm[llm.Topic != "other"].MisconceptionId)
    ids = sorted(int(i) for i in trained | tagged)
    FILE.write_text(json.dumps({"MisconceptionIds": ids}, indent=0))
    for split in ("val", "test"):
        rows = pairs[pairs.Split == split]
        print(f"{split}: true misconception among candidates for {rows.MisconceptionId.isin(ids).mean():.1%}")
    print(f"{len(ids)} candidates ({len(trained)} with training questions, {len(tagged)} tagged by the LLM) -> {FILE}")


@lru_cache(maxsize=1)
def candidate_ids() -> frozenset:
    return frozenset(json.loads(FILE.read_text(encoding="utf-8"))["MisconceptionIds"])


if __name__ == "__main__":
    build()
