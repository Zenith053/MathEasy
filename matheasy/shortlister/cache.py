"""
The shortlist cache, so the final runs need no GPU (Role A).

Covers every tutor item and probe, with every answer a rule gives (B's and the
held-back ones) and every careless slip. Queries come from locate() and query(), as
in the tutor, so a cached step is found by its exact query text. A query that is not
cached goes to `fallback` (e.g. a ModelShortlister) or, without one, ranks nothing.

The file holds our own queries and Eedi IDs, no Eedi text. Names are not stored:
EediShortlister only reads IDs and scores.

  ranker = CachedShortlist()
  diagnose(before, after, shortlister=EediShortlister(ranker))

Usage: python -m matheasy.shortlister.cache --count         -> how many queries (no GPU)
       python -m matheasy.shortlister.cache --adapter DIR   -> data/shortlister/cache.json
"""

import argparse
import json
from pathlib import Path

from matheasy.diagnosis import ParseError, locate, probe_items
from matheasy.diagnosis.interfaces import query
from matheasy.evaluation.items import SAMPLE_ITEMS, SLIP_ITEMS
from matheasy.evaluation.students import rule_answers, slips

CACHE = Path(__file__).resolve().parents[2] / "data" / "shortlister" / "cache.json"
K = 25


def cache_items():
    return list(dict.fromkeys([*SAMPLE_ITEMS, *SLIP_ITEMS, *probe_items()]))


def cache_queries():
    """The query for every wrong answer on every item, sorted so builds are repeatable."""
    queries = set()
    for item in cache_items():
        for wrong in [*rule_answers(item), *slips(item)]:
            try:
                error = locate(item, wrong)
            except (ParseError, ValueError):  # an answer our parser cannot read
                continue
            if error is not None:
                queries.add(query(error))
    return sorted(queries)


def build(rank, queries, adapter, k=K):
    """The cache contents, given rank(queries, k) -> [[(id, name, score), ...], ...]."""
    ranked = rank(queries, k)
    return {"adapter": adapter, "k": k,
            "queries": {q: [[mid, round(score, 4)] for mid, _name, score in row] for q, row in zip(queries, ranked)}}


class CachedShortlist:
    """Same signature as shortlist(): query, k -> [(id, name, score), ...]. `misses` counts fallbacks."""

    def __init__(self, path=CACHE, fallback=None):
        self.table = json.loads(Path(path).read_text(encoding="utf-8"))["queries"]
        self.fallback = fallback
        self.misses = 0

    def __call__(self, query, k=K):
        row = self.table.get(query)
        if row is not None:
            return [(mid, "", score) for mid, score in row[:k]]
        self.misses += 1
        return self.fallback(query, k) if self.fallback else []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--adapter")
    ap.add_argument("--out", type=Path, default=CACHE)
    a = ap.parse_args()
    queries = cache_queries()
    print(f"{len(queries)} queries from {len(cache_items())} items")
    if a.adapter:
        from .model import ModelShortlister

        cache = build(ModelShortlister(a.adapter).rank, queries, Path(a.adapter).name)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(cache), encoding="utf-8")
        print(f"-> {a.out}")
