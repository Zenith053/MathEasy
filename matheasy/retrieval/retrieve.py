"""
retrieve(): textbook passages for the hinter to cite (Role A).

Week-1 placeholder: BM25 over the OpenStax passages. Embeddings are added in
week 2 (BM25 + embeddings); the signature stays the same, so C's hinter does
not change.

  retrieve(query, k=3) -> [{"ref", "book", "section", "section_title",
                            "subsection", "text", "score"}, ...]  best first

`query` is free text, e.g. the diagnosed misconception plus the topic. "ref" is
what the hinter cites, e.g. "Prealgebra 2e, 4.2 Multiply and Divide Fractions >
Simplify Fractions". Needs data/openstax/passages.jsonl (python -m matheasy.retrieval.openstax).

Usage: python -m matheasy.retrieval.retrieve "adds numerators and denominators"
"""

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

PASSAGES = Path(__file__).resolve().parents[2] / "data" / "openstax" / "passages.jsonl"

STOPWORDS = set("""a an the of to in on for and or is are be by with as at it this that
when from not does do their they them which what than then into its we you can""".split())


def tokens(text):
    return [w for w in re.findall(r"[a-z]{2,}", str(text).lower()) if w not in STOPWORDS]


@lru_cache(maxsize=1)
def _index():
    rows = [json.loads(line) for line in PASSAGES.read_text(encoding="utf-8").splitlines()]
    # Headings carry the topic, so they are indexed with the text.
    docs = [tokens(f"{r['section_title']} {r['subsection']} {r['text']}") for r in rows]
    return rows, BM25Okapi(docs)


def retrieve(query, k=3):
    rows, bm25 = _index()
    scores = bm25.get_scores(tokens(query))
    top = sorted(range(len(rows)), key=lambda i: -scores[i])[:k]
    return [{**rows[i], "score": float(scores[i])} for i in top]


if __name__ == "__main__":
    queries = sys.argv[1:] or [
        "Adds the numerators and adds the denominators when adding fractions",
        "Thinks that the longer decimal is always larger, 0.25 > 0.3",
        "Only multiplies the first term in the expansion of a bracket",
        "Believes multiplying two negatives gives a negative answer",
    ]
    for q in queries:
        print(f"\n{q}")
        for p in retrieve(q):
            print(f"  {p['score']:5.1f}  {p['ref']}")
