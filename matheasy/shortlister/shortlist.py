"""
shortlist(): rank Eedi misconceptions for one wrong step (Role A).

Week-1 placeholder: BM25 over the misconception names. The fine-tuned model
replaces the scoring in week 2-3; the signature stays the same, so B's
diagnoser and the B2 baseline do not change.

  shortlist(query, k=25) -> [(misconception_id, name, score), ...]  best first

`query` is the text from build_query(). Candidates are all ~2,600 Eedi
misconceptions, so a wrong answer can be ranked even when its misconception
never appears in training. Needs data/eedi/raw/misconception_mapping.csv (see
prepare_eedi.py).

Usage: python -m matheasy.shortlister.shortlist  -> MAP@25 and Recall@10 on val and test
"""

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

DATA = Path(__file__).resolve().parents[2] / "data" / "eedi"

STOPWORDS = set("""a an the of to in on for and or is are be by with as at it this that
when from not does do their they them which what than then into its""".split())


def tokens(text):
    """Words of 2+ letters; drops LaTeX commands (\\frac), variables (x, m) and digits."""
    text = re.sub(r"\\[a-zA-Z]+", " ", str(text)).lower()
    return [w for w in re.findall(r"[a-z]{2,}", text) if w not in STOPWORDS]


@lru_cache(maxsize=1)
def _index():
    names = pd.read_csv(DATA / "raw" / "misconception_mapping.csv")
    bm25 = BM25Okapi([tokens(n) for n in names.MisconceptionName])
    return names.MisconceptionId.tolist(), names.MisconceptionName.str.strip().tolist(), bm25


def shortlist(query, k=25):
    ids, names, bm25 = _index()
    scores = bm25.get_scores(tokens(query))
    top = sorted(range(len(ids)), key=lambda i: -scores[i])[:k]
    return [(ids[i], names[i], float(scores[i])) for i in top]


def map_at_k(ranked, true, k=25):
    """Mean of 1/rank of the true misconception; 0 when it is not in the top k."""
    return sum(1 / (r.index(t) + 1) if t in r[:k] else 0 for r, t in zip(ranked, true)) / len(true)


def recall_at_k(ranked, true, k=10):
    return sum(t in r[:k] for r, t in zip(ranked, true)) / len(true)


if __name__ == "__main__":
    queries = pd.read_csv(DATA / "queries.csv")
    for split in ["val", "test"]:
        q = queries[queries.Split == split]
        ranked = [[m for m, _, _ in shortlist(text)] for text in q.Query]
        true = q.MisconceptionId.tolist()
        print(f"{split}: {len(q)} wrong answers  MAP@25 = {map_at_k(ranked, true):.3f}  "
              f"Recall@10 = {recall_at_k(ranked, true):.3f}")
    example = queries[queries.Split == "test"].iloc[0]
    print(f"\nexample: {example.Query[:160]}...\ntrue: {example.MisconceptionName.strip()}")
    for mid, name, score in shortlist(example.Query, k=5):
        print(f"  {score:5.1f}  {mid:5d}  {name}")
