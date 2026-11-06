"""
retrieve(): textbook passages for the hinter to cite (Role A).

Hybrid search over the OpenStax passages: BM25 (exact words) and embeddings
(meaning, e.g. "multiplies only the first term in the bracket" -> Distributive
Property), merged by reciprocal-rank fusion. Embeddings are model2vec static
embeddings: small, CPU-only, no torch. Without model2vec installed, BM25 alone.

  retrieve(query, k=3, mode="hybrid") -> [{"ref", "book", "section", "section_title",
                                           "subsection", "text", "score", "similarity"}, ...]  best first

mode is "hybrid", "bm25" or "embed" (the last two for comparison). "score" only
orders the results: in hybrid mode it reflects rank, so it cannot show a weak match.
"similarity" is the passage's cosine similarity to the query, comparable between
queries; it is None in BM25 mode.

`query` is free text, e.g. the diagnosed misconception plus the topic. "ref" is
what the hinter cites, e.g. "Prealgebra 2e, 4.2 Multiply and Divide Fractions >
Simplify Fractions". Needs data/openstax/passages.jsonl (python -m matheasy.retrieval.openstax).

Usage: python -m matheasy.retrieval.retrieve "adds numerators and denominators"
       python -m matheasy.retrieval.retrieve --eval   -> right-section rate for each mode
"""

import json
import re
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

PASSAGES = Path(__file__).resolve().parents[2] / "data" / "openstax" / "passages.jsonl"
EVAL = Path(__file__).resolve().parent / "eval_sections.json"
EMBED_MODEL = "minishlab/potion-base-8M"
RRF_K = 60  # standard reciprocal-rank-fusion constant
# Best similarity on the 30 eval queries with a right-section result: at least 0.56.
# On 10 off-topic queries: at most 0.39. An on-topic wrong section can still score high.
WEAK_BELOW = 0.45

# Eedi and the tutor use UK maths words; OpenStax is American. The US words are
# added to the query (not replacing the UK ones).
UK_TO_US = {
    r"\bbrackets?\b": "parentheses distributive property",
    r"\bexpand(s|ing|ed)?\b": "distribute multiply",
    r"\btop and bottom\b": "numerator and denominator",
    r"\bindices\b|\bpowers?\b": "exponents",
    r"\bbidmas\b|\bbodmas\b": "order of operations",
    r"\brecurring\b": "repeating",
    r"\bhighest common factor\b": "greatest common factor",
    r"\blowest common multiple\b": "least common multiple",
    r"\bdirected numbers?\b": "integers",
}


def americanise(query):
    extra = [us for uk, us in UK_TO_US.items() if re.search(uk, query, flags=re.I)]
    return " ".join([query, *extra])

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


@lru_cache(maxsize=1)
def _embedder():
    try:
        from model2vec import StaticModel
    except ImportError:
        warnings.warn("model2vec not installed: retrieve() uses BM25 only (pip install model2vec)")
        return None
    return StaticModel.from_pretrained(EMBED_MODEL)


def _embed(texts):
    e = _embedder().encode(texts)
    return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)


@lru_cache(maxsize=1)
def _vectors():
    rows, _ = _index()
    return _embed([f"{r['section_title']}. {r['subsection']}. {r['text']}" for r in rows])


def retrieve(query, k=3, mode="hybrid"):
    rows, bm25 = _index()
    query = americanise(query)
    if mode != "bm25" and _embedder() is None:
        mode = "bm25"
    cosine = None if mode == "bm25" else _vectors() @ _embed([query])[0]
    if mode == "bm25":
        scores = bm25.get_scores(tokens(query))
    elif mode == "embed":
        scores = cosine
    else:
        # Reciprocal-rank fusion: no score scales to match, no weight to tune.
        scores = np.zeros(len(rows))
        for s in (bm25.get_scores(tokens(query)), cosine):
            scores[np.argsort(-s)] += 1 / (RRF_K + np.arange(1, len(rows) + 1))
    top = sorted(range(len(rows)), key=lambda i: -scores[i])[:k]
    return [{**rows[i], "score": float(scores[i]), "similarity": None if cosine is None else float(cosine[i])}
            for i in top]


def is_weak(passages):
    """True if even the best passage is unrelated to the query. False when similarity is unknown."""
    known = [p["similarity"] for p in passages if p.get("similarity") is not None]
    return bool(known) and max(known) < WEAK_BELOW


def right_section_rate(k=3, mode="hybrid"):
    """Share of the hand-labelled queries with a passage from a correct section in the top k."""
    cases = json.loads(EVAL.read_text(encoding="utf-8"))["queries"]
    hits = [any(p["section_title"] in c["sections"] for p in retrieve(f"{c['query']} {c['topic']}", k, mode))
            for c in cases]
    return sum(hits) / len(hits), [c["query"] for c, h in zip(cases, hits) if not h]


if __name__ == "__main__":
    if sys.argv[1:] == ["--eval"]:
        for mode in ["bm25", "embed", "hybrid"]:
            rate1, _ = right_section_rate(1, mode)
            rate3, missed = right_section_rate(3, mode)
            print(f"{mode:6s}  right section in top 1: {rate1:.0%}  top 3: {rate3:.0%}")
        print("\nhybrid misses (top 3):", *missed, sep="\n  ")
        sys.exit()
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
