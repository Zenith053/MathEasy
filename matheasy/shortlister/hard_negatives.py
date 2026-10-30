"""
Hard negatives for the shortlister training (Role A, week 2).

For each training pair (query, true misconception) pick misconceptions that are
easy to confuse with the true one, so the model learns to tell them apart
instead of only separating unrelated misconceptions:

  1. siblings: misconceptions behind the other wrong options of the same
     question (same question, different error);
  2. look-alikes: misconceptions whose names are closest to the true name
     (BM25 between names).

Names that are near-duplicates of the true name, or a more general version of
it, are skipped (Eedi has many rewordings of the same misconception), so a
correct label is never a negative.
Only training questions are used, so val and test stay untouched.

Round 1 uses these text-based negatives. Round 2 re-mines negatives with the
round-1 model (in the training notebook).

Usage: python -m matheasy.shortlister.hard_negatives
  -> data/eedi/train_negatives.csv  (QuestionId, WrongAnswer, MisconceptionId, NegativeIds)
"""

from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

from .shortlist import tokens

DATA = Path(__file__).resolve().parents[2] / "data" / "eedi"
N_NEG = 7          # negatives stored per pair; the notebook may use fewer
NEAR_DUP = 0.8     # word overlap (Jaccard) at which a name counts as the same misconception


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def same_misconception(a, b):
    """Near-duplicate names, or one name is a more general version of the other."""
    return jaccard(a, b) >= NEAR_DUP or set(a) <= set(b) or set(b) <= set(a)


def look_alikes(names):
    """For every misconception, the other misconception IDs ranked by name similarity, best first."""
    ids = names.MisconceptionId.tolist()
    toks = [tokens(n) for n in names.MisconceptionName]
    bm25 = BM25Okapi(toks)
    out = {}
    for i, mid in enumerate(ids):
        scores = bm25.get_scores(toks[i])
        ranked = sorted(range(len(ids)), key=lambda j: -scores[j])
        out[mid] = [ids[j] for j in ranked[:60]
                    if j != i and not same_misconception(toks[i], toks[j])]
    return out, dict(zip(ids, toks))


def mine(pairs, names, n_neg=N_NEG):
    """One row per training pair with up to n_neg negative misconception IDs."""
    similar, toks = look_alikes(names)
    siblings = pairs.groupby("QuestionId").MisconceptionId.apply(list).to_dict()
    rows = []
    for r in pairs.itertuples():
        pos = r.MisconceptionId
        negs = []
        for cand in [m for m in siblings[r.QuestionId] if m != pos] + similar[pos]:
            if cand not in negs and not same_misconception(toks[pos], toks[cand]):
                negs.append(cand)
        rows.append({"QuestionId": r.QuestionId, "WrongAnswer": r.WrongAnswer, "MisconceptionId": pos,
                     "NegativeIds": " ".join(map(str, negs[:n_neg]))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    queries = pd.read_csv(DATA / "queries.csv")
    names = pd.read_csv(DATA / "raw" / "misconception_mapping.csv")
    train = queries[queries.Split == "train"]
    negs = mine(train, names)

    n_sib = sum(len(set(map(int, row.NegativeIds.split())) &
                    set(train[train.QuestionId == row.QuestionId].MisconceptionId)) for row in negs.itertuples())
    counts = negs.NegativeIds.str.split().map(len)
    print(f"{len(negs)} training pairs, {counts.mean():.1f} negatives each (min {counts.min()}), "
          f"{n_sib / counts.sum():.0%} of negatives are siblings")
    name_of = dict(zip(names.MisconceptionId, names.MisconceptionName.str.strip()))
    ex = negs.iloc[0]
    print(f"\nexample, true: {name_of[ex.MisconceptionId]}")
    for m in ex.NegativeIds.split()[:4]:
        print(f"  negative: {name_of[int(m)]}")

    out = DATA / "train_negatives.csv"
    negs.to_csv(out, index=False)
    print(f"-> {out}")
