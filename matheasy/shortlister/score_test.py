"""
Test scores of every trained adapter with 95% intervals, and paired differences
between runs (Role A, needs a GPU).

The training notebook prints point scores only. This ranks the test rows again with
each adapter (and BM25), then resamples the rows 2,000 times for intervals. A
difference between two runs is paired: each resample scores both runs on the same
rows, so the interval reflects the runs, not which rows happened to be drawn. Only
scores are written, no Eedi text.

Usage: python -m matheasy.shortlister.score_test --adapters DIR [DIR ...]
           --pairs NAME:NAME [...] --out FILE
NAME is an adapter folder's name, or BM25.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .shortlist import DATA, map_at_k, recall_at_k, shortlist

N_BOOT = 2000
SEED = 20261016


def per_row(ranked, true):
    """(reciprocal rank if in the top 25, else 0; 1 if in the top 10, else 0) for each row.
    Their means are MAP@25 and Recall@10."""
    rr = np.array([1 / (r.index(t) + 1) if t in r[:25] else 0.0 for r, t in zip(ranked, true)])
    hit = np.array([1.0 if t in r[:10] else 0.0 for r, t in zip(ranked, true)])
    return rr, hit


def _draws(n):
    """The same resampled row indices for every call of the same size."""
    return np.random.default_rng(SEED).integers(0, n, (N_BOOT, n))


def interval(values):
    """Mean of `values`, with a 95% bootstrap interval over rows."""
    values = np.asarray(values, dtype=float)
    means = values[_draws(len(values))].mean(axis=1)
    return {"n": len(values), "mean": float(values.mean()),
            "ci95": [float(x) for x in np.percentile(means, [2.5, 97.5])]}


def summary(rr, hit, unseen):
    """MAP@25 and Recall@10 on all rows, and MAP@25 on rows whose misconception is seen
    or unseen in training, each with its interval."""
    return {"MAP@25": interval(rr), "Recall@10": interval(hit),
            "MAP@25 seen": interval(rr[~unseen]), "MAP@25 unseen": interval(rr[unseen])}


def difference(a, b, unseen):
    """a minus b, paired by row: MAP@25 (all, seen, unseen) and Recall@10."""
    (rr_a, hit_a), (rr_b, hit_b) = a, b
    return {"MAP@25": interval(rr_a - rr_b), "Recall@10": interval(hit_a - hit_b),
            "MAP@25 seen": interval((rr_a - rr_b)[~unseen]),
            "MAP@25 unseen": interval((rr_a - rr_b)[unseen])}


def main(adapters, pairs, out):
    queries = pd.read_csv(DATA / "queries.csv")
    seen = set(queries[queries.Split == "train"].MisconceptionId)
    test = queries[queries.Split == "test"]
    true = test.MisconceptionId.tolist()
    unseen = np.array([t not in seen for t in true])

    def score(ranked):
        rr, hit = per_row(ranked, true)
        assert abs(rr.mean() - map_at_k(ranked, true, 25)) < 1e-9
        assert abs(hit.mean() - recall_at_k(ranked, true, 10)) < 1e-9
        return rr, hit

    rows = {"BM25": score([[m for m, _n, _s in shortlist(q, 25)] for q in test.Query])}
    for path in adapters:
        import torch  # only the adapters need it

        from .model import ModelShortlister
        ranker = ModelShortlister(path)
        rows[Path(path).name] = score([[m for m, _n, _s in row] for row in ranker.rank(test.Query.tolist())])
        print(Path(path).name, "MAP@25", round(rows[Path(path).name][0].mean(), 4), flush=True)
        del ranker
        gc.collect()
        torch.cuda.empty_cache()

    results = {"test rows": len(true), "unseen rows": int(unseen.sum()), "resamples": N_BOOT,
               "runs": {name: summary(rr, hit, unseen) for name, (rr, hit) in rows.items()},
               "differences": {f"{a} - {b}": difference(rows[a], rows[b], unseen)
                               for a, b in (p.split(":") for p in pairs)}}
    print(json.dumps(results, indent=1))
    Path(out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="*", default=[], help="NAME:NAME, scored as the first minus the second")
    ap.add_argument("--out", default="test_scores.json")
    a = ap.parse_args()
    main(a.adapters, a.pairs, a.out)
