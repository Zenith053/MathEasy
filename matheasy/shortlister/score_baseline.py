"""
How good is a trained shortlister at the tutor's job? (Needs a GPU, or patience on a CPU.)

The tutor only sends calculation steps (prepare_eedi.py's Form), only ranks the
in-topic names (candidates.py), and the LLM then picks from the top 10. So this
scores val and test calculation rows four ways:

                      all 2,587 names     972 in-topic names
  training format           a                     b
  live format               c                     d   <- the number that matters

Live format exists only for rows locate() can read (live_queries.py), so c and d use
those rows, and a and b are also given on the same rows for a like-for-like gap.
Only scores are written, no Eedi text.

Usage: python -m matheasy.shortlister.score_baseline --adapter data/shortlister/adapter-3B-llm
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from .candidates import candidate_ids
from .model import DATA, ModelShortlister
from .shortlist import map_at_k, recall_at_k

KEY = ["QuestionId", "WrongAnswer"]


def scores(ranked, true, seen):
    unseen = [t not in seen for t in true]
    pick = lambda keep: ([r for r, u in zip(ranked, unseen) if u == keep], [t for t, u in zip(true, unseen) if u == keep])
    return {"n": len(true), "Recall@10": recall_at_k(ranked, true, 10), "Recall@25": recall_at_k(ranked, true, 25),
            "Recall@50": recall_at_k(ranked, true, 50), "MAP@25": map_at_k(ranked, true, 25),
            "Recall@10 seen": recall_at_k(*pick(False), 10), "Recall@10 unseen": recall_at_k(*pick(True), 10),
            "top-1": sum(r[:1] == [t] for r, t in zip(ranked, true)) / max(len(true), 1)}


def main(adapter, out):
    pairs = pd.read_csv(DATA / "pairs.csv")
    seen = set(pairs[pairs.Split == "train"].MisconceptionId)
    calc = pairs[pairs.Split.isin(["val", "test"]) & (pairs.Form == "calculation")][KEY + ["MisconceptionId"]]
    train_fmt = pd.read_csv(DATA / "queries.csv")[KEY + ["Query"]].merge(calc, on=KEY)
    live_fmt = pd.read_csv(DATA / "live_queries.csv")[KEY + ["Query"]].merge(calc, on=KEY)

    ranker = ModelShortlister(adapter)
    everything = len(ranker.ids)
    keep = candidate_ids()

    def rank(df):
        full = [[mid for mid, _name, _score in row] for row in ranker.rank(df.Query.tolist(), k=everything)]
        return full, [[m for m in row if m in keep] for row in full]

    results = {"rows": {"calculation, val + test": len(calc), "of which live format": len(live_fmt)}}
    t_all, t_in = rank(train_fmt)
    l_all, l_in = rank(live_fmt)
    true_t, true_l = train_fmt.MisconceptionId.tolist(), live_fmt.MisconceptionId.tolist()
    live_keys = set(map(tuple, live_fmt[KEY].values))
    same = [i for i, k in enumerate(map(tuple, train_fmt[KEY].values)) if k in live_keys]
    results["all calculation rows, training format"] = {
        "all names": scores(t_all, true_t, seen), "in-topic names": scores(t_in, true_t, seen)}
    results["live-format rows"] = {
        "a: training format, all names": scores([t_all[i] for i in same], [true_t[i] for i in same], seen),
        "b: training format, in-topic names": scores([t_in[i] for i in same], [true_t[i] for i in same], seen),
        "c: live format, all names": scores(l_all, true_l, seen),
        "d: live format, in-topic names": scores(l_in, true_l, seen)}
    text = json.dumps(results, indent=1)
    print(text)
    Path(out).write_text(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", default="baseline_scores.json")
    ap.add_argument("--small-gpu", action="store_true", help="small batches, for a GPU with about 4 GB")
    a = ap.parse_args()
    if a.small_gpu:
        from . import model
        model.NAME_BATCH, model.QUERY_BATCH = 8, 2
    main(a.adapter, a.out)
