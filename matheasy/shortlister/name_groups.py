"""
Groups of Eedi misconception names that describe the same mistake (to-do T7, Role A).

Eedi often names one mistake several ways. The near-duplicates take up places in the
shortlist the LLM picks from, split its choice, and make a right pick look wrong in
evaluation. Similarity alone cannot group them: names above 0.98 cosine can be opposite
beliefs. So pairs are found by code and judged by a person:

  1. python -m matheasy.shortlister.name_groups pairs   -> data/eedi/name_pairs.csv
     In-topic name pairs with cosine >= 0.90 (model2vec) and word overlap >= 0.5.
  2. Fill in the `same` column of each pair: yes or no. A second person checks.
  3. python -m matheasy.shortlister.name_groups groups  -> data/eedi/name_groups.json
     Pairs judged "yes" are joined into groups. A group whose members were not all
     judged the same as each other is reported, so a chain a=b, b=c cannot quietly
     merge a and c.

Both files hold Eedi names or IDs, so they stay out of git like all Eedi-derived files.

  group_of()   -> {MisconceptionId: group key}, the smallest ID in its group; {} without the file
  canonical()  -> the group key of one ID (the ID itself when it is in no group)
  members()    -> every ID in that ID's group

Used (T7 step 3) by diagnosis/eedi.options() (one name per group in the LLM's list),
diagnose() (a pick is recorded as its group), bank.for_eedi() (a rule mapped to one member
owns the group), evaluation.metrics.accepted() and eval/eedi_pick.py (a pick in the right
group counts as right). Without the group file, all of these behave as before.
"""

import argparse
import json
import re
from functools import lru_cache
from itertools import combinations
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "eedi"
PAIRS = DATA / "name_pairs.csv"
GROUPS = DATA / "name_groups.json"
MIN_COSINE = 0.90
MIN_OVERLAP = 0.5
EMBED_MODEL = "minishlab/potion-base-8M"  # the CPU model retrieve() uses


def overlap(a: str, b: str) -> float:
    """Share of words the two names have in common (Jaccard)."""
    wa, wb = (set(re.findall(r"[a-z0-9]+", t.lower())) for t in (a, b))
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def join(pairs):
    """Groups from pairs judged the same: {member: smallest member of its group}."""
    parent = {}

    def root(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {x: root(x) for x in parent}


def unchecked(groups, same):
    """For each group of 3 or more, the member pairs nobody judged the same."""
    judged = {frozenset(p) for p in same}
    members = {}
    for x, g in groups.items():
        members.setdefault(g, []).append(x)
    return {g: [p for p in combinations(sorted(m), 2) if frozenset(p) not in judged]
            for g, m in members.items() if len(m) > 2}


@lru_cache(maxsize=1)
def group_of() -> dict:
    """{MisconceptionId: group key} for every grouped name; names in no group are absent."""
    if not GROUPS.exists():
        return {}
    return {int(k): v for k, v in json.loads(GROUPS.read_text(encoding="utf-8"))["group_of"].items()}


def canonical(misconception_id: int) -> int:
    """The group key (its smallest ID) of a grouped name, else the ID itself: one ID per mistake."""
    return group_of().get(int(misconception_id), int(misconception_id))


def members(misconception_id: int) -> set:
    """Every name in the same group as this one, itself included."""
    key = canonical(misconception_id)
    return {m for m, g in group_of().items() if g == key} | {int(misconception_id)}


def write_pairs(out=PAIRS):
    import numpy as np
    import pandas as pd
    from model2vec import StaticModel

    from .candidates import candidate_ids

    names = pd.read_csv(DATA / "raw" / "misconception_mapping.csv")
    names = names[names.MisconceptionId.isin(candidate_ids())].reset_index(drop=True)
    ids, text = names.MisconceptionId.tolist(), names.MisconceptionName.str.strip().tolist()
    vec = StaticModel.from_pretrained(EMBED_MODEL).encode(text)
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    sim = vec @ vec.T
    rows = [{"a": ids[i], "b": ids[j], "cosine": round(float(sim[i, j]), 3),
             "overlap": round(overlap(text[i], text[j]), 2), "same": "",
             "name_a": text[i], "name_b": text[j]}
            for i, j in combinations(range(len(ids)), 2)
            if sim[i, j] >= MIN_COSINE and overlap(text[i], text[j]) >= MIN_OVERLAP]
    pd.DataFrame(rows).sort_values("cosine", ascending=False).to_csv(out, index=False, encoding="utf-8")
    print(f"{len(rows)} pairs from {len(ids)} in-topic names -> {out}")


def write_groups(pairs=PAIRS, out=GROUPS):
    import pandas as pd

    judged = pd.read_csv(pairs, encoding="utf-8")
    answer = judged.same.fillna("").str.strip().str.lower()
    if not answer.isin(["yes", "no"]).all():
        raise SystemExit(f"{(~answer.isin(['yes', 'no'])).sum()} pairs are not judged yes or no yet")
    same = [(int(a), int(b)) for a, b, s in zip(judged.a, judged.b, answer) if s == "yes"]
    groups = join(same)
    gaps = unchecked(groups, same)
    out.write_text(json.dumps({"pairs judged": len(judged), "pairs same": len(same),
                               "group_of": {str(k): v for k, v in sorted(groups.items())}}, indent=1),
                   encoding="utf-8")
    sizes = pd.Series(groups).value_counts()
    print(f"{len(same)} of {len(judged)} pairs judged the same -> {sizes.size} groups, "
          f"{len(groups)} names ({(sizes > 2).sum()} groups of 3 or more) -> {out}")
    for g, missing in gaps.items():
        if missing:
            print(f"  group {g}: members never judged against each other: {missing}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pairs", "groups"])
    step = ap.parse_args().step
    write_pairs() if step == "pairs" else write_groups()
