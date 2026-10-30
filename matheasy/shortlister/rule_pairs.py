"""
Rule-generated training pairs for the shortlister (Role A, run 4).

For every rule in B's bank that is mapped to Eedi misconceptions, make wrong
steps by running the rule on generated expressions, then build the query exactly
as the live tutor does (B's locate(), then the query() in
matheasy/diagnosis/interfaces.py). So these pairs look like live queries, unlike
the Eedi ones whose "Before" is a whole question.

Expressions use the probe number ranges in docs/diagnosis_rules.md (fractions:
denominators 2-12, numerators below the denominator). D's tests never match a
probe item, so training on this range cannot leak test items.

A rule with no Eedi mapping yet (bank.yaml `eedi: []`) gives no pairs: the
shortlister ranks Eedi misconceptions, so a pair needs an Eedi label.

Usage: python -m matheasy.shortlister.rule_pairs
  -> data/eedi/rule_pairs.csv  (MisconceptionId, RuleId, Before, Query)
"""

import itertools
import random
from pathlib import Path

import pandas as pd

from matheasy.diagnosis import equivalent, load_bank, locate, run_rule
from matheasy.diagnosis.interfaces import query

DATA = Path(__file__).resolve().parents[2] / "data" / "eedi"
SEED = 20261016
PER_RULE = 30  # wrong steps per rule


def fraction_inputs():
    """a/b op c/d with denominators 2-12 and numerators below the denominator."""
    fracs = [f"{n}/{d}" for d in range(2, 13) for n in range(1, d)]
    return [f"{x} {op} {y}" for x, y in itertools.product(fracs, repeat=2) for op in ["+", "-", "×", "÷"]]


# Input generators. Every rule is tried on all of them: a rule returns None where its
# misconception does not apply, so each rule picks its own inputs. Add decimals and
# expressions generators here to cover those rules.
INPUTS = [fraction_inputs]


def pairs_for_rule(rule_id, inputs, n=PER_RULE, seed=SEED):
    """Up to n (before, query) pairs where the rule applies and gives a wrong answer."""
    rng = random.Random(f"{seed}-{rule_id}")
    out = []
    for before in rng.sample(inputs, len(inputs)):
        wrong = run_rule(rule_id, before)
        if wrong is None or equivalent(wrong, before):
            continue
        error = locate(before, wrong)
        if error is None:
            continue
        out.append((before, query(error)))
        if len(out) == n:
            break
    return out


def build(bank=None):
    bank = bank or load_bank()
    rows, skipped = [], []
    inputs = [before for make in INPUTS for before in make()]
    for rule_id in bank.rules():
        m = bank[rule_id]
        pairs = pairs_for_rule(rule_id, inputs) if m.eedi else []
        if not pairs:
            skipped.append(rule_id)
            continue
        for before, q in pairs:
            rows += [{"MisconceptionId": e, "RuleId": rule_id, "Before": before, "Query": q} for e in m.eedi]
    return pd.DataFrame(rows, columns=["MisconceptionId", "RuleId", "Before", "Query"]), skipped


if __name__ == "__main__":
    pairs, skipped = build()
    print(f"{len(pairs)} pairs from {pairs.RuleId.nunique()} rules, {pairs.MisconceptionId.nunique()} Eedi misconceptions")
    if skipped:
        print(f"skipped {len(skipped)} rules (no Eedi mapping in bank.yaml, or no generated input it applies to):")
        print(*skipped, sep="\n  ")
    out = DATA / "rule_pairs.csv"
    pairs.to_csv(out, index=False)
    print(f"-> {out}")
