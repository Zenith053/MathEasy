"""Does the LLM pick the right Eedi misconception from the shortlister's names? (Stage 3 of diagnose().)

Wrong answers whose Eedi misconception is known, with no Eedi question sent anywhere:

  held-back   D's held-back rules (evaluation/held_back.py) with Eedi IDs (eval/held_back_rules.json),
              run on our own sample items and probe items, keeping only answers no rule of ours
              explains: the case stage 3 exists for.
  mapped      some of our rules mapped to Eedi (bank.yaml), on probe items they get wrong. A rule
              would catch these first, so stage 3 is called directly to measure the pick alone.

For each: the shortlister (v2, local GPU) gives EEDI_K names, then one LLM call (explain()) picks.

    PYTHONPATH=. .venv/bin/python eval/eedi_pick.py --dry-run     # the cases, no GPU, no LLM
    GEMINI_API_KEY=... PYTHONPATH=. .venv/bin/python eval/eedi_pick.py [--limit 24]
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from matheasy.diagnosis import EEDI_K, check_step, diagnose, equivalent, load_bank, locate, probe_items
from matheasy.diagnosis.eedi import options
from matheasy.diagnosis.explain import explain
from matheasy.diagnosis.probes import _bank as probe_bank
from matheasy.diagnosis.solve_back import solve_back
from matheasy.evaluation.held_back import held_back_rule
from matheasy.evaluation.items import SAMPLE_ITEMS

ROOT = Path(__file__).resolve().parents[1]
PER_HELD_BACK = 2
MAPPED = ["fractions.add_numerators_denominators", "decimals.ignore_decimal_point",
          "negative_numbers.negative_times_negative_negative", "order_of_operations.strict_left_to_right",
          "exponents.multiply_base_by_exponent", "algebra.divide_one_term_only",
          "percentages.percent_equals_decimal", "exponents.zero_exponent_equals_zero"]
SPACING_S = 4.5  # the free tier allows 15 requests a minute


def held_back_cases():
    status = json.loads((ROOT / "eval" / "held_back_rules.json").read_text())["eedi_status"]
    items = list(dict.fromkeys([*SAMPLE_ITEMS, *probe_items()]))
    for name, ids in status.items():
        if not isinstance(ids, list):
            continue  # "never_seen": no Eedi name to find
        rule, found = held_back_rule(name), 0
        for item in items:
            try:
                wrong = rule.predict(item)
                if wrong is None or check_step(item, wrong):
                    continue
                if diagnose(item, wrong).status != "unknown":
                    continue  # one of our rules explains it: stage 3 would never see it
            except Exception:
                continue
            yield {"kind": "held-back", "source": name, "before": item, "after": wrong, "truth": [f"eedi:{i}" for i in ids]}
            found += 1
            if found == PER_HELD_BACK:
                break


def mapped_cases():
    bank = load_bank()
    for name in MAPPED:
        rid = f"rule:{name}"
        entry = next((e for e in probe_bank() if rid in e["predictions"]), None)
        if entry:
            yield {"kind": "mapped", "source": name, "before": entry["question"], "after": entry["predictions"][rid],
                   "truth": [f"eedi:{i}" for i in bank[rid].eedi]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24, help="at most this many LLM calls")
    ap.add_argument("--dry-run", action="store_true", help="list the cases only")
    ap.add_argument("--out", default="eedi_pick_results.json")
    args = ap.parse_args()
    cases = (list(held_back_cases()) + list(mapped_cases()))[:args.limit]
    for c in cases:
        print(f"{c['kind']:9} {c['source'][:42]:42} {c['before']:>18} -> {c['after']:<14} truth {c['truth']}")
    print(f"{len(cases)} cases")
    if args.dry_run:
        return

    from matheasy.diagnosis import load_ranker
    from matheasy.llm import LLMClient, load_config
    ranker = load_ranker()
    if ranker is None:
        sys.exit("the shortlister did not load (needs torch, a GPU and data/shortlister/adapter-3B-v2)")
    session = LLMClient(load_config(), run_id="eedi-pick").session("eedi-pick", "eval", budget_tokens=None)
    for c in cases:
        located = locate(c["before"], c["after"])
        offered = options(located, ranker, k=EEDI_K)
        e = explain(located, question=f"Work out {c['before']}", session=session,
                    pieces=solve_back(c["before"], c["after"]), options=offered)
        c.update(offered=[m for m, _ in offered], in_list=any(m in c["truth"] for m, _ in offered),
                 pick=e.eedi, fit=e.eedi_fit, kind_said=e.kind, right=e.eedi in c["truth"], working=e.status)
        print(f"  {c['source'][:40]:40} in list {c['in_list']!s:5} pick {e.eedi} ({e.eedi_fit}) "
              f"{'RIGHT' if c['right'] else 'wrong'}  [{e.kind}, {e.status}]")
        time.sleep(SPACING_S)

    n = len(cases)
    in_list = [c for c in cases if c["in_list"]]
    picked = [c for c in cases if c["pick"]]
    report = {
        "cases": n, "llm_calls": n,
        "truth in the shortlister's list": f"{len(in_list)}/{n}",
        "right pick overall": f"{sum(c['right'] for c in cases)}/{n}",
        "right pick when the truth was in the list": f"{sum(c['right'] for c in in_list)}/{len(in_list)}",
        "no pick (slip or unclear)": f"{n - len(picked)}/{n}",
        "exact picks right": f"{sum(c['right'] for c in picked if c['fit'] == 'exact')}/{sum(c['fit'] == 'exact' for c in picked)}",
        "close picks right": f"{sum(c['right'] for c in picked if c['fit'] == 'close')}/{sum(c['fit'] == 'close' for c in picked)}",
        "by kind": {k: f"{sum(c['right'] for c in cases if c['kind'] == k)}/{sum(c['kind'] == k for c in cases)}"
                    for k in ("held-back", "mapped")},
    }
    print(json.dumps(report, indent=1))
    Path(args.out).write_text(json.dumps({"report": report, "cases": cases}, indent=1))


if __name__ == "__main__":
    main()
