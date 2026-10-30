"""How often diagnose() names the right topic for a wrong step: rules, solve back, then the LLM.

Two sets of wrong steps with a known topic:

  rules  every worked example in tests/test_diagnosis_rule_library.py (PREDICTS), put
         inside a longer line: "2/3 + 1/5" -> "3" becomes "2/3 + 1/5 + 5 - 3" -> "5",
         so one operation is right and one is the rule's mistake. Equations and
         comparisons are used as they are. The topic is the rule's, by rule_topic().
  hand   mistakes no rule covers (slips, equations), labelled by hand, including slips
         inside algebra questions, where the right topic is not the question's.

The topics named are the rules' topics when rules diagnose the step (clear or
ambiguous), else the LLM's (explain()). A step counts as right if the true topic is
among those named; extra topics are counted separately. --no-llm scores the code alone.

    GEMINI_API_KEY=... PYTHONPATH=. .venv/bin/python eval/topic_accuracy.py [--set rules|hand] [--no-llm] [--dry-run]
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from test_diagnosis_rule_library import PREDICTS  # noqa: E402

from matheasy.diagnosis import check_step, diagnose, parse  # noqa: E402
from matheasy.diagnosis.explain import rule_topic  # noqa: E402
from matheasy.diagnosis.maths import Rel, _kinds  # noqa: E402
from matheasy.diagnosis.rules._common import decimal_node, evaluate, fraction_node  # noqa: E402
from matheasy.diagnosis.solve_back import _operations  # noqa: E402

# (before, student's line, true topic)
HAND = [
    ("1/2 + 1/3 + 1/6", "6/12", "fractions"),
    ("2x + 3 = 11", "x = 7", "solving_equations"),
    ("3(x + 4) = 21", "x = 17/3", "expanding_brackets"),
    ("5x - 2 = 3x + 8", "x = 5/4", "solving_equations"),
    ("4x + 2x - 3 + 5", "6x^2 + 2", "collecting_like_terms"),
    ("2(x - 3) - (x + 1)", "x - 5", "expanding_brackets"),
    ("x/2 + x/3 = 5", "x = 25/2", "fractions"),
    ("0.5 × 0.4 + 1.2", "3.2", "decimals"),
    ("3^2 × 3^4 - 10", "9^6 - 10", "exponents"),
    ("7 × 8 + 6", "60", "whole_number_arithmetic"),
    ("3/4 ÷ 1/2 + 1", "11/8", "fractions"),
    ("2/3 × 9 - 1", "17", "fractions"),
    # slips inside algebra: the topic is the slip's, not the question's
    ("7x = 63", "x = 8", "whole_number_arithmetic"),
    ("4(x + 7)", "4x + 32", "whole_number_arithmetic"),
    ("x - 15 = 27", "x = 32", "whole_number_arithmetic"),
    ("3x + 5x - 4", "8x^2 - 4", "collecting_like_terms"),
    ("-3 - 5 + 10", "12", "negative_numbers"),
]


def rule_cases():
    """(before, student's line, true topic) from each rule's worked examples, in a longer line."""
    cases = []
    for rule, problem, answer in PREDICTS:
        before, after = problem, answer
        if not isinstance(parse(problem), Rel):
            before = f"{problem} + 5 - 3"
            value = evaluate(parse(answer))
            if value is None or _operations(parse(answer)):
                after = f"{answer} + 2"  # letters or a written form (4.5 × 10^4): kept, and 5 - 3 done
            else:
                number = decimal_node if "decimals" in _kinds(parse(problem)) else fraction_node
                after = str(number(value + 2))
        try:
            if check_step(before, after):
                continue  # the mistake cancels out in this line
        except Exception:
            continue
        cases.append((before, after, rule_topic(rule)))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["rules", "hand", "both"], default="both")
    ap.add_argument("--no-llm", action="store_true", help="the code alone: rules and solve back")
    ap.add_argument("--dry-run", action="store_true", help="list the cases, no model calls")
    args = ap.parse_args()
    cases = (rule_cases() if args.set in ("rules", "both") else []) + (HAND if args.set in ("hand", "both") else [])
    if args.dry_run:
        for before, after, topic in cases:
            print(f"{before:32} -> {after:22} {topic}")
        print(f"{len(cases)} cases")
        return

    session = None
    if not args.no_llm:
        from matheasy.llm import LLMClient, load_config
        session = LLMClient(load_config(), run_id="topic-accuracy").session("topic-accuracy", "eval", budget_tokens=None)
    counts, misses, by_topic = Counter(), [], Counter()
    for before, after, topic in cases:
        d = diagnose(before, after, question=f"Work out {before}", session=session)
        if d.status != "unknown":
            named, source = {rule_topic(r) for r in d.beliefs}, "rules"
        elif d.explanation is not None:
            named, source = set(d.explanation.topics) - {"other"}, "llm"
        else:
            named, source = set(), "none"
        outcome = "right" if topic in named else "wrong" if named else "nothing"
        counts[outcome] += 1
        counts[source] += 1
        counts["named"] += len(named)
        by_topic[topic, outcome == "right"] += 1
        if outcome != "right":
            misses.append((before, after, topic, sorted(named), source))
    n = len(cases)
    print(f"right topic named: {counts['right']}/{n} ({counts['right'] / n:.0%}), wrong {counts['wrong']}, "
          f"nothing {counts['nothing']}; by rules {counts['rules']}, by the LLM {counts['llm']}")
    print(f"topics named: {counts['named']}, of which right {counts['right']} "
          f"(precision {counts['right'] / max(counts['named'], 1):.0%})")
    for topic in sorted({t for t, _ in by_topic}):
        print(f"  {topic:26} {by_topic[topic, True]}/{by_topic[topic, True] + by_topic[topic, False]}")
    print("\nmissed:")
    for before, after, topic, named, source in misses:
        print(f"  {before} -> {after}   true {topic}, named {named} by {source}")


if __name__ == "__main__":
    main()
