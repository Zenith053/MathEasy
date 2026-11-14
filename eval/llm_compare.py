"""Gemini or an open model on vLLM: which one is better for this project? (A's test, 14 November)

`run` puts whichever model the LLM config names through three tests and saves scores only:

  pick    eval/eedi_pick.py's 22 cases: does the LLM pick the right Eedi name from the
          shortlister's list? (needs the shortlister: a GPU and data/)
  topics  eval/topic_accuracy.py's cases: is the right topic named for a wrong step? Only the
          steps no rule explains reach the LLM, so only those are compared.
  tutor   the full tutor, one session each, on every student the rules cannot read (known as
          text, never seen, slip only), every 3rd student with a rule, and every 3rd adversarial
          student, scored with D's metrics.

Only the model changes between two runs. The rest is held fixed for both:
  - the simulated students judge hints with the keyword judge, and adversarial students
    use their template attacks: with the LLM playing the student too, a change of model
    would change the students as well as the tutor;
  - no textbook citations (the index is not on Kaggle);
  - no Eedi name groups (data/eedi/name_groups.json is not on Kaggle).

    MATHEASY_LLM_CONFIG=config/llm_vllm.yaml python eval/llm_compare.py run --out vllm.json
    GEMINI_API_KEY=... python eval/llm_compare.py run --out gemini.json
    python eval/llm_compare.py compare gemini.json vllm.json

The saved file holds right/wrong per case, D's metrics per student, and call counts: no Eedi
names, IDs or questions, so it may leave Kaggle. Logs and turn records stay in --work.
A run stopped by a quota carries on where it stopped when run again (calls are cached).
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval"), str(ROOT / "tests")]

TUTOR_METRICS = ("top1_accuracy", "topic_accuracy", "over_claiming_rate", "unknowable_claim_rate", "leak_rate")
TUTOR_GROUPS = ("all", "has_rule", "known_as_text", "slip_only", "attack:all")
EVERY = 3  # every 3rd student with a rule, and every 3rd adversarial student
SPACING_S = 4.5  # Gemini's free tier allows 15 requests a minute


def _llm_errors():
    from matheasy.llm import BudgetExceeded, LLMCallError, LLMOutputError
    return BudgetExceeded, LLMCallError, LLMOutputError


LLM_ERRORS = _llm_errors()


def fixed_conditions():
    """Turn off what would differ between machines or models but is not the model under test."""
    from matheasy.shortlister import name_groups
    from matheasy.tutor import runner
    name_groups.GROUPS = ROOT / "no-name-groups.json"  # does not exist: every name stands alone
    name_groups.group_of.cache_clear()
    runner.LLMFeedbackJudge = lambda session: runner.keyword_judge
    runner.LLMAttackLine = lambda session: None


def case_key(before, after, source=""):
    return f"{source}|{before}|{after}"


def pick_test(session, ranker, spacing):
    from eedi_pick import held_back_cases, mapped_cases
    from matheasy.diagnosis import EEDI_K, locate
    from matheasy.diagnosis.eedi import options
    from matheasy.diagnosis.explain import explain
    from matheasy.diagnosis.solve_back import solve_back
    out = {}
    for c in list(held_back_cases()) + list(mapped_cases()):
        located = locate(c["before"], c["after"])
        offered = options(located, ranker, k=EEDI_K)
        try:
            e = explain(located, question=f"Work out {c['before']}", session=session,
                        pieces=solve_back(c["before"], c["after"]), options=offered)
            pick, failed = e.eedi, False
        except LLM_ERRORS:  # a reply that is not the JSON asked for: counted as wrong
            pick, failed = None, True
        right = bool(pick) and pick in c["truth"]
        out[case_key(c["before"], c["after"], c["source"])] = {
            "right": right, "in_list": any(m in c["truth"] for m, _ in offered), "picked": bool(pick),
            "failed": failed}
        print(f"  pick   {c['source'][:40]:40} {'RIGHT' if right else 'wrong'}", flush=True)
        time.sleep(spacing)
    return out


def topic_test(session, spacing):
    from matheasy.diagnosis import diagnose
    from topic_accuracy import HAND, rule_cases
    from matheasy.diagnosis.explain import rule_topic
    out = {}
    for before, after, topic in rule_cases() + HAND:
        if diagnose(before, after).status != "unknown":
            continue  # a rule explains it: the LLM is never asked, so both models score the same
        try:
            d = diagnose(before, after, question=f"Work out {before}", session=session)
        except LLM_ERRORS:  # a reply that is not the JSON asked for: counted as wrong
            out[case_key(before, after)] = {"right": False, "extra": 0, "failed": True}
            print(f"  topic  {before:28} FAILED", flush=True)
            continue
        if d.status != "unknown":
            named = {rule_topic(r) for r in d.beliefs}
        elif d.explanation is not None:
            named = set(d.explanation.topics) - {"other"}
        else:
            named = set()
        out[case_key(before, after)] = {"right": topic in named, "extra": len(named - {topic})}
        print(f"  topic  {before:28} {'RIGHT' if topic in named else 'wrong'}", flush=True)
        time.sleep(spacing)
    return out


def tutor_students():
    from matheasy.evaluation import adversarial_roster, roster
    from matheasy.tutor.runner import RELAPSE_RATE, SLIP_ONLY, PER_ATTACK, keyword_judge
    students = roster(keyword_judge, slip_only=SLIP_ONLY, relapse_rate=RELAPSE_RATE)
    chosen = [s for s in students if s.group != "has_rule"]
    chosen += [s for s in students if s.group == "has_rule"][::EVERY]
    chosen += adversarial_roster(keyword_judge, per_attack=PER_ATTACK)[::EVERY]
    return chosen


def tutor_test(client, config_hash, ranker, work):
    from matheasy.evaluation.log_fields import read_jsonl
    from matheasy.evaluation.metrics import grouped_sessions
    from matheasy.evaluation.stats import per_student, units
    from matheasy.tutor.runner import keyword_judge, run_condition
    students = [type(s).from_state(s.to_state(), judge=keyword_judge) for s in tutor_students()]
    result = run_condition("full", students, work, sessions=1, shard="0of1", client=client,
                           config_hash=config_hash, retrieve=None, eedi_ranker=ranker)
    print(f"  tutor  {result['ran']} sessions run, {result['skipped']} already done", flush=True)
    if result["stopped"]:
        sys.exit(f"\nStopped: the model is unavailable ({result['stopped']}). Run the same command again later.")
    grouped = grouped_sessions(read_jsonl([work / "turns"])).get("full", {})
    out = {}
    for group in TUTOR_GROUPS:
        ss = grouped.get(group, [])
        computed = per_student(ss)
        out[group] = {m: units(ss, m, computed=computed).by_student for m in TUTOR_METRICS}
    return out


def call_stats(log_dir: Path) -> dict:
    """From the LLM call logs: calls, errors, replies that were empty or not valid JSON, speed."""
    calls = errors = bad = 0
    latencies, starts = [], []
    for f in sorted(log_dir.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("event") == "error":
                errors += 1
            if r.get("event") != "call":
                continue
            calls += 1
            text = r.get("response")
            if not text:
                bad += 1
            elif r.get("json_mode"):
                try:
                    json.loads(text)
                except json.JSONDecodeError:
                    bad += 1
                    starts.append(text[:10])  # 10 characters show the format; too few to hold a name
            if not r.get("cached"):
                latencies.append(r["latency_ms"])
    latencies.sort()
    return {"calls": calls, "errors_retried": errors, "empty_or_bad_json": bad, "fresh_calls": len(latencies),
            "median_latency_ms": latencies[len(latencies) // 2] if latencies else None,
            "bad_json_starts": starts[:5]}


def run(args):
    from matheasy.diagnosis import load_ranker
    from matheasy.llm import LLMClient, load_config
    fixed_conditions()
    config = load_config()
    work = Path(args.work) if args.work else ROOT / "runs" / f"llm-compare-{config.provider}"
    config = dataclasses.replace(config, log_dir=work / "llm")
    model = f"{config.provider}:{config.model()}"
    print(f"model {model}, config {config.hash}, work in {work}", flush=True)
    client = LLMClient(config, run_id="llm-compare")
    session = client.session("llm-compare", "eval", budget_tokens=None)
    spacing = SPACING_S if config.provider == "gemini" else 0.0
    tests = args.tests.split(",")
    ranker = None
    if {"pick", "tutor"} & set(tests):
        ranker = load_ranker()
        if ranker is None:
            sys.exit("the shortlister did not load (needs torch, a GPU and data/shortlister/adapter-3B-v2)")
    scores = {"model": model, "config_hash": config.hash}
    if "pick" in tests:
        scores["pick"] = pick_test(session, ranker, spacing)
    if "topics" in tests:
        scores["topics"] = topic_test(session, spacing)
    if "tutor" in tests:
        scores["tutor"] = tutor_test(client, config.hash, ranker, work)
    scores["calls"] = call_stats(work / "llm")
    Path(args.out).write_text(json.dumps(scores, indent=1))
    print(f"saved {args.out}: {json.dumps(scores['calls'])}")


def compare(args):
    from matheasy.evaluation.stats import Units, compare as paired, holm
    a, b = (json.loads(Path(p).read_text()) for p in (args.a, args.b))
    na, nb = a["model"].split(":")[-1].split("/")[-1], b["model"].split(":")[-1].split("/")[-1]
    rows = []
    for test in ("pick", "topics"):
        if test in a and test in b:
            ua = Units(f"{test}: right", {k: (float(v["right"]), 1) for k, v in a[test].items()})
            ub = Units(f"{test}: right", {k: (float(v["right"]), 1) for k, v in b[test].items()})
            rows.append(("cases", paired(ua, ub, a=na, b=nb, seed=20261114)))
    if "tutor" in a and "tutor" in b:
        for group in TUTOR_GROUPS:
            for metric in TUTOR_METRICS:
                ua = Units(metric, {k: tuple(v) for k, v in a["tutor"][group][metric].items()})
                ub = Units(metric, {k: tuple(v) for k, v in b["tutor"][group][metric].items()})
                c = paired(ua, ub, a=na, b=nb, seed=20261114)
                if c.students:
                    rows.append((group, c))
    adjusted = holm([c.p_value for _, c in rows])
    f = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    print(f"a = {a['model']}\nb = {b['model']}\n")
    print(f"{'group':<14}{'metric':<24}{'a':>7}{'b':>7}{'a - b':>8}  95% CI            {'p':>6}{'p Holm':>8}  n")
    for (group, c), p in zip(rows, adjusted):
        print(f"{group:<14}{c.metric:<24}{f(c.estimate_a):>7}{f(c.estimate_b):>7}{f(c.difference):>8}  "
              f"[{f(c.ci[0])}, {f(c.ci[1])}]  {f(c.p_value):>6}{f(p):>8}  {c.students}")
    print(f"\ncalls  a: {json.dumps(a['calls'])}\n       b: {json.dumps(b['calls'])}")
    print("\nleak, over-claiming and unknowable-claim rates: lower is better.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run the tests on the model the LLM config names")
    r.add_argument("--out", required=True, help="scores file (no Eedi data in it)")
    r.add_argument("--tests", default="pick,topics,tutor")
    r.add_argument("--work", help="logs and turn records (default runs/llm-compare-<provider>)")
    c = sub.add_parser("compare", help="paired comparison of two scores files")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    main()
