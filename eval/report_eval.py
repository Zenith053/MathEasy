"""The tests behind the project report (docs/report_prompt.md and docs/report_prompt_v2.md).

One script, one sub-command per part. Every result goes under runs/report/ (git-ignored:
some results hold Eedi IDs). Calls at temperature 0 are cached, so a run stopped by the
free tier's daily quota carries on where it stopped when run again.

  sessions    the tutor conditions on simulated students (Part 1 and the session ablations)
  format      input reading, transcribe(): code only / LLM only / as shipped   (Part 2.1)
  diagnosis   diagnose(): code only / LLM only / as shipped, Eedi off and on    (Part 2.2)
  slips       the slip and number-fact checks on and off, no LLM               (Part 2.2, A8)
  hinter      the hinter told the right, no or a wrong diagnosis; the rewrite loop (Part 2.3, A10)
  critic      the critic on hints with known verdicts; consistency with the cache off (Part 2.4)

    PYTHONPATH=. .venv/bin/python eval/report_eval.py sessions --offline
    GEMINI_API_KEY=... PYTHONPATH=. .venv/bin/python eval/report_eval.py sessions --conditions full,b0
    GEMINI_API_KEY=... PYTHONPATH=. .venv/bin/python eval/report_eval.py diagnosis [--eedi]

Students in the session runs judge hints with the keyword judge and adversarial students use
their template attacks (as eval/llm_compare.py does): the students stay the same whichever
tutor runs, and no LLM grades another LLM's output inside a session. The ablations are applied
here, by patching, so the tutor's own code is unchanged:
  nodiag      the full tutor, but neither the hint nor the analysis is told the misconception
              (the step is still checked and located, and memory still records it)
  nollmdiag   the full tutor with the diagnoser's LLM off (the runner's --no-llm-diagnosis)
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval"), str(ROOT / "tests")]
OUT = ROOT / "runs" / "report"
JUDGE_MODEL = "gemini-3-flash-preview"  # gemini-2.5-pro is retired for new keys; Pro models have no free quota
GAP_S = 4.2  # the free tier allows 15 requests a minute


# --------------------------------------------------------------------------- the LLM client

class Throttled:
    """The backend, at most one fresh request every GAP_S seconds (cached replies are free)."""

    def __init__(self, inner, gap: float = GAP_S):
        self.inner, self.gap, self.last = inner, gap, 0.0

    def generate(self, request):
        wait = self.last + self.gap - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            return self.inner.generate(request)
        finally:
            self.last = time.time()

    def __getattr__(self, name):
        return getattr(self.inner, name)


def make_client(name: str, *, cache: bool = True):
    from matheasy.llm import LLMClient, load_config
    config = load_config()
    config = dataclasses.replace(config, log_dir=OUT / name / "llm", cache_enabled=cache,
                                 models={**config.models, "large": JUDGE_MODEL})
    client = LLMClient(config, run_id=name)
    client.backend = Throttled(client.backend)
    return client


def save(name: str, data) -> Path:
    path = OUT / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def call_stats(name: str) -> dict:
    """Calls, cache hits, errors, tokens and median latency from this part's LLM logs."""
    calls = cached = errors = tokens = 0
    latencies, roles = [], Counter()
    for f in sorted((OUT / name / "llm").glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("event") == "error":
                errors += 1
            if r.get("event") != "call":
                continue
            calls += 1
            roles[r.get("role")] += 1
            tokens += (r.get("input_tokens") or 0) + (r.get("output_tokens") or 0) + (r.get("thought_tokens") or 0)
            if r.get("cached"):
                cached += 1
            elif r.get("latency_ms") is not None:
                latencies.append(r["latency_ms"])
    latencies.sort()
    return {"calls": calls, "cached": cached, "fresh": calls - cached, "errors": errors, "tokens": tokens,
            "median_latency_ms": latencies[len(latencies) // 2] if latencies else None, "by_role": dict(roles)}


def unavailable():
    from matheasy.llm import LLMUnavailable
    return LLMUnavailable


# --------------------------------------------------------------------------- Part 1 and 3: sessions

SESSION_CONDITIONS = ("full", "b2", "b0", "b1", "nodiag", "nollmdiag")


def fixed_students(sample: str):
    """'all': D's roster, slip-only students and 6 adversarial students per attack.
    'llm': every student the rules cannot read, every 8th with a rule, and one per attack."""
    from matheasy.evaluation import adversarial_roster, roster
    from matheasy.tutor.runner import PER_ATTACK, RELAPSE_RATE, SLIP_ONLY, keyword_judge
    students = roster(keyword_judge, slip_only=SLIP_ONLY, relapse_rate=RELAPSE_RATE)
    attackers = adversarial_roster(keyword_judge, per_attack=PER_ATTACK)
    if sample == "all":
        return students + attackers
    chosen = [s for s in students if s.group != "has_rule"]
    chosen += [s for s in students if s.group == "has_rule"][::8]
    chosen += attackers[::PER_ATTACK]
    return chosen


@contextlib.contextmanager
def condition_patches(condition: str):
    """The session ablations, applied for one condition's run only."""
    from matheasy.tutor import orchestrator, runner
    saved = {"make_tutor": runner.make_tutor, "WITH_MEMORY": runner.WITH_MEMORY,
             "hint_request": orchestrator.hint_request, "analyse": orchestrator.analyse}
    real_make_tutor = runner.make_tutor

    def make_tutor(cond, log, store, **kw):
        if cond == "nollmdiag":
            kw["llm_diagnosis"] = False
        return real_make_tutor("full" if cond in ("nodiag", "nollmdiag") else cond, log, store, **kw)

    runner.make_tutor = make_tutor
    runner.WITH_MEMORY = (*runner.WITH_MEMORY, "nodiag", "nollmdiag")
    if condition == "nodiag":
        real_request, real_analyse = orchestrator.hint_request, orchestrator.analyse

        def no_diagnosis_request(problem, answer, diagnosis, **kw):
            return dataclasses.replace(real_request(problem, answer, diagnosis, **kw), misconception=None,
                                       certainty=None)

        orchestrator.hint_request = no_diagnosis_request
        orchestrator.analyse = lambda diagnosis, *a, **k: real_analyse(None, *a, **k)
    try:
        yield
    finally:
        runner.make_tutor, runner.WITH_MEMORY = saved["make_tutor"], saved["WITH_MEMORY"]
        orchestrator.hint_request, orchestrator.analyse = saved["hint_request"], saved["analyse"]


def sessions(args):
    from matheasy.llm import load_config
    from matheasy.tutor import runner
    from matheasy.tutor.runner import keyword_judge, run_condition
    from matheasy.tutor.__main__ import textbook_retrieve
    runner.LLMFeedbackJudge = lambda session: keyword_judge  # the students stay fixed (see the docstring)
    runner.LLMAttackLine = lambda session: None
    name = "sessions-offline" if args.offline else "sessions-llm"
    work = OUT / name
    client = None if args.offline else make_client(name)
    config_hash = load_config().hash
    sample = "all" if args.offline else args.sample
    sessions_per = 2 if args.offline else args.sessions
    retrieve = textbook_retrieve()
    summary = {}
    for condition in args.conditions.split(","):
        if condition not in SESSION_CONDITIONS:
            raise SystemExit(f"unknown condition {condition}")
        if condition in ("b0", "b1") and client is None:
            print(f"{condition}: needs the LLM, skipped offline", flush=True)
            continue
        students = [type(s).from_state(s.to_state(), judge=keyword_judge) for s in fixed_students(sample)]
        t = time.time()
        with condition_patches(condition):
            result = run_condition(condition, students, work, sessions=sessions_per, shard="0of1", client=client,
                                   config_hash=config_hash, retrieve=retrieve)
        summary[condition] = {"ran": result["ran"], "skipped": result["skipped"], "stopped": result["stopped"],
                              "seconds": round(time.time() - t, 1)}
        print(f"{condition}: {summary[condition]}", flush=True)
        if result["stopped"]:
            print("Stopped: the model is unavailable (daily quota?). Run the same command again later.", flush=True)
            break
    summary["calls"] = call_stats(name) if client else None
    save(f"{name}-summary", summary)


# --------------------------------------------------------------------------- Part 2.1: input reading

# (question, the student's line, kind, intended readings). "not_maths" lines have no reading.
FORMAT_CASES = [
    # standard notation: read as typed, and an LLM must not "correct" the student's mistake
    ("1/2 + 1/3", "2/5", "standard", ["2/5"]),
    ("Simplify x^2 × x^3", "x^6", "standard", ["x^6"]),
    ("Expand 3(x + 4)", "3x + 4", "standard", ["3x + 4"]),
    ("Simplify (3x)^2", "3x^2", "standard", ["3x^2"]),
    ("Expand (x + 3)^2", "x^2 + 9", "standard", ["x^2 + 9"]),
    ("0.5 + 0.25", "0.30", "standard", ["0.30"]),
    ("Solve 2x + 3 = 11", "2x = 14", "standard", ["2x = 14"]),
    ("Expand -2(x - 3)", "-2x - 6", "standard", ["-2x - 6"]),
    ("1/2 + 1/3 + 1/4", "5/6 + 1/4", "standard", ["5/6 + 1/4"]),
    ("Simplify 2x + 3x", "5x^2", "standard", ["5x^2"]),
    ("Work out 25% of 80", "25% × 80", "standard", ["25% × 80"]),
    ("Simplify a/b × 1/c", "a/bc", "standard", ["a/bc"]),
    # ambiguous notation: the reading needs the student's meaning
    ("Simplify a^b × a^c", "a^bc", "ambiguous", ["a^(bc)"]),
    ("Simplify 2^m × 2^n", "2^mn", "ambiguous", ["2^(mn)"]),
    ("Simplify x^2 × y", "x^2y", "ambiguous", ["x^2 × y"]),
    ("Simplify x^2 × x^3", "x^2x^3", "ambiguous", ["x^2 × x^3"]),
    ("Divide x by 2y", "x/2y", "ambiguous", ["x/(2y)"]),
    ("Divide 3 by 4y", "3/4y", "ambiguous", ["3/(4y)"]),
    ("Find half of x", "1/2x", "ambiguous", ["1/2 × x", "x/2"]),
    ("Find a third of x", "1/3x", "ambiguous", ["1/3 × x", "x/3"]),
    ("Write the cube root of x squared as a power", "x^2/3", "ambiguous", ["x^(2/3)"]),
    ("Divide x squared by 3", "x^2/3", "ambiguous", ["x^2 ÷ 3"]),
    ("Divide e squared by a", "e^2/a", "ambiguous", ["e^2 ÷ a"]),
    ("Simplify y^3 × y^4", "y^34", "standard", ["y^34"]),
    # runs of letters: a word, or letters multiplied
    ("Simplify x × y × z × w + 2", "xyzw + 2", "letters", ["x × y × z × w + 2"]),
    ("Fuel for 3 days is 45 litres", "fuel × 3 = 45", "letters", ["{fuel} × 3 = 45"]),
    ("Write the cost of n pens", "cost × n", "letters", ["{cost} × n"]),
    ("Simplify a × b × c × d × 2", "abcd × 2", "letters", ["a × b × c × d × 2"]),
    ("Simplify p × q × r × s - 1", "pqrs - 1", "letters", ["p × q × r × s - 1"]),
    ("The price of petrol rose by 5", "petrol + 5", "letters", ["{petrol} + 5"]),
    ("Simplify 2 × w × x × y × z", "2wxyz", "letters", ["2 × w × x × y × z", "2wxyz"]),
    ("Each box holds 12 eggs. How many eggs in b boxes?", "eggs = 12b", "letters", ["{eggs} = 12b"]),
    # not maths: the student is asked to type the answer as maths
    ("1/2 + 1/3", "hello", "not_maths", []),
    ("Solve 2x + 3 = 11", "I don't know", "not_maths", []),
    ("1/2 + 1/3", "five sixths", "not_maths", []),
    ("3 1/2 × 2", "three and a half times two", "not_maths", []),
    ("Simplify x^2 × x^3", "help", "not_maths", []),
    ("1/2 + 1/3", "what is the answer?", "not_maths", []),
    ("Solve 2x = 14", "x is seven", "not_maths", []),
    ("0.5 + 0.25", "zero point seven five", "not_maths", []),
]


def same_reading(a: str, b: str) -> bool:
    from matheasy.diagnosis import equivalent, parse
    if a.replace(" ", "") == b.replace(" ", ""):
        return True
    try:
        parse(a), parse(b)
        return bool(equivalent(a, b)) and _same_letters(a, b)
    except Exception:
        return False


def _same_letters(a: str, b: str) -> bool:
    import re
    return sorted(re.findall(r"[0-9]+|[a-z]", a)) == sorted(re.findall(r"[0-9]+|[a-z]", b))


def format_outcome(t, kind: str, intended: list[str]) -> str:
    """hit / asked (unsure, the intended reading offered) / miss / abstain."""
    if kind == "not_maths":
        return "hit" if t.status in ("not_maths", "unreadable") else "miss"
    if t.status == "ok":
        return "hit" if any(same_reading(t.line, i) for i in intended) else "miss"
    if t.status == "unsure":
        offered = [c for c in (t.choices or ()) if c] + ([t.line] if t.line else [])
        return "asked" if any(same_reading(c, i) for c in offered for i in intended) else "miss"
    if t.status == "unfaithful":
        return "abstain"  # code refused the model's rewrite: the student is asked again
    return "abstain"


def llm_only_transcribe(line: str, question: str, session):
    """Every line through the model, standard or not: the 'LLM only' variant."""
    from matheasy.diagnosis.maths import unknown_runs
    from matheasy.diagnosis.transcribe import Transcription, _apply, _classify, _parses, _rephrase
    from matheasy.diagnosis import notation_issue
    text = line.strip()
    runs = unknown_runs(text) if _parses(text) else []
    if runs:
        kinds = _classify(text, runs, question, session)
        if kinds is None:
            return Transcription("unreadable", None, line, used_llm=True)
        text = _apply(text, kinds)
        if notation_issue(text) is None:
            return Transcription("ok", text, line, used_llm=True)
    return _rephrase(text, line, question, session)


def format_test(args):
    from matheasy.diagnosis import transcribe
    client = make_client("format")
    session = client.session("format", "eval", budget_tokens=None)
    rows = []
    for question, line, kind, intended in FORMAT_CASES:
        row = {"question": question, "line": line, "kind": kind, "intended": intended}
        for variant in ("code", "llm_only", "shipped"):
            t0 = time.time()
            try:
                if variant == "code":
                    t = transcribe(line, question=question, session=None)
                elif variant == "llm_only":
                    t = llm_only_transcribe(line, question, session)
                else:
                    t = transcribe(line, question=question, session=session)
            except unavailable() as e:
                save("format", {"rows": rows, "stopped": str(e)})
                raise SystemExit(f"stopped: {e}")
            except Exception as e:  # an LLM reply code could not use
                row[variant] = {"outcome": "abstain", "status": "error", "error": str(e)[:200]}
                continue
            row[variant] = {"outcome": format_outcome(t, kind, intended), "status": t.status, "line": t.line,
                            "choices": list(t.choices or ()), "used_llm": t.used_llm,
                            "seconds": round(time.time() - t0, 2)}
        rows.append(row)
        print(f"{line!r:32} " + "  ".join(f"{v}={row[v]['outcome']}" for v in ("code", "llm_only", "shipped")),
              flush=True)
    save("format", {"rows": rows, "calls": call_stats("format")})


# --------------------------------------------------------------------------- Part 2.2: diagnosis

def rule_steps():
    """(rule name, before, after, topic) from each rule's worked examples, inside a longer line
    (as eval/topic_accuracy.py builds them), plus its hand cases with a topic only."""
    from test_diagnosis_rule_library import PREDICTS
    from topic_accuracy import HAND
    from matheasy.diagnosis import check_step, parse
    from matheasy.diagnosis.explain import rule_topic
    from matheasy.diagnosis.maths import Rel, _kinds
    from matheasy.diagnosis.rules._common import decimal_node, evaluate, fraction_node
    from matheasy.diagnosis.solve_back import _operations
    cases = []
    for rule, problem, answer in PREDICTS:
        before, after = problem, answer
        if not isinstance(parse(problem), Rel):
            before = f"{problem} + 5 - 3"
            value = evaluate(parse(answer))
            if value is None or _operations(parse(answer)):
                after = f"{answer} + 2"
            else:
                number = decimal_node if "decimals" in _kinds(parse(problem)) else fraction_node
                after = str(number(value + 2))
        try:
            if check_step(before, after):
                continue
        except Exception:
            continue
        cases.append({"set": "rules", "rule": rule, "before": before, "after": after, "topic": rule_topic(rule)})
    cases += [{"set": "hand", "rule": None, "before": b, "after": a, "topic": t} for b, a, t in HAND]
    return cases


def eedi_steps(split: str = "test"):
    """Eedi's own wrong answers in our notation (data/eedi/live_queries.csv), with Eedi's label."""
    import pandas as pd
    from matheasy.diagnosis import check_step, locate
    rows = pd.read_csv(ROOT / "data" / "eedi" / "live_queries.csv")
    out = []
    for r in rows[rows.Split == split].itertuples():
        parts = dict(p.split(": ", 1) for p in r.Query.split(" | ") if ": " in p)
        before, after = parts.get("Before"), parts.get("Student wrote")
        try:
            if not before or not after or check_step(before, after) or locate(before, after) is None:
                continue
        except Exception:
            continue
        out.append({"set": "eedi", "id": f"q{r.QuestionId}{r.WrongAnswer}", "before": before, "after": after,
                    "eedi": int(r.MisconceptionId), "topic": r.Topic})
    return out


def accepted_for(case) -> set[str]:
    from matheasy.diagnosis import load_bank
    from matheasy.diagnosis.ids import eedi_id, rule_id
    from matheasy.evaluation.metrics import accepted
    from matheasy.shortlister.name_groups import members
    if case["set"] == "eedi":
        return {eedi_id(m) for m in members(case["eedi"])} | set(load_bank().for_eedi(case["eedi"]))
    if case.get("rule"):
        return accepted(rule_id(case["rule"]), "has_rule")
    return set()


def score_diagnosis(d, case) -> dict:
    """hit / miss / abstain on the misconception; topic hit; the trace stages that decided."""
    ok = accepted_for(case)
    named = d.misconception
    if ok:
        outcome = "abstain" if named is None else ("hit" if named in ok else "miss")
    else:
        outcome = None
    e = d.explanation
    return {"outcome": outcome, "status": d.status, "named": named, "tied": list(d.beliefs) if d.status == "ambiguous" else [],
            "topic_hit": case["topic"] in d.topics if case["set"] != "eedi" else None, "topics": list(d.topics),
            "probe": d.probe.question if d.probe else None,
            "llm": e is not None, "explanation_status": e.status if e else None,
            "eedi_pick": e.eedi if e else None, "eedi_fit": getattr(e, "eedi_fit", None) if e else None,
            "kind": e.kind if e else None}


def after_probe(case, d):
    """Answer the probe as the true rule predicts (no student), then diagnose again."""
    from matheasy.diagnosis import FollowUp, diagnose
    from matheasy.diagnosis.ids import rule_id
    if d.probe is None or not case.get("rule"):
        return None
    answer = d.probe.predictions.get(rule_id(case["rule"]))
    if answer is None:
        return None
    d2 = diagnose(case["before"], case["after"], follow_ups=[FollowUp(d.probe, answer)])
    if d2.probe is not None:
        answer2 = d2.probe.predictions.get(rule_id(case["rule"]))
        if answer2 is not None:
            d2 = diagnose(case["before"], case["after"],
                          follow_ups=[FollowUp(d.probe, answer), FollowUp(d2.probe, answer2)])
    return d2


def diagnosis_test(args):
    from matheasy.diagnosis import diagnose, locate
    from matheasy.diagnosis.eedi import options as eedi_options
    from matheasy.diagnosis.explain import explain
    ranker = None
    if args.eedi:
        from matheasy.diagnosis import load_ranker
        print("loading the Eedi shortlister...", flush=True)
        ranker = load_ranker()
        if ranker is None:
            raise SystemExit("no GPU or adapter: the Eedi variants need the shortlister")
    name = "diagnosis-eedi" if args.eedi else "diagnosis"
    client = make_client(name)
    session = client.session(name, "eval", budget_tokens=None)
    cases = (eedi_steps() if args.eedi else rule_steps() + eedi_steps())
    results = []
    try:
        for i, case in enumerate(cases):
            q = f"Work out {case['before']}"
            row = dict(case)
            row["accepted"] = sorted(accepted_for(case))
            if not args.eedi:
                t0 = time.time()
                d = diagnose(case["before"], case["after"], question=q)
                row["code"] = score_diagnosis(d, case) | {"seconds": round(time.time() - t0, 3)}
                d2 = after_probe(case, d)
                row["code_after_probe"] = score_diagnosis(d2, case) if d2 is not None else None
                t0 = time.time()
                d = diagnose(case["before"], case["after"], question=q, session=session)
                row["shipped"] = score_diagnosis(d, case) | {"seconds": round(time.time() - t0, 3)}
                if case["set"] != "eedi" and i % 3 == 0:  # LLM only, on every 3rd rule step: the topic
                    e = explain(locate(case["before"], case["after"]), question=q, session=session)
                    row["llm_only"] = {"topic_hit": case["topic"] in e.topics, "topics": list(e.topics),
                                       "explanation_status": e.status, "kind": e.kind}
            else:
                t0 = time.time()
                d = diagnose(case["before"], case["after"], question=q, session=session, eedi_ranker=ranker)
                row["shipped_eedi"] = score_diagnosis(d, case) | {"seconds": round(time.time() - t0, 3)}
                located = locate(case["before"], case["after"])
                choices = eedi_options(located, ranker)
                e = explain(located, question=q, session=session, options=choices)
                ok = accepted_for(case)
                exact = {f"eedi:{case['eedi']}"}
                row["llm_only_eedi"] = {
                    "outcome": "abstain" if e.eedi is None else ("hit" if e.eedi in ok else "miss"),
                    "exact": e.eedi in exact, "pick": e.eedi, "fit": getattr(e, "eedi_fit", None),
                    "explanation_status": e.status, "offered_has_truth": any(m in ok for m, _ in choices)}
            results.append(row)
            if i % 10 == 0:
                print(f"{i + 1}/{len(cases)}", flush=True)
                save(name, {"rows": results, "partial": True})
    except unavailable() as e:
        save(name, {"rows": results, "stopped": str(e), "calls": call_stats(name)})
        raise SystemExit(f"stopped after {len(results)} cases: {e}")
    save(name, {"rows": results, "calls": call_stats(name)})


# --------------------------------------------------------------------------- slip checks (A8)

SLIP_CASES = [  # (question, working, is it a slip?)
    ("1/2 + 1/3 + 1/4", "5/6 + 1/4\n6/10", True),
    ("x^2 × x^3 + x^4 × x^2", "x^5 + x^8", True),
    ("-2(x + y - z - k)", "-2x - 2y + 2z - 2k", True),
    ("3(x + 2) + 2(x + 5)", "3x + 6 + 2x + 5", True),
    ("6 + 3", "8", True),
    ("7 × 8 + 6", "54 + 6\n60", True),
    ("2x + 6 + 3", "2x + 8", True),
    ("x^2 × x^3 + x^4 × x^2 + x^3 × x^3", "x^5 + x^6 + x^9", True),
    ("1/2 + 1/3", "2/5", False),
    ("1/2 + 1/3 + 1/4", "2/5 + 1/4\n3/9", False),
    ("x^2 × x^3", "x^6", False),
    ("3(x + 4)", "3x + 4", False),
    ("-2(x + y - z - k)", "-2x - 2y - 2z - 2k", False),
    ("2(x + 3) + 4(x + 1)", "2x + 3 + 4x + 1", False),
    ("47 + 38", "75", False),
    ("x + x", "x^2", False),
]


@contextlib.contextmanager
def slip_checks_off():
    from matheasy.diagnosis import diagnose as diagnose_module
    from matheasy.tutor import planner
    from matheasy.ui import __main__ as page
    mod = sys.modules["matheasy.diagnosis.diagnose"]
    saved = (mod.bracket_slip, mod.number_fact, page.uses_slip, planner.is_slip)
    mod.bracket_slip = lambda *a, **k: None
    mod.number_fact = lambda *a, **k: None
    page.uses_slip = lambda *a, **k: None
    planner.is_slip = lambda *a, **k: False
    try:
        yield
    finally:
        mod.bracket_slip, mod.number_fact, page.uses_slip, planner.is_slip = saved


def slips_test(args):
    import tempfile
    from matheasy.ui.__main__ import Tutor
    from matheasy.ui.db import Database
    rows = []
    for variant in ("on", "off"):
        db = Database(Path(tempfile.mkdtemp()) / "t.db")
        db.add_user("eval", "eval12345")
        tutor = Tutor(llm=False, eedi=False, db=db)
        ctx = slip_checks_off() if variant == "off" else contextlib.nullcontext()
        with ctx:
            for i, (q, working, is_slip) in enumerate(SLIP_CASES):
                db.add_user(f"s{variant}{i}", "eval12345")  # a fresh student each: no memory between cases
                r = tutor.analyze(q, working, f"s{variant}{i}")
                said = bool(r.get("slip"))
                rows.append({"variant": variant, "question": q, "working": working, "gold_slip": is_slip,
                             "said_slip": said, "hit": said == is_slip, "misconception": r.get("misconception"),
                             "why_slip": r.get("why_slip")})
    save("slips", {"rows": rows})
    for v in ("on", "off"):
        rs = [r for r in rows if r["variant"] == v]
        print(v, sum(r["hit"] for r in rs), "/", len(rs))


# --------------------------------------------------------------------------- Part 2.3: hinter

JUDGE_SYSTEM = """\
You check one hint written by a maths tutor for a student aged 11 to 14. You are told the
question, the correct answer, what the student wrote, and the student's actual mistake. Judge
only the hint. Answer each question with true or false, and give one short reason."""
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "maths_correct": {"type": "boolean", "description": "every maths statement in the hint is correct"},
        "no_answer": {"type": "boolean", "description": "it does not state or work out the final answer"},
        "not_doing_the_step": {"type": "boolean",
                               "description": "it does not do the wrong step for the student (asking them to is fine)"},
        "targets_the_mistake": {"type": "boolean", "description": "it is aimed at the student's actual mistake"},
        "age_appropriate": {"type": "boolean", "description": "its words suit a student aged 11 to 14"},
        "reason": {"type": "string"},
    },
    "required": ["maths_correct", "no_answer", "not_doing_the_step", "targets_the_mistake", "age_appropriate",
                 "reason"],
}
CRITERIA = ("maths_correct", "no_answer", "not_doing_the_step", "targets_the_mistake", "age_appropriate")


def hinter_cases(n: int = 24):
    """One worked example per rule, round-robin over topics, that the code diagnoses as that rule."""
    from test_diagnosis_rule_library import PREDICTS
    from matheasy.diagnosis import diagnose
    from matheasy.diagnosis.explain import rule_topic
    from matheasy.diagnosis.ids import rule_id
    by_topic, seen = defaultdict(list), set()
    for rule, problem, answer in PREDICTS:
        if rule in seen:
            continue
        try:
            d = diagnose(problem, answer)
        except Exception:
            continue
        if d.status == "clear" and d.misconception == rule_id(rule):
            seen.add(rule)
            by_topic[rule_topic(rule)].append((rule, problem, answer, d))
    cases, topics = [], sorted(by_topic)
    while len(cases) < n and any(by_topic.values()):
        for t in topics:
            if by_topic[t] and len(cases) < n:
                cases.append(by_topic[t].pop(0))
    return cases


def judge(session, problem, request, true_mistake: str, text: str) -> dict:
    content = "\n".join([f"Question: {problem.question}", f"Correct answer: {problem.correct_answer}",
                         f"Student wrote: {request.student_answer}", f"The student's actual mistake: {true_mistake}",
                         f"Hint:\n{text}"])
    from matheasy.llm import LLMOutputError
    try:  # the judge model thinks before answering: give it room, or the JSON is cut off
        r = session.call("judge", [{"role": "user", "content": content}], system=JUDGE_SYSTEM, model="large",
                         json_schema=JUDGE_SCHEMA, max_output_tokens=8192)
    except LLMOutputError as e:
        return {k: None for k in CRITERIA} | {"reason": f"judge error: {e}"[:200]}
    return {k: r.parsed.get(k) for k in (*CRITERIA, "reason")}


def code_checks(hint, problem) -> dict:
    from matheasy.tutor.critic import MAX_HINT_CHARS, leaks_answer
    return {"no_leak": not leaks_answer(hint.text, problem.correct_answer, problem.question),
            "cites": bool(hint.citations), "length_ok": 0 < len(hint.text) <= MAX_HINT_CHARS}


def hinter_test(args):
    from matheasy.diagnosis.ids import rule_id
    from matheasy.llm import LLMOutputError
    from matheasy.tutor.__main__ import textbook_retrieve
    from matheasy.tutor.critic import TutorCritic
    from matheasy.tutor.describe import misconception_text
    from matheasy.tutor.hinter import LLMHinter
    from matheasy.tutor.hinting import hint_request, write_hint
    from matheasy.tutor.placeholders import TemplateHinter
    from matheasy.tutor.session import make_problem
    client = make_client("hinter")
    session = client.session("hinter", "eval", budget_tokens=None)
    retrieve = textbook_retrieve()
    hinter, template, critic = LLMHinter(retrieve), TemplateHinter(), TutorCritic()
    cases = hinter_cases(args.n)
    rules_by_topic = defaultdict(list)
    for rule, *_ in cases:
        from matheasy.diagnosis.explain import rule_topic
        rules_by_topic[rule_topic(rule)].append(rule)
    rows = []
    try:
        for idx, (rule, before, after, d) in enumerate(cases):
            problem = make_problem(f"h{idx}", before)
            right = hint_request(problem, after, d)
            truth = misconception_text(rule_id(rule))
            from matheasy.diagnosis.explain import rule_topic
            others = [r for r in rules_by_topic[rule_topic(rule)] if r != rule] or \
                     [r for r, *_ in cases if r != rule]
            wrong_rule = others[idx % len(others)]
            requests = {"right": right,
                        "none": dataclasses.replace(right, misconception=None, certainty=None),
                        "wrong": dataclasses.replace(right, misconception=misconception_text(rule_id(wrong_rule)))}
            row = {"rule": rule, "before": before, "after": after, "wrong_rule": wrong_rule, "truth": truth}
            for variant, request in requests.items():
                try:
                    h = hinter.hint(request, None, llm=session)
                except LLMOutputError as e:  # a reply the hinter could not use: a miss for the hinter
                    row[variant] = {"hint": None, "error": str(e)[:200]}
                    continue
                row[variant] = {"hint": h.text, "citation": h.citations[0] if h.citations else None,
                                "code": code_checks(h, problem), "judge": judge(session, problem, right, truth, h.text)}
            h = template.hint(right, None)
            row["template"] = {"hint": h.text, "code": code_checks(h, problem) | {"cites": None},
                               "judge": judge(session, problem, right, truth, h.text)}
            # A10: the shipped loop (hinter + critic), up to two rewrites; attempt 1 is the "right" hint above
            w = write_hint(hinter, critic, right, problem, llm=session, max_rewrites=2)
            row["loop"] = {"approved_at": None if w.hint is None else len(w.attempts) + 1,
                           "rejections": [a["reasons"] for a in w.attempts], "dropped": w.dropped,
                           "error": str(w.error) if w.error else None}
            rows.append(row)
            print(f"{idx + 1}/{len(cases)} {rule}: approved at {row['loop']['approved_at']}", flush=True)
            save("hinter", {"rows": rows, "partial": True})
    except unavailable() as e:
        save("hinter", {"rows": rows, "stopped": str(e), "calls": call_stats("hinter")})
        raise SystemExit(f"stopped after {len(rows)} cases: {e}")
    save("hinter", {"rows": rows, "calls": call_stats("hinter")})


# --------------------------------------------------------------------------- Part 2.4: critic

# Six problems, each with its true mistake; for each, hints with a known verdict.
CRITIC_PROBLEMS = {
    "P1": ("1/2 + 1/3", "2/5", "fractions.add_numerators_denominators"),
    "P2": ("x^2 × x^3", "x^6", "exponents.multiply_exponents_when_multiplying_powers"),
    "P3": ("3(x + 4)", "3x + 4", "brackets.multiply_first_term_only"),
    "P4": ("(x + 3)^2", "x^2 + 9", "exponents.distribute_exponent_over_addition"),
    "P5": ("0.5 + 0.25", "0.30", None),
    "P6": ("2x + 3 = 11", "2x = 14", None),
}
# (problem, label, the fault if any, hint). label: approve / reject.
CRITIC_HINTS = [
    ("P1", "approve", None, "Halves and thirds are different-sized pieces. Before you add, can you write both "
                            "fractions with the same denominator?"),
    ("P1", "approve", None, "You read both fractions correctly. Adding the tops and adding the bottoms doesn't "
                            "work for fractions: what denominator could both fractions share?"),
    ("P1", "reject", "leak", "Change both to sixths: 3/6 + 2/6 makes 5/6."),
    ("P1", "reject", "wrong_maths", "To add fractions, multiply the denominators to get 6, then add the two "
                                    "numerators 1 + 1 to get the top."),
    ("P1", "reject", "does_the_step", "First, 1/2 is 3/6 and 1/3 is 2/6. Now add those two fractions."),
    ("P1", "reject", "off_target", "Remember that when you multiply fractions, you multiply the numerators and "
                                   "multiply the denominators."),
    ("P1", "reject", "too_advanced", "Apply the additive structure of the field of rationals: map each "
                                     "summand to the least common multiple of the denominators via prime "
                                     "factorisation before summing."),
    ("P2", "approve", None, "Write x^2 and x^3 out as x's multiplied together. How many x's are there altogether?"),
    ("P2", "approve", None, "You kept the base x, which is right. When you multiply powers of the same base, "
                            "what should happen to the indices: add or multiply?"),
    ("P2", "reject", "leak", "When multiplying powers you add the indices, so the answer is x^5."),
    ("P2", "reject", "wrong_maths", "When you multiply powers of the same base, you multiply the bases and "
                                    "keep the indices the same."),
    ("P2", "reject", "does_the_step", "x^2 × x^3 is x × x × x × x × x, so count them and write the power."),
    ("P2", "reject", "off_target", "Remember that a power of a power, like (x^2)^3, means you multiply the indices."),
    ("P2", "reject", "too_advanced", "By the homomorphism from the additive group of exponents to the "
                                     "multiplicative group, the product corresponds to the sum of the exponents."),
    ("P3", "approve", None, "The 3 multiplies everything inside the bracket. Did you multiply both terms by 3?"),
    ("P3", "approve", None, "You multiplied 3 by x correctly. Look at the 4 inside the bracket: has it been "
                            "multiplied by 3 as well?"),
    ("P3", "reject", "leak", "Multiply both terms: 3 × x and 3 × 4, giving 3x + 12."),
    ("P3", "reject", "wrong_maths", "When you expand a bracket, only the first term inside is multiplied by "
                                    "the number outside."),
    ("P3", "reject", "does_the_step", "3 × 4 is 12, so change your 4 to 12."),
    ("P3", "reject", "off_target", "Check your signs: a negative number outside the bracket changes every "
                                   "sign inside."),
    ("P3", "reject", "too_advanced", "Use the distributive law of multiplication over addition in a "
                                     "commutative ring to expand the product."),
    ("P4", "approve", None, "(x + 3)^2 means (x + 3) × (x + 3). Try multiplying the two brackets out term by term."),
    ("P4", "approve", None, "You squared x and 3 correctly. Squaring a bracket isn't the same as squaring "
                            "each term: what happens when you write it as two brackets multiplied?"),
    ("P4", "reject", "leak", "Expand it as (x + 3)(x + 3) to get x^2 + 6x + 9."),
    ("P4", "reject", "wrong_maths", "Squaring a sum is the same as adding the squares, so check your "
                                    "arithmetic on 3^2."),
    ("P4", "reject", "does_the_step", "The middle terms are 3x and 3x, which make 6x. Put that between "
                                      "your two terms."),
    ("P4", "reject", "off_target", "Remember that when you multiply powers of the same base you add the indices."),
    ("P4", "reject", "too_advanced", "Apply the binomial theorem with binomial coefficients C(2, k) to "
                                     "obtain each term of the expansion."),
    ("P5", "approve", None, "Line the numbers up by their decimal points, not by their last digits. "
                            "What do you get then?"),
    ("P5", "approve", None, "Think of 0.5 as 0.50. Now both numbers have two decimal places: try adding again."),
    ("P5", "reject", "leak", "0.50 + 0.25 is 0.75."),
    ("P5", "reject", "wrong_maths", "When adding decimals, line up the last digits on the right, as you do "
                                    "with whole numbers."),
    ("P5", "reject", "does_the_step", "Write 0.5 as 0.50, then add 50 and 25 hundredths to get the hundredths."),
    ("P5", "reject", "off_target", "Remember to simplify your fraction by dividing the top and bottom by "
                                   "a common factor."),
    ("P5", "reject", "too_advanced", "Normalise both operands to a common radix exponent before performing "
                                     "the base-10 addition with carry propagation."),
    ("P6", "approve", None, "To move the 3 to the other side, you do the opposite operation. What is the "
                            "opposite of adding 3?"),
    ("P6", "approve", None, "You kept 2x on the left, which is right. To get rid of + 3, should you add 3 "
                            "or take 3 away from both sides?"),
    ("P6", "reject", "leak", "Subtract 3 from both sides, then divide by 2: x = 4."),
    ("P6", "reject", "wrong_maths", "When a number moves to the other side of the equals sign, it keeps its sign."),
    ("P6", "reject", "does_the_step", "Subtract 3 from both sides: 2x = 8. Now finish it."),
    ("P6", "reject", "off_target", "Remember that when you multiply fractions you multiply the tops and the bottoms."),
    ("P6", "reject", "too_advanced", "Apply the additive inverse in the group of integers to both sides of "
                                     "the equation to isolate the linear term."),
]


def critic_test(args):
    from matheasy.diagnosis import diagnose
    from matheasy.diagnosis.ids import rule_id
    from matheasy.tutor.critic import TutorCritic
    from matheasy.tutor.describe import misconception_text
    from matheasy.tutor.hinting import hint_request
    from matheasy.tutor.session import make_problem
    critic = TutorCritic()
    client = make_client("critic")
    session = client.session("critic", "eval", budget_tokens=None)
    fresh = make_client("critic-repeat", cache=False)  # consistency: real repeat calls, not the cache
    repeat = fresh.session("critic-repeat", "eval", budget_tokens=None)
    rows = []
    try:
        for i, (pid, label, fault, text) in enumerate(CRITIC_HINTS):
            before, after, rule = CRITIC_PROBLEMS[pid]
            problem = make_problem(pid, before)
            try:
                d = diagnose(before, after)
            except Exception:
                d = None
            request = hint_request(problem, after, d)
            if rule and request.misconception is None:
                request = dataclasses.replace(request, misconception=misconception_text(rule_id(rule)),
                                              certainty="certain")
            code = critic.code_checks(text, "hint", problem)
            llm = critic.llm_review(text, problem, request, session)
            repeats = [critic.llm_review(text, problem, request, repeat).passed for _ in range(args.repeats)]
            shipped = not code and llm.passed
            rows.append({"problem": pid, "label": label, "fault": fault, "hint": text,
                         "misconception_given": request.misconception,
                         "code_pass": not code, "code_reasons": code, "llm_pass": llm.passed,
                         "llm_reasons": list(llm.reasons), "fits": llm.misconception_fits,
                         "shipped_pass": shipped, "repeats": repeats})
            print(f"{i + 1}/{len(CRITIC_HINTS)} {pid} {label:7} {fault or '':14} code={not code} "
                  f"llm={llm.passed} repeats={repeats}", flush=True)
            save("critic", {"rows": rows, "partial": True})
    except unavailable() as e:
        save("critic", {"rows": rows, "stopped": str(e)})
        raise SystemExit(f"stopped after {len(rows)} hints: {e}")
    save("critic", {"rows": rows, "calls": call_stats("critic") | {"repeat": call_stats("critic-repeat")}})


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="part", required=True)
    s = sub.add_parser("sessions")
    s.add_argument("--offline", action="store_true", help="no LLM: the whole roster, 2 sessions (free)")
    s.add_argument("--conditions", default="full,b0,nodiag,b1,b2,nollmdiag")
    s.add_argument("--sample", default="llm", choices=["llm", "all"])
    s.add_argument("--sessions", type=int, default=1)
    sub.add_parser("format")
    d = sub.add_parser("diagnosis")
    d.add_argument("--eedi", action="store_true", help="the Eedi variants (needs the shortlister on a GPU)")
    sub.add_parser("slips")
    h = sub.add_parser("hinter")
    h.add_argument("--n", type=int, default=24)
    c = sub.add_parser("critic")
    c.add_argument("--repeats", type=int, default=2, help="extra reviews of each hint with the cache off")
    args = ap.parse_args()
    {"sessions": sessions, "format": format_test, "diagnosis": diagnosis_test, "slips": slips_test,
     "hinter": hinter_test, "critic": critic_test}[args.part](args)


if __name__ == "__main__":
    main()
