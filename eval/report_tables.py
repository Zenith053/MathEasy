"""Tables for the project report, from the results eval/report_eval.py saves under runs/report/.

    PYTHONPATH=. .venv/bin/python eval/report_tables.py > runs/report/tables.md

Prints Markdown. A part whose results are missing is listed as not run. Holds no Eedi
names or questions; the per-case files under runs/report/ do and stay out of git.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "report"


def load(name):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(a, b):
    return "–" if not b else f"{a}/{b} ({100 * a / b:.0f}%)"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def outcomes(xs):
    c = Counter(xs)
    n = len(xs)
    answered = c["hit"] + c["miss"]
    return {"n": n, "hit": c["hit"], "miss": c["miss"], "abstain": c["abstain"], "asked": c["asked"],
            "hit_rate": pct(c["hit"], n), "coverage": pct(answered, n), "precision": pct(c["hit"], answered)}


def stopped_note(d):
    if d and d.get("stopped"):
        return f"\n\n*Stopped early ({len(d['rows'])} cases done): {d['stopped'][:120]}*"
    if d and d.get("partial"):
        return f"\n\n*Partial results: {len(d['rows'])} cases done.*"
    return ""


def calls_line(c):
    if not c:
        return ""
    return (f"\n\nLLM calls: {c['calls']} ({c['fresh']} fresh, {c['cached']} from the cache), {c['tokens']:,} tokens, "
            f"median {c['median_latency_ms']} ms per fresh call.")


# --------------------------------------------------------------------------- format

def format_tables():
    d = load("format")
    if not d:
        return "### Input reading\n\nNot run."
    rows = d["rows"]
    kinds = ["standard", "ambiguous", "letters", "not_maths"]
    out = ["### Input reading (`transcribe()`): outcome by kind of line",
           "Hit: read as meant (or, for a line that is not maths, the student is asked to type maths). "
           "Asked: unsure, and the meant reading is offered to the student. Miss: read wrongly. "
           "Abstain: not read; the student is asked again.\n"]
    body = []
    for kind in kinds + ["all"]:
        rs = [r for r in rows if kind == "all" or r["kind"] == kind]
        line = [kind, len(rs)]
        for v in ("code", "llm_only", "shipped"):
            o = outcomes([r[v]["outcome"] for r in rs])
            line.append(f"{o['hit']} / {o['asked']} / {o['miss']} / {o['abstain']}")
        body.append(line)
    out.append(table(["lines", "n", "code only: hit / asked / miss / abstain", "LLM only", "as shipped"], body))
    llm_lines = {v: sum(1 for r in rows if r[v].get("used_llm")) for v in ("code", "llm_only", "shipped")}
    out.append(f"\nLines that called the model: code only {llm_lines['code']}, LLM only {llm_lines['llm_only']}, "
               f"as shipped {llm_lines['shipped']} (of {len(rows)}).")
    misses = [(r["line"], v, r[v]) for r in rows for v in ("llm_only", "shipped") if r[v]["outcome"] == "miss"]
    if misses:
        out.append("\nMisses:\n")
        out.append(table(["typed", "variant", "read as"],
                         [[f"`{l}`", v, f"`{x.get('line')}` ({x['status']})"] for l, v, x in misses]))
    return "\n".join(out) + calls_line(d.get("calls"))


# --------------------------------------------------------------------------- diagnosis

def diagnosis_tables():
    d, de = load("diagnosis"), load("diagnosis-eedi")
    out = []
    if d:
        rows = d["rows"]
        rule_rows = [r for r in rows if r["set"] == "rules"]
        topic_rows = [r for r in rows if r["set"] in ("rules", "hand")]
        eedi_rows = [r for r in rows if r["set"] == "eedi"]
        out.append("### Diagnosis on rule-generated steps (a consistency check: the rules made these steps)\n")
        body = []
        for label, key in (("code only", "code"), ("code only, probe answered", "code_after_probe"),
                           ("as shipped (LLM fallback)", "shipped")):
            pick = lambda r: r.get(key) or r["code"]  # no probe asked: the code-only result stands
            o = outcomes([pick(r)["outcome"] for r in rule_rows if key != "shipped" or r.get("shipped")])
            th = [pick(r)["topic_hit"] for r in topic_rows if key != "shipped" or r.get("shipped")]
            body.append([label, o["n"], o["hit_rate"], o["miss"], o["abstain"], pct(sum(th), len(th)) if th else "–"])
        sub = [r for r in topic_rows if r.get("llm_only")]
        if sub:
            body.append(["LLM only (every 3rd step)", len(sub), "–", "–", "–",
                         pct(sum(r["llm_only"]["topic_hit"] for r in sub), len(sub))])
            body.append(["  as shipped, same steps", len(sub), "–", "–", "–",
                         pct(sum(r["shipped"]["topic_hit"] for r in sub), len(sub))])
            body.append(["  code only, same steps", len(sub), "–", "–", "–",
                         pct(sum(r["code"]["topic_hit"] for r in sub), len(sub))])
        out.append(table(["variant", "steps", "right misconception", "wrong", "none named", "right topic"], body))
        ambiguous = [r for r in rule_rows if r["code"]["status"] == "ambiguous"]
        settled = [r for r in ambiguous if r.get("code_after_probe") and r["code_after_probe"]["outcome"] == "hit"]
        out.append(f"\nTies between rules: {len(ambiguous)}; settled to the right rule by answering the probe as "
                   f"the true rule predicts: {len(settled)}.")
        by_topic = defaultdict(list)
        for r in rule_rows:
            by_topic[r["topic"]].append(r)
        out.append("\nBy topic (code only, right misconception):\n")
        out.append(table(["topic", "steps", "right"],
                         [[t, len(rs), pct(sum(r["code"]["outcome"] == "hit" for r in rs), len(rs))]
                          for t, rs in sorted(by_topic.items())]))
        if eedi_rows:
            out.append("\n### Diagnosis on Eedi's own wrong answers (test split, not generated by our rules)\n")
            out.append(eedi_table(eedi_rows, de["rows"] if de else []))
        out.append(stopped_note(d) + calls_line(d.get("calls")))
    if de:
        out.append(stopped_note(de) + "\n\nEedi run: " + calls_line(de.get("calls")).strip())
    return "\n".join(out) if out else "### Diagnosis\n\nNot run."


def eedi_table(rows, erows):
    by_id = {r["id"]: r for r in erows}
    variants = [("code only", lambda r: r["code"]["outcome"]),
                ("as shipped, Eedi off", lambda r: r["shipped"]["outcome"])]
    if erows:
        variants += [("as shipped, Eedi on", lambda r: by_id[r["id"]]["shipped_eedi"]["outcome"]),
                     ("LLM only (pick from Eedi's names)", lambda r: by_id[r["id"]]["llm_only_eedi"]["outcome"])]
    rows_used = [r for r in rows if not erows or r["id"] in by_id]
    body = []
    for label, f in variants:
        o = outcomes([f(r) for r in rows_used])
        body.append([label, o["n"], o["hit_rate"], o["miss"], o["abstain"], o["coverage"], o["precision"]])
    out = [table(["variant", "steps", "right misconception", "wrong", "none named", "answered", "right when answered"],
                 body)]
    if erows:
        offered = sum(by_id[r["id"]]["llm_only_eedi"]["offered_has_truth"] for r in rows_used)
        exact = sum(by_id[r["id"]]["llm_only_eedi"]["exact"] for r in rows_used)
        group = sum(by_id[r["id"]]["llm_only_eedi"]["outcome"] == "hit" for r in rows_used)
        out.append(f"\nThe right name was among the 20 offered for {pct(offered, len(rows_used))} of steps. "
                   f"LLM-only picks: exactly right {exact}, right counting near-duplicate groups (T7) {group}.")
        dropped = [by_id[r["id"]]["shipped_eedi"] for r in rows_used
                   if by_id[r["id"]]["shipped_eedi"]["eedi_pick"] and by_id[r["id"]]["shipped_eedi"]["explanation_status"] != "verified"]
        out.append(f"T6: picks dropped because code could not verify the model's reading: {len(dropped)}.")
        by_topic = defaultdict(list)
        for r in rows_used:
            by_topic[r["topic"]].append(r)
        out.append("\nBy Eedi topic (right misconception):\n")
        out.append(table(["topic", "steps", "code only", "shipped, Eedi on", "LLM only"],
                         [[t, len(rs), pct(sum(r["code"]["outcome"] == "hit" for r in rs), len(rs)),
                           pct(sum(by_id[r["id"]]["shipped_eedi"]["outcome"] == "hit" for r in rs), len(rs)),
                           pct(sum(by_id[r["id"]]["llm_only_eedi"]["outcome"] == "hit" for r in rs), len(rs))]
                          for t, rs in sorted(by_topic.items())]))
    return "\n".join(out)


# --------------------------------------------------------------------------- slips

def slips_tables():
    d = load("slips")
    if not d:
        return "### Slip checks\n\nNot run."
    body = []
    for v in ("on", "off"):
        rs = [r for r in d["rows"] if r["variant"] == v]
        slips = [r for r in rs if r["gold_slip"]]
        habits = [r for r in rs if not r["gold_slip"]]
        body.append([f"checks {v}", len(rs), pct(sum(r["hit"] for r in rs), len(rs)),
                     pct(sum(r["said_slip"] for r in slips), len(slips)),
                     pct(sum(not r["said_slip"] for r in habits), len(habits))])
    misses = [r for r in d["rows"] if r["variant"] == "on" and not r["hit"]]
    out = ["### Slip checks (bracket, rule-use counting T9, number facts), on and off; no LLM\n",
           table(["variant", "cases", "right", "slips called slips", "habits called habits"], body)]
    if misses:
        out.append("\nWrong with the checks on:\n")
        out.append(table(["question", "working", "should be"],
                         [[f"`{r['question']}`", f"`{r['working'].replace(chr(10), ' → ')}`",
                           "slip" if r["gold_slip"] else "misconception"] for r in misses]))
    return "\n".join(out)


# --------------------------------------------------------------------------- hinter

def hinter_tables():
    d = load("hinter")
    if not d:
        return "### Hinter\n\nNot run."
    rows = d["rows"]
    crit = ("maths_correct", "no_answer", "not_doing_the_step", "targets_the_mistake", "age_appropriate")
    body = []
    for v, label in (("right", "told the right misconception"), ("none", "told nothing"),
                     ("wrong", "told a wrong misconception"), ("template", "template (no LLM)")):
        rs = [r[v] for r in rows if v in r]
        if not rs:
            continue
        line = [label, len(rs), pct(sum(r["code"]["no_leak"] for r in rs), len(rs))]
        line += [pct(sum(bool(r["judge"].get(c)) for r in rs), len(rs)) for c in crit]
        allok = sum(r["code"]["no_leak"] and r["code"]["length_ok"] and all(r["judge"].get(c) for c in crit)
                    for r in rs)
        line.append(pct(allok, len(rs)))
        body.append(line)
    out = ["### Hinter: one hint per step, judged by code and by a stronger model (gemini-3-flash-preview)\n",
           table(["hinter", "hints", "no leak (code)", "maths right", "no answer", "doesn't do the step",
                  "aimed at the real mistake", "age 11-14", "all pass (hit)"], body)]
    loop = [r["loop"] for r in rows if "loop" in r]
    if loop:
        at = Counter(l["approved_at"] for l in loop)
        n = len(loop)
        out.append("\n**A10, the hinter-critic loop** (as shipped): approved on attempt 1: "
                   f"{at[1]}; on attempt 2: {at[2]}; on attempt 3: {at[3]}; never (safe template): {at[None]} "
                   f"(of {n}). So with 0 rewrites {pct(at[1], n)} pass, with 1 (the page) {pct(at[1] + at[2], n)}, "
                   f"with 2 (the runner) {pct(at[1] + at[2] + at[3], n)}.")
        reasons = Counter()
        for l in loop:
            for rs in l["rejections"]:
                for x in rs:
                    reasons[x[:90]] += 1
        if reasons:
            out.append("\nMost common critic reasons for rejecting:\n")
            out.append(table(["reason (first 90 characters)", "times"], reasons.most_common(6)))
    return "\n".join(out) + stopped_note(d) + calls_line(d.get("calls"))


# --------------------------------------------------------------------------- critic

def critic_tables():
    d = load("critic")
    if not d:
        return "### Critic\n\nNot run."
    rows = d["rows"]
    body = []
    for v, key in (("code checks only", "code_pass"), ("LLM review only", "llm_pass"), ("both, as shipped", "shipped_pass")):
        good = [r for r in rows if r["label"] == "approve"]
        bad = [r for r in rows if r["label"] == "reject"]
        right = sum((r[key] and r["label"] == "approve") or (not r[key] and r["label"] == "reject") for r in rows)
        body.append([v, len(rows), pct(right, len(rows)), pct(sum(not r[key] for r in bad), len(bad)),
                     pct(sum(not r[key] for r in good), len(good))])
    out = ["### Critic: hints with a known verdict (written for this test, labelled by their author)\n",
           table(["critic", "hints", "right verdict (hit)", "faulty hints rejected", "good hints rejected (false)"],
                 body)]
    faults = defaultdict(list)
    for r in rows:
        faults[r["fault"] or "good (approve)"].append(r)
    out.append("\nBy fault (share given the right verdict):\n")
    out.append(table(["fault", "hints", "code only", "LLM only", "as shipped"],
                     [[f, len(rs), pct(sum((not r["code_pass"]) == (r["label"] == "reject") for r in rs), len(rs)),
                       pct(sum((not r["llm_pass"]) == (r["label"] == "reject") for r in rs), len(rs)),
                       pct(sum((not r["shipped_pass"]) == (r["label"] == "reject") for r in rs), len(rs))]
                      for f, rs in sorted(faults.items())]))
    rep = [r for r in rows if r.get("repeats")]
    if rep:
        same = sum(all(x == r["llm_pass"] for x in r["repeats"]) for r in rep)
        out.append(f"\n**Consistency:** each hint reviewed {1 + len(rep[0]['repeats'])} times (the repeats with "
                   f"the cache off): the same verdict every time for {pct(same, len(rep))} of hints.")
    return "\n".join(out) + stopped_note(d) + calls_line(d.get("calls"))


def main():
    print("# Report tables (generated by eval/report_tables.py)\n")
    for f in (format_tables, diagnosis_tables, slips_tables, hinter_tables, critic_tables):
        print(f() + "\n")


if __name__ == "__main__":
    main()
