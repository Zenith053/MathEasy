# Prompt: the MathEasy project report (draft for Rahul to review)

Status: draft, 15 November 2026. Not run yet. Edit anything below. The **Decisions needed** section at the end must be answered before the runs start, because several of them change how many API calls the work needs.

---

## The task

Write one report on the whole MathEasy project ("Diagnose Before You Hint"): what the system is, how it works, and how well each part of it works, with evidence. It covers the whole team's work, not one role's. It has five parts:

0. **The system.** The problem; the approach (a LeetCode-style tutor: the student submits working, the tutor finds the first wrong step, diagnoses the misconception, then hints); the pipeline from input to hint, marking each LLM call and each code step; the data and models; who built what.
1. **Baselines.** Run every tutor condition on the same simulated students and compare them.
2. **Component tests (hit / miss).** Test each of the four stages that may call an LLM (input format, diagnosis, hinter, critic) on its own, on a fixed set of cases with known right answers. Where a stage has both a code path and an LLM path, compare code only, LLM only, and the two together as shipped.
3. **Ablations.** Switch parts of the tutor off one at a time (the LLM, Eedi, probes and the others below) and measure what each part adds.
4. **Design choices.** For each design decision the team made, give the test that backs it, or say plainly that none does.

Reuse the 16 studies in `docs/evaluations.md` where they already answer a question. Re-run one only if the code it measured has changed since (T2, T3, T6, T7, T9 and the number-fact fix in `4750027` all changed the diagnoser). Draw the project description from the code and the existing docs: `docs/design_decisions.md`, `docs/Mid_Submission_Report.md`, `docs/report_A_draft.md`, `docs/diagnosis_rules.md`, `docs/diagnosis_fallback.md`, `UI_CHANGES.md` and the README.

## Part 0. The system

- **Problem and goal.** Why diagnose before hinting; the project's original hypotheses H1–H3, and what the 30 October pivot to the LeetCode-style tutor changed.
- **Pipeline.** One diagram and a short walk-through:
  - read the input (`transcribe()`);
  - check each step (SymPy);
  - locate the first wrong step;
  - diagnose it: rules, solve back, slip and number-fact checks, probes, then the LLM fallback with Eedi names;
  - slip or habit;
  - memory;
  - textbook retrieval;
  - hinter and critic, with rewrites;
  - what the page shows.

  For each stage: code or LLM, which model, and calls per Analyze.
- **Data and models.**
  - MalruleLib (the rules);
  - Eedi "Mining Misconceptions in Mathematics" (the names and the shortlister's training data);
  - the shortlister (Qwen2.5-3B with the v2 LoRA adapter);
  - OpenStax Prealgebra 2e and Elementary Algebra 2e (1,272 passages);
  - the probe bank;
  - Gemini 3.1 flash-lite as the backbone, gemini-2.5-pro as the judge, and Qwen on vLLM as the open alternative.

  For each: its licence and what stays out of git.
- **The tutors compared.** Full, B2, B1 and B0, and what each one leaves out.
- **Evaluation design.**
  - simulated students with a planted misconception, plus adversarial students;
  - pre/post tests;
  - the metrics;
  - the statistics;
  - human labels to check the LLM judge.
- **Team.** Roles A to D and the T-tasks, in one table.

## Ground rules

- **Do not change the tutor's behaviour.** A new switch for an ablation is allowed only when it is evaluation-only, off by default, and covered by a test. The full suite must still pass.
- **Every number must come from a run in this repo.** Each table names its command, the commit and the config hash. A run that couldn't be done is listed as not run, with the reason, never estimated.
- **Keep Eedi data out of git.** That covers the raw files, the name lists and the per-case pick results. The report may quote a few Eedi names as examples, as the docs already do.
- **Never print or log the Gemini API key.**
- **Statistics:**
  - 95% bootstrap intervals over students (cluster bootstrap);
  - paired tests between conditions on the same students, with Holm correction;
  - comparisons within a student group only (has_rule, known_as_text, never_seen, slip_only, attack), never on the pooled "all" group, which misleads (Simpson's paradox);
  - small samples (under about 20) are reported as counts, not percentages, with a caveat.
- **Hit, miss and abstain are three separate outcomes.**
  - **Hit:** the right answer.
  - **Miss:** a wrong answer.
  - **Abstain:** no answer: "unknown", "unsure", or the safe fixed hint.

  An abstention is not counted as a miss. Report coverage (share answered) and precision (hits among those answered) next to the overall hit rate.

## Part 1. Baselines

Run with `python -m matheasy.tutor.runner <run> --llm --conditions full,b2,b0,b1 --sessions 2`, then the checker (`python -m matheasy.evaluation runs/<run>`) and `python -m matheasy.evaluation.report`.

| Condition | What it is |
|---|---|
| full | The tutor: rules, probes, LLM fallback, hinter and critic, memory and planner |
| b2 | The same with one-shot diagnosis (no probes) |
| b1 | One agent with the same tools and retrieval, but no hypothesis testing, memory or critic |
| b0 | One prompted Socratic tutor, no tools |

- **Students:** D's roster (74: 64 has_rule, 8 known_as_text, 2 never_seen), the 3 slip-only students, and 36 adversarial students (6 for each of the six attack techniques). Relapse 0.3, 2 sessions.
- **Metrics, by group:**
  - top-1 diagnosis and topic accuracy;
  - leak rate (overall, and under each attack);
  - turns to the first leak;
  - learning gain (pre/post) and retention;
  - calibration (ECE, Brier);
  - practice relevance;
  - tokens and requests per session, and time per session.
- **Size:** about 880 sessions, roughly 8,000–9,000 Gemini requests (from the trial: 7.6 requests per full session, 10 per B0 session).

## Part 2. Component tests (hit / miss)

Each stage is tested alone, on a fixed set of cases with known answers, without simulated students. That makes these tests cheap, repeatable, and comparable across variants.

### 2.1 Input reading ("format"): `transcribe()`

- **Cases (about 60, written by hand):**
  - lines already in standard notation;
  - ambiguous notation (`a^bc`, `1/2x`, `x/2y`, `x^2/3`);
  - runs of letters that are a word or a product (`fuel`, `xyzw`);
  - lines that aren't maths (`hello`, `five`, `help`);
  - lines where an LLM might "fix" the student's mistake.
- **Gold label:** the intended reading, or the right status (ok / unsure / not_maths).
- **Variants:**

  | Variant | What runs |
  |---|---|
  | Code only | No LLM: a line code can't settle comes back unsure |
  | LLM only | Every line goes to the LLM |
  | As shipped | Code first, the LLM only for flagged lines |

- **Hit:** the right reading or status.
- **Also report:**
  - unfaithful rewrites: the LLM changes a number or letter, or corrects the mistake;
  - LLM calls per line;
  - time per line.

### 2.2 Diagnosis: `diagnose()`

- **Cases:**
  - the 230 rule-generated probe-bank steps (study 13), so each has a known rule;
  - the 22 Eedi pick cases (study 7), whose answers are Eedi names;
  - the slip and number-fact cases (study 14 and `4750027`);
  - the hand cases for T2, T6 and T9;
  - the held-back students' steps.
- **Gold label:** the misconception (any name in its T7 group counts), "slip", or the topic.
- **Variants:**

  | Variant | What runs |
  |---|---|
  | Code only | `--no-llm-diagnosis`: rules, solve back, slip checks |
  | LLM only | `explain()` alone, no rules first. Needs an evaluation-only switch |
  | As shipped, Eedi off | Rules, then the LLM fallback |
  | As shipped, Eedi on | Rules, then the LLM fallback choosing from 20 Eedi names |

- **Hit:** top-1 matches. **Also report:**
  - topic hit;
  - abstain rate;
  - wrong-but-confident rate (a "clear" diagnosis that is wrong);
  - LLM calls and time per step.
- **Sub-studies (no API calls):**
  - T7 groups on against off (the re-score of the 22 picks, step 4);
  - T6, an Eedi pick from an unverified reading kept against dropped;
  - T9 and bracket-slip checks on against off.

### 2.3 Hinter (LLM only)

- **Cases:** about 40 wrong steps across the topics, each paired with a diagnosis.
- **Variants (one hint per step in each):**
  - **Right diagnosis:** the step's true misconception. This is how the shipped tutor runs.
  - **No diagnosis:** the hinter is told nothing about the mistake. This is the "diagnose before you hint" question itself.
  - **Wrong diagnosis:** a plausible but wrong misconception.
  - **Reference:** the template hint, as a code-only fallback.
- **Hit, checked by code:**
  - no leak (the critic's code check);
  - cites one of the retrieved passages;
  - length within limits.
- **Hit, judged by an LLM:**
  - correct maths;
  - doesn't do the step for the student;
  - aimed at the student's actual mistake;
  - right for ages 11–14.
- **The judge:** the "large" model in `config/llm.yaml` (gemini-2.5-pro). It is checked against human labels (`docs/labelling_protocol.md`, κ ≥ 0.7) before its numbers are used. If κ is lower, report the human labels only.

### 2.4 Critic (LLM only)

- **Cases (about 60 hints with known verdicts):**
  - good hints;
  - hints with one planted fault: leaks the answer, wrong maths, does the step, off the student's mistake, too advanced, or restates what the student already had right (the T6 false rejection).
- **Gold label:** approve or reject, and for a reject, which fault.
- **Variants:**
  - code checks only;
  - LLM review only;
  - both, as shipped.
- **Hit:** the verdict matches the gold label. **Also report:**
  - precision and recall of rejection;
  - false rejections of good hints;
  - consistency: each hint reviewed 3 times; how often the verdicts agree.

## Part 3. Ablations

The full tutor on the same students, with one part changed at a time, compared with the full tutor. Where a part can be measured at the component level (Part 2), do that instead of full sessions to save quota.

| # | Ablation | How | Existing switch? | Level |
|---|---|---|---|---|
| A1 | Everything with the LLM off | Runner without `--llm`: template hinter, code-only critic, no fallback | yes | sessions (free) |
| A2 | Diagnoser's LLM off, the rest on | `--no-llm-diagnosis` | yes | sessions |
| A3 | Eedi on against off | `--eedi` (Mac GPU, slow; or a teammate's GPU) | yes | sessions, plus component 2.2 |
| A4 | Probes off | Condition `b2` | yes | sessions (run in Part 1) |
| A5 | Memory and planner off | B1 is the nearest; a clean switch would be new | partly | sessions |
| A6 | Critic's LLM review off (code checks only) | New evaluation-only switch | no | component 2.4, plus sessions if quota allows |
| A7 | Textbook retrieval off | New evaluation-only switch | no | component 2.3 |
| A8 | Slip checks off (bracket, T9, number facts) | New evaluation-only switch | no | component 2.2 |
| A9 | T7 name groups off | Move `name_groups.json` aside | yes | component 2.2 |
| A10 | Hint rewrites: 0, 1 (page), 2 (runner) | `max_rewrites` | yes | component 2.3 and 2.4 |
| A11 | Gemini against Qwen on vLLM | Study 12 already covers this; re-run only if a teammate's GPU is free | — | reuse |

For each ablation, report:
- the change in top-1, leak rate, learning gain and API calls;
- its 95% interval;
- the paired test against the full tutor.

## Part 4. Design choices and their evidence

One table: decision → the test that backs it (study number or the new result) → the result → caveats. At least these:

1. Code first, at most one LLM call per diagnosis (study 1, Part 2.2).
2. Probes to tell tied rules apart (study 10, A4).
3. An Eedi pick only from a verified reading (T6; Part 2.2).
4. Near-duplicate Eedi names grouped by people's judgement, not by similarity (T7; study 8, A9).
5. A slip is an inconsistency code can find (T9, bracket and number-fact checks; study 14, A8).
6. The critic as a hard gate (study 11, Part 2.4, A6).
7. Hints aimed at the diagnosis, with certainty levels (study 15, Part 2.3).
8. Hybrid textbook search (study 9, A7).
9. Twenty in-topic Eedi names for the LLM (studies 6 and 7).
10. The shortlister v2 fine-tune (studies 3–5).
11. Gemini rather than vLLM for the final runs (study 12).
12. Temperature 0 and the cache; the 20,000-token session budget (study 16, Part 1 token counts).
13. One rewrite on the page and two in the runner (A10).

## Deliverables

- **`docs/report.md`:** the project report. Sections:
  - summary;
  - Part 0, the system;
  - setup (students, conditions, models, data, commit and config hash);
  - Parts 1 to 4;
  - limitations;
  - what was not run, and why;
  - future work.
- **Run folders** under `runs/` (git-ignored): the logs, the turn records and `results.md` from the report command.
- **Tables:** a results table for each part.
- **Charts** (optional): baselines by group, and the ablation deltas with their intervals.
- **New studies in `docs/evaluations.md`:** numbered from 17, in the existing format (question, setup, results, supports, read with care, source).
- **Evaluation scripts:** any new switch or script for the component tests, with tests.
- **Commits:** everything pushed to `main`.

## Decisions needed (answer before anything runs)

1. **Gemini quota.** On the free tier, the week-3 trial stopped at 63 sessions in a day (about 500 requests). The whole plan is roughly 12,000–14,000 requests. That is about 25 days on one free key, or a few hours on a paid key, where the cost is small at flash-lite prices. Which: a paid key, teammates each running one shard of the students on their own key (the runner's `--shard 0/4` option), or fewer runs?
2. **Who reads it and in what form.** The course's final submission (a length limit? a PDF?), the team, or both? Markdown in `docs/`, a PDF, or a shareable web page?
3. **Students after T1.** The team retired the held-back rules. Run now on the current roster (whose known_as_text and never_seen students are still the held-back ones), or wait until the T1 rework gives a new source of unseen students?
4. **Sample size.** The full roster and 2 sessions (about 880 sessions for the baselines), or a smaller sample if quota is short?
5. **Eedi in the session runs.** On this Mac (5–15 s per diagnosis, so long runs), on a teammate's GPU, or component level only (Part 2.2)?
6. **"Format".** Is it the input reading (`transcribe()`), as assumed here, or something else, such as the output format of the LLM's JSON replies?
7. **New evaluation-only switches.** OK to add them for: LLM-only diagnosis (2.2), critic LLM off (A6), retrieval off (A7) and slip checks off (A8)? Each is small, off by default and tested.
8. **Human labels.** Who labels, and how many? The judge needs about 60–80 hints labelled by at least two people for κ.
9. **Deadline.** When is the report due? This decides what gets cut if quota or time is short.
