# MathEasy: project report (Report 1)

*Diagnose Before You Hint* · CSE team project · 15 November 2026

Written to `docs/report_prompt.md` (v1). Report 2 (`docs/report_2.md`) follows the revised prompt and the same runs. Every number below comes from a run in this repo:
- **Code:** commit `1f65ae2`, LLM config `95b6fb45c934`. The test scripts are `eval/report_eval.py` and `eval/report_tables.py`.
- **Run logs:** `runs/report/`, which stays out of git.

Where a part could not be run, it says so and why.

## Summary

- **The system works end to end:** reading, checking, locating, diagnosing, hinting and memory. It runs as a practice page and as an experiment runner, and leaked no answers in any test: 0 of 678 offline sessions, including 216 sessions with adversarial students.
- **Code first, LLM second is the right order for every stage tested:**
  - **Input reading:** on its own, the LLM "corrected" the student's own mistakes, rewriting `3x + 4` as `3(x + 4)`. Code first kept every standard line exactly as typed (13 of 13).
  - **Diagnosis:** the LLM fallback raised topic accuracy from 90% to 97% on rule-generated steps.
  - **Critic:** the LLM review catches 27 of 30 faulty hints, against 6 of 30 for code checks alone, and rejected none of 12 good hints.
- **Eedi is what lets the tutor name mistakes it has no rule for.** On 96 of Eedi's own wrong answers, the right misconception was named in:
  - 3% of cases with code alone, and the same without Eedi;
  - **17% with the Eedi shortlister on;**
  - 23% with the LLM picking alone, but with 59 wrong names against 40.
- **Probes add 10 points of top-1 diagnosis** (72% → 82%) on rule-generated steps. Across whole sessions, the full tutor beats B2 (no probes) on top-1, 1.00 against 0.72, on students whose misconception is a rule.
- **Not finished:**
  - **The B0 and B1 baselines with the LLM:** only 8 of about 160 planned sessions ran before Gemini's free tier stopped the run at its limit of **500 requests a day**.
  - **The hinter test:** 5 of 24 steps ran.

  Both continue after the quota resets. See section 8 for the plan.

## 1. The system

### 1.1 The problem

A student's wrong answer usually comes from a specific misconception, such as adding the numerators and the denominators of two fractions, or squaring each term of a bracket. A tutor that hints without knowing which misconception it is dealing with gives generic help, or gives the answer away. MathEasy diagnoses first: it finds the step that went wrong, names the misconception behind it, and only then writes a hint aimed at that misconception.

On 30 October 2026 the team changed the project's shape (`docs/design_decisions.md`). The original plan tested three hypotheses (H1: hypothesis testing diagnoses better than a direct LLM; H2: separate roles with memory teach better and leak less; H3: a small fine-tuned shortlister ranks misconceptions nearly as well as a large model). The project became a **LeetCode-style tutor**: the student submits working, and the tutor analyses the submission as LeetCode analyses code. For each wrong answer, it shows:
- where the working went wrong;
- what the mistake is;
- which topics it involves;
- what to practise next.

The baselines and the experiment runner were kept, so the hypotheses can still be measured, and this report does.

### 1.2 How a submission is analysed

```
Student submits a problem and their working, one step per line
 │
 1  READ        transcribe()        code; LLM only for ambiguous notation (a^bc, 1/2x) or
                                    a run of letters that may be a word (fuel, xyzw)
 2  CHECK       parse(), check_step()   code (SymPy): every step right → "correct", done
 3  LOCATE      locate()            code: the part of the first wrong step that changed
 4  DIAGNOSE    diagnose()
      rules        code: 60 MalruleLib rules + the team's bracket, like-term and carry rules
      solve back   code: a line that collapsed to one number is split into its operations
      slips        code: one bracket term wrong among right ones; a wrong number fact (6 + 3 → 8)
      probe        code: rules that tie are told apart by a follow-up question
      fallback     LLM: explain() reads the step and reconstructs the working; code re-checks
                   every step it claims. With Eedi on, the shortlister (Qwen2.5-3B + LoRA, on a
                   GPU) ranks Eedi's misconception names, 20 in-topic names (one per
                   near-duplicate group) go to the LLM, and it picks one. A pick counts only if
                   code verified the reading.
 5  SLIP/HABIT  uses_slip()         code: the same kind of step done right elsewhere → a slip
 6  MEMORY      the student store   code: mistakes and mastery per topic (SQLite on the page)
 7  HINT        retrieve()          code: BM25 + model2vec search over OpenStax passages
                hinter              LLM: one hint, citing one passage; never sees the answer
                critic              code leak check, then an LLM review (sees the answer);
                                    rejected → rewrite (1 on the page, 2 in the runner) → else
                                    a safe fixed hint
 8  SHOW        where, what, topics, hint with citation, passages, practice problems
```

LLM calls per Analyze: none for a correct answer; two (hinter and critic) when a rule explains the mistake; up to about five when the fallback runs and the hint is rewritten. The Eedi shortlister is a ranking model, not a chat call.

### 1.3 Data and models

| Source | Used for | Licence / handling |
|---|---|---|
| MalruleLib (Chen, Liu and Sonkar 2026) | The code rules: 60 of its 102 rules re-implemented, its code read but never copied or run | Non-commercial, share-alike; whether the repo's licence must match is still open |
| Eedi "Mining Misconceptions in Mathematics" (Kaggle) | 2,587 misconception names (972 in our topics) and about 1,870 questions with labelled wrong answers: trains the shortlister and supplies the names the LLM picks from | Competition data: kept in `data/`, out of git, with every derived file |
| Shortlister v2 | Qwen2.5-3B-Instruct with a LoRA adapter, fine-tuned by A on Eedi plus LLM-written examples | Out of git (`data/shortlister/`) |
| Eedi name groups (T7) | 14 pairs of near-duplicate names, judged by a person | Out of git |
| OpenStax Prealgebra 2e, Elementary Algebra 2e | 1,272 textbook passages the hints cite | CC BY 4.0 |
| Probe bank | Follow-up questions that tell tied rules apart | Ours |
| Gemini 3.1 flash-lite | The backbone: transcriber, explainer, hinter, critic, B0/B1 | Google API, free tier for this report |
| gemini-3-flash-preview | The judge in this report's hinter test | Google API |
| Qwen2.5 on vLLM | The open alternative (study 12) | Runs on a team member's GPU |

### 1.4 The tutors compared

| Condition | What it is |
|---|---|
| full | The tutor above: rules, probes, LLM fallback, hinter and critic, memory and planner |
| b2 | The same, with one-shot diagnosis: no probes |
| b1 | One agent with the same tools and retrieval, but no hypothesis testing, memory or critic |
| b0 | One prompted Socratic tutor, no tools |
| nodiag (ablation) | The full tutor, but neither the hint nor the analysis is told the misconception |
| nollmdiag (ablation) | The full tutor with the diagnoser's LLM off |

### 1.5 How it is evaluated

- **Simulated students** (D): each holds one planted misconception, a rule that decides its wrong answers. Groups:
  - has_rule (64): the misconception is one of our rules;
  - known_as_text (8): only Eedi's name for it is known;
  - never_seen (2): neither;
  - slip_only (3).
- **Adversarial students:** 36, 6 for each of the six techniques from Zhao, Knežević and Käser (ACL 2026), such as a direct request for the answer, an emotional threat, or a deliberately wrong answer.
- **Pre- and post-tests:** the same ideas with new numbers, never a practice item or a probe.
- **Metrics:**
  - top-1 diagnosis, topic accuracy and over-claiming;
  - leak rate and turns to the first leak;
  - learning gain and retention;
  - calibration (ECE, Brier);
  - practice relevance.
- **Statistics:**
  - 95% bootstrap intervals over students;
  - paired permutation tests between conditions on the same students, Holm-corrected;
  - comparisons within a student group only.
- **Component tests** (this report): each stage alone, on cases with known answers. Every result is one of three outcomes:
  - **hit:** right;
  - **miss:** wrong;
  - **abstain:** no answer.

### 1.6 Team

| Role | Member | Built |
|---|---|---|
| A · Shortlister and retrieval | Swaraj Kumar | Eedi data, the fine-tuned shortlister, textbook search, T2 and T3 rules, number facts |
| B · Diagnosis engine | Peeyush Prashant | Parser, rules, misconception and probe banks, diagnose(), B2 |
| C · Agents and orchestration | Shubham Paliwal | LLM client, orchestrator, hinter, critic, store, runner, B0 and B1 (LangGraph) |
| D · Evaluation | Rahul Saha | Simulated and adversarial students, pre/post tests, metrics, statistics, log checker, labelling; tutor tasks T6, T7, T9 |

## 2. Setup

| Item | Value |
|---|---|
| Code | Commit `1f65ae2` (14 November), tutor code unchanged during the runs. The ablations are applied by patching inside `eval/report_eval.py`. |
| LLM | `gemini-3.1-flash-lite`, temperature 0, with a cache. Config hash `95b6fb45c934`. Free tier: 15 requests a minute, 500 a day. |
| Judge | `gemini-3-flash-preview`: `gemini-2.5-pro` is retired for new keys, and Pro models have no free quota. |
| Eedi shortlister | adapter-3B-v2 on Apple-silicon GPU (MPS), 20 in-topic names per step, T7 groups on |
| Simulated students | Offline: the whole roster, 113 students, 2 sessions each.<br>With the LLM: 27 students, 1 session each: every student the rules cannot read, every 8th with a rule, and one per attack. |
| Students' judge | The keyword judge and template attacks (as `eval/llm_compare.py` does), so the students are the same whichever tutor runs |
| Statistics | 95% bootstrap intervals over students (2,000 resamples); paired permutation tests; Holm correction across each table; comparisons within a student group |

## 3. Baselines (Part 1)

### 3.1 What ran

| Condition | Offline (LLM off) | With the LLM |
|---|---|---|
| full | 226 sessions | 8 sessions, then the daily quota ran out |
| b2 | 226 sessions | not run (quota) |
| b0, b1 | need the LLM | not run (quota) |

Offline, the hinter is the template hinter, the critic runs its code checks only, and the diagnoser has no LLM fallback. These runs measure the diagnosis, the probes, memory and the safety checks, not the quality of LLM hints.

### 3.2 Results, offline

| Condition | Group | Top-1 diagnosis | Right topic | Practice relevance | Learning gain |
|---|---|---|---|---|---|
| full | has_rule (64) | 1.00 [1.00, 1.00] | 1.00 | 0.99 | 0.58 [0.46, 0.72] |
| full | known_as_text (8) | 0.00 | 0.92 [0.83, 1.00] | 0.28 | 0.00 |
| full | never_seen (2) | – | 0.50 | 0.00 | 0.00 |
| b2 | has_rule (64) | 0.72 [0.53, 0.89] | 1.00 | 0.90 | 0.52 [0.39, 0.65] |
| b2 | known_as_text (8) | 0.00 | 0.92 | 0.28 | 0.00 |

| Condition | Calibration (ECE) | Brier | Follow-ups per episode | Leak rate (all students) | Leak rate under attack (36) |
|---|---|---|---|---|---|
| full | 0.09 | 0.18 | 0.05 | 0.00 | 0.00 under all six techniques |
| b2 | 0.13 | 0.32 | 0.00 | 0.00 | 0.00 |

Full against B2, paired by student (has_rule): top-1 +0.28 [0.11, 0.47] (p = 0.01; 0.36 after Holm across the 15 comparisons in the table), learning gain +0.06 [0.01, 0.13], practice relevance +0.09 [0.01, 0.19].

### 3.3 Earlier evidence for B0

The week-3 trial (`docs/evaluations.md` study 11, earlier code, Gemini) is the latest B0 measurement:
- **Leak rate:** 0.08 for B0, against 0.00 for the full tutor.
- **API requests per session:** 10.0 for B0, against 7.6.

B1 has not been measured with the LLM.

## 4. Component tests (Part 2)

Each stage was tested on its own, on cases with known answers. Outcomes are:
- **hit:** right;
- **miss:** wrong;
- **abstain:** no answer, which is not counted as wrong.

### 4.1 Input format: `transcribe()`

40 typed lines, written for this test, each with the reading the student meant.

| Lines | n | Code only (hit / miss / abstain) | LLM only | As shipped (code first) |
|---|---|---|---|---|
| standard notation | 13 | 13 / 0 / 0 | 10 / 2 / 1 | **13 / 0 / 0** |
| ambiguous notation (`a^bc`, `1/2x`, `x/2y`, `x^2/3`) | 11 | 0 / 0 / 11 | 10 / 1 / 0 | 10 / 1 / 0 |
| runs of letters (`xyzw`, `fuel`) | 8 | 1 / 0 / 7 | 8 / 0 / 0 | 8 / 0 / 0 |
| not maths (`hello`, `five sixths`) | 8 | 8 / 0 / 0 | 6 / 2 / 0 | 6 / 2 / 0 |
| **all** | 40 | 22 / 0 / 18 | 34 / 5 / 1 | **37 / 3 / 0** |

Model calls: as shipped, 26 of 40 lines; LLM only, all 40.

- **LLM only "corrects" the student:** `3x + 4` became `3(x + 4)`, and `3x^2` became `(3x)^2`. Both rewrites pass the faithfulness check, because only brackets changed. The diagnoser would then see a right answer where the student made the mistake.
- **As shipped, three misses:**
  - `e^2/a` was read as `e^(2/a)` when the student meant `e^2 ÷ a`;
  - `hello` and `help` were read as named quantities, `{hello}`.

  The analysis page has a separate `not_maths()` check that catches those two; the experiment runner does not.

### 4.2 Diagnosis: `diagnose()`

**Rule-generated steps** (107 steps from the rules' own worked examples, each inside a longer line, plus 17 hand cases with a topic):

| Variant | Right misconception | Wrong | None named | Right topic (124) |
|---|---|---|---|---|
| Code only | 77/107 (72%) | 1 | 29 | 111 (90%) |
| Code only, the probe answered as the true rule predicts | 88/107 (82%) | 1 | 18 | 111 (90%) |
| As shipped (LLM fallback, Eedi off) | 77/107 (72%) | 1 | 29 | 120 (97%) |

On every 3rd step (42), right topic was:
- LLM only (`explain()` with no rules first): 33 (79%);
- code only: 38 (90%);
- as shipped: 41 (98%).

**Eedi's own wrong answers** (96 steps, the test split of `data/eedi/live_queries.csv`, labelled by Eedi; a name in the same T7 group counts):

| Variant | Right misconception | Wrong | None named | Right when it names one |
|---|---|---|---|---|
| Code only | 3 (3%) | 5 | 88 | 3/8 |
| As shipped, Eedi off | 3 (3%) | 5 | 88 | 3/8 |
| **As shipped, Eedi on** | **16 (17%)** | 40 | 40 | 16/56 (29%) |
| LLM only, picking from the 20 Eedi names | 22 (23%) | 59 | 15 | 22/81 (27%) |

- **The shortlist:** the right name was among the 20 offered for 66 of 96 steps (69%).
- **Exact against close:** picks the LLM marked "exact" were right 18 of 48 times; picks marked "close", 4 of 33.
- **T6:** 25 picks were dropped because code could not verify the reading; 5 of them would have been right. Among picks that were kept, 13 of 48 were right.

**Slip checks** (16 hand cases, no LLM):

| Variant | Right | Slips called slips | Misconceptions called misconceptions |
|---|---|---|---|
| Checks on (bracket, T9 rule-use counting, number facts) | 14/16 | 6/8 | 8/8 |
| Checks off (A8) | 8/16 | 0/8 | 8/8 |

Two slips were missed with the checks on: `3(x + 2) + 2(x + 5) → 3x + 6 + 2x + 5`, and `7 × 8 + 6 → 54 + 6 → 60`.

### 4.3 Hinter (LLM only): partial, 5 of 24 steps

Each step got one hint each from:
- the hinter told the right misconception;
- the hinter told nothing about the mistake;
- the hinter told a wrong misconception;
- the template hinter.

Code checked for leaks and length. A stronger model (gemini-3-flash-preview) judged maths, giving the answer, doing the step, aim at the real mistake, and age.

| Hinter | All checks pass |
|---|---|
| Told the right misconception | 5/5 |
| Told nothing | 5/5 |
| Told a wrong misconception | 3/5 (one did the step; one aimed at the wrong mistake) |
| Template (no LLM) | 4/5 |

The rewrite loop as shipped (A10) approved 4 of 5 hints on the first attempt and 5 of 5 by the second. These are preliminary, at n = 5. The other 19 steps run after the quota resets.

### 4.4 Critic (LLM only)

42 hints written for this test across six problems: 12 good, and 30 with one planted fault each.

| Critic | Right verdict | Faulty hints rejected | Good hints rejected |
|---|---|---|---|
| Code checks only | 18/42 (43%) | 6/30 | 0/12 |
| LLM review only | 39/42 (93%) | 27/30 | 0/12 |
| **Both, as shipped** | **39/42 (93%)** | 27/30 | 0/12 |

| Fault | Code only | As shipped |
|---|---|---|
| leaks the answer | 6/6 | 6/6 |
| wrong maths | 0/6 | 6/6 |
| aimed at a different mistake | 0/6 | 6/6 |
| too advanced for 11–14 | 0/6 | 6/6 |
| does the step for the student | 0/6 | **3/6** |

**Consistency:** each hint was reviewed three times, the two repeats with the cache off. All 42 got the same verdict every time.

## 5. Ablations (Part 3)

| # | Ablation | Result | Status |
|---|---|---|---|
| A1 | LLM off entirely | Offline sessions (section 3.2): diagnosis by rules and probes alone gives top-1 1.00 on has_rule students and 0.00 on known_as_text students; no leaks | done |
| A2 | Diagnoser's LLM off | Component: topic accuracy 90% off against 97% on; misconception top-1 unchanged on rule steps (72%) and on Eedi steps without Eedi (3%) | done (component); sessions pending |
| A3 | Eedi off / on | Component: right misconception on Eedi's wrong answers, 3% off against 17% on | done (component) |
| A4 | Probes off (B2) | Sessions: top-1 1.00 against 0.72 (has_rule); component: 72% → 82% when the probe is answered | done |
| A5 | Memory and planner off | B1 is the nearest condition | not run (quota) |
| A6 | Critic's LLM review off | Component: right verdict 43% against 93%; only leaks are caught | done (component) |
| A7 | Textbook retrieval off | — | not run |
| A8 | Slip checks off | Component: 0 of 8 slips found against 6 of 8 | done |
| A9 | T7 name groups off | On these 96 steps, no pick fell on the other member of a group: exact and grouped scores are the same (22) | done |
| A10 | Hint rewrites 0 / 1 / 2 | 4/5, 5/5, 5/5 approved (n = 5) | partial |
| A11 | Gemini against vLLM | Study 12: Gemini better on topics (19 against 13 of 22) and picks (5 against 3) | earlier study |
| — | No diagnosis in feedback (`nodiag`) | Offline: learning gain 0.00 against 0.58 (has_rule). Under the keyword judge, a student learns only when the hint names its misconception, so this difference is built into the simulation | done offline; LLM sessions pending |

## 6. Design choices and their evidence (Part 4)

| # | Decision | Evidence | Result | Caveat |
|---|---|---|---|---|
| 1 | Code first, at most one LLM call per diagnosis | 4.1, 4.2; study 1 | LLM only reads standard lines worse (10/13 against 13/13) and names topics worse (79% against 98%) | Hand-written format cases |
| 2 | Probes for tied rules | 3.2, 4.2; study 10 | Top-1 1.00 against 0.72 in sessions; 72% → 82% per step | Simulated students answer exactly as their rule predicts |
| 3 | An Eedi pick only from a verified reading (T6) | 4.2 | Dropped picks were right 5/25 (20%), against 13/48 (27%) for kept picks | Small difference |
| 4 | Near-duplicate names grouped by people (T7) | 4.2, study 8 | No effect on these 96 steps | Only 14 groups |
| 5 | Slips found by code (T9, bracket, number facts) | 4.2 | 6/8 slips found, no misconception called a slip | 16 hand cases |
| 6 | The critic as a gate | 4.4; study 11 | 93% right verdicts, no false rejections; no leaks in any session | Hints written for the test |
| 7 | Hints aimed at the diagnosis, with certainty levels | 4.3; study 15 | Preliminary: right and no diagnosis both 5/5, wrong 3/5 | n = 5 |
| 8 | Hybrid textbook search | Study 9 | Right section in the top 3: 97%, against 87% for one method | Not re-run |
| 9 | 20 in-topic Eedi names | 4.2; studies 6, 7 | The right name is among the 20 for 69% of steps | — |
| 10 | Shortlister v2 | Studies 3–5; 4.2 | With Eedi on, 17% against 3% | — |
| 11 | Gemini for the final runs | Study 12 | Better than Qwen 7B on vLLM | — |
| 12 | Temperature 0, cache, 20,000-token budget | Study 16; section 7 | Repeats within this report cost no requests | — |
| 13 | One rewrite on the page, two in the runner | 4.3 | All 5 approved by the second attempt | n = 5 |

## 7. Limitations

- **The rules made most of the test material.** This applies to the has_rule students and the rule-generated diagnosis steps, which therefore score near the top by construction. The Eedi test split is the one diagnosis set the rules did not make.
- **The format and critic cases were written for this report**, and labelled by their author.
- **The simulated students judge hints with a keyword judge**, so learning gain measures whether a hint names the misconception's description.
- **The LLM baselines B0 and B1, and the hinter test, are incomplete** because of the free tier's daily limit.
- **No real students have used the tutor.**

## 8. Not run yet, and the plan

| Part | Remaining | API requests | When |
|---|---|---|---|
| Hinter (4.3) | 19 steps | about 120 | after the daily reset (about 13:30 IST, 15 November) |
| LLM sessions: full, b0, nodiag, b1, b2, nollmdiag | about 160 sessions | about 1,300 | 3 days on one free key; under an hour, for about $1, on a paid key; or one shard per team member |

To continue, run each command again. Finished work is kept, and calls at temperature 0 come from the cache:

```
PYTHONPATH=. .venv/bin/python eval/report_eval.py hinter
PYTHONPATH=. .venv/bin/python eval/report_eval.py sessions --conditions full,b0,nodiag,b1,b2,nollmdiag
PYTHONPATH=. .venv/bin/python -m matheasy.evaluation.report runs/report/sessions-llm/turns --out runs/report/sessions-llm.md
PYTHONPATH=. .venv/bin/python eval/report_tables.py > runs/report/tables.md
```

## 9. Future work

- Finish the LLM baselines, and label hints by hand to check the judge (`docs/labelling_protocol.md`).
- Map more of our rules to Eedi IDs. For example, `percent_equals_decimal` also covers Eedi 329, a mapping gap that cost a hit in 4.2.
- Split lines of plain numbers into parts, so slips inside them can be found.
- A study with real students.
