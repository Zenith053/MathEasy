# MathEasy: project report (Report 2, revised method)

*Diagnose Before You Hint* · CSE team project · 15 November 2026

Written to `docs/report_prompt_v2.md`, which is the v1 prompt plus twelve changes from a review. Report 1 (`docs/report_1.md`) follows v1, and both use the same runs. This version differs in what it trusts:
- **The main diagnosis result** comes from data our rules did not make.
- **Three results were named as the main ones before the runs.**
- **Every result the method could have produced by construction** is marked as such.

Code: commit `1f65ae2`, frozen for every run; LLM config `95b6fb45c934`. Scripts: `eval/report_eval.py`, `eval/report_tables.py`. Run logs: `runs/report/` (out of git).

## Summary: the three main results

**1. Naming the misconception on real wrong answers.** On 96 of Eedi's own wrong answers (a test split the shortlister never trained on), the right misconception was named in:
- **3% of cases with code alone**, and the same with the LLM but no Eedi;
- **17% as shipped with Eedi**, which is right in 29% of the cases where it names one;
- **23% with the LLM picking alone**, which is right in only 27%: 59 wrong names against 40.

The shipped design gives up 6 points of hits for 19 fewer wrong diagnoses. **This is the honest headline for diagnosis.** The near-perfect top-1 scores elsewhere come from students and steps that our rules generated.

**2. Answer leaks.** No leaks in any session this round:
- 678 offline sessions, 216 of them with adversarial students across all six attack techniques;
- 8 LLM sessions.

The comparison with B0 couldn't be re-run within the free quota. The last measurement (week 3, earlier code) had B0 leaking in 8% of sessions, against 0% for the full tutor.

**3. Does diagnosing before hinting help? Not shown yet.**
- **Offline sessions:** the tutor without a diagnosis in its feedback produced no learning at all (0.00 against 0.58). But this is **built into the simulation**: its students only learn from a hint that names their misconception.
- **The hinter test:** the hinter told nothing about the mistake still wrote a well-aimed hint for all 5 steps tested. A capable LLM infers simple mistakes itself.
- **Next:** the LLM session runs and human labels, both pending, are needed to settle it.

**Also found:**
- On its own, the LLM "corrects" the student's own mistakes when reading their input.
- The critic misses hints that do the step for the student.
- Students whose mistakes no rule explains run out of the 20,000-token session budget.
- The judge model rates a vague template hint as well-aimed.

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

## 2. Method: what could mislead, and what this report does about it

| Risk | What this report does |
|---|---|
| **Circularity.** The simulated has_rule students and the rule-generated steps are made by our own rules, so the rules score near 1.00 by construction | The main diagnosis test is Eedi's own labelled wrong answers (`data/eedi/live_queries.csv`, test split, 96 steps). Rule-generated results are reported as a consistency check only |
| **No clean test of the title claim.** Full against B0 changes many things at once | A `nodiag` ablation: the full tutor whose hint and analysis are never told the misconception. The hinter test gives the same hinter the right, no, or a wrong diagnosis |
| **An LLM grading an LLM** | Students in sessions use the keyword judge and template attacks, so they are identical across tutors. The hinter judge is a stronger model, checked here for leniency, and is not trusted without human labels |
| **A cached "repeat"** | The critic's consistency repeats run with the cache off |
| **Moving code** | Every run uses commit `1f65ae2`; later pushes were not pulled |
| **Multiple comparisons** | Three main results named before the runs; Holm correction within each table |
| **Small or author-made samples** | Counts shown with every percentage; the format and critic cases are marked as author-labelled |

## 3. Setup

| Item | Value |
|---|---|
| LLM | `gemini-3.1-flash-lite`, temperature 0, with a cache. Free tier: 15 requests a minute, **500 a day**. |
| Judge | `gemini-3-flash-preview` (`gemini-2.5-pro` is retired for new keys; Pro has no free quota) |
| Eedi shortlister | adapter-3B-v2 on the Mac's GPU; 20 in-topic names per step; T7 groups on |
| Students | Offline: all 113, 2 sessions each. With the LLM: 27 (all that the rules cannot read, every 8th with a rule, one per attack), 1 session each |
| Requests used | 498 fresh calls, about 481,000 input and 70,000 output tokens. That would cost about **$0.23** at paid list prices ($0.25 per million in, $1.50 per million out) |

## 4. Results

### 4.1 Main result 1: naming the misconception on Eedi's own wrong answers

| Variant | Right misconception | Wrong | None named | Answered | Right when answered |
|---|---|---|---|---|---|
| Code only | 3 (3%) | 5 | 88 | 8% | 3/8 |
| As shipped, Eedi off | 3 (3%) | 5 | 88 | 8% | 3/8 |
| **As shipped, Eedi on** | **16 (17%)** | **40** | 40 | 58% | **16/56 (29%)** |
| LLM only, picking from 20 Eedi names | 22 (23%) | 59 | 15 | 84% | 22/81 (27%) |

| Eedi topic | Steps | Code only | As shipped, Eedi on | LLM only |
|---|---|---|---|---|
| fractions | 15 | 2 (13%) | 7 (47%) | 9 (60%) |
| decimals | 51 | 1 (2%) | 6 (12%) | 9 (18%) |
| expressions | 30 | 0 | 3 (10%) | 4 (13%) |

- **The shortlister is the bottleneck.** The right name was among the 20 offered for 66 of 96 steps (69%), but the LLM picked it for only 22.
- **The LLM's own confidence is informative.** Picks it marked "exact" were right 18 of 48 times; "close" ones, 4 of 33. Showing only "exact" picks would raise precision further.
- **T6 (picks only from a verified reading)** dropped 25 picks, of which 5 would have been right (20%), against 13 of 48 (27%) for picks it kept. It helps a little.
- **T7 (near-duplicate groups)** changed no result on these 96 steps. Its 14 groups are too few to matter here.

### 4.2 Consistency check: steps our rules generated

| Variant | Right misconception (107) | Wrong | None named | Right topic (124) |
|---|---|---|---|---|
| Code only | 77 (72%) | 1 | 29 | 111 (90%) |
| Code only, probe answered as the true rule predicts | 88 (82%) | 1 | 18 | — |
| As shipped (LLM fallback, Eedi off) | 77 (72%) | 1 | 29 | 120 (97%) |
| LLM only (every 3rd step: 42) | — | — | — | 33/42 (79%), against 41/42 shipped |

- **Probes:** there were 17 ties between rules. Answering the probe settled 11 of them to the right rule.
- **By topic:**
  - strong: negative numbers, ratios, percentages and solving equations, all right;
  - weak: decimals (1/7) and scientific notation (0/7).

  The weak topics are largely an artefact of the test. Each step is wrapped in "+ 5 − 3", and solve back can't take a written form like `4.5 × 10^4` apart.

### 4.3 Main result 2: answer leaks

| Condition | Sessions | Leak rate | Under attack (6 techniques × 6 students) | Safe-template rate |
|---|---|---|---|---|
| full, offline | 226 | 0.00 | 0.00 under every technique | 0.01 (0.03 under attack) |
| b2, offline | 226 | 0.00 | 0.00 | 0.01 |
| nodiag, offline | 226 | 0.00 | 0.00 | 0.00 |
| full, with the LLM | 8 | 0.00 | not reached | 0.10 |
| b0, with the LLM | — | not run (quota); 0.08 in week 3 | — | — |

The code leak check runs on every message the tutor writes, so the offline zeros test that check, not an LLM's restraint. The LLM sessions that test both haven't run yet.

### 4.4 Main result 3: does diagnosing first help?

**In sessions (offline):**

| Condition | Learning gain (has_rule) | Fixed rate (has_rule) | Mistake shown (all ordinary students) |
|---|---|---|---|
| full | 0.58 [0.46, 0.72] | 0.67 | 0.68 |
| nodiag | 0.00 | 0.00 | 0.00 |

The difference (0.58, p = 0.02 after Holm) is **built into the simulation**. A simulated student "learns" only from a message that quotes its misconception's description, and without a diagnosis nothing names it. This isn't evidence that diagnosis helps. It shows that the simulated students reward naming the mistake.

**In the hinter (5 of 24 steps; the rest after the quota resets):**

| The hinter was told | All checks pass | Aimed at the real mistake (judge) |
|---|---|---|
| the right misconception | 5/5 | 5/5 |
| nothing | 5/5 | 5/5 |
| a wrong misconception | 3/5 | 4/5 |
| (template, no LLM) | 4/5 | **4/5** |

Two readings:
- **Telling the hinter nothing did no harm on these simple steps.** It sees the question, the answer and the wrong part, and infers the mistake.
- **The judge is too lenient to trust alone.** It called the template hint "That method doesn't work here. Think about what each part of the question means…" aimed at the real mistake 4 times out of 5. Its scores need checking against human labels before they are used (`docs/labelling_protocol.md`, κ ≥ 0.7).

**Verdict: not shown either way yet.** What would settle it:
- the hinter test on all 24 steps, including Eedi steps where the mistake is not obvious from the answer;
- the `nodiag` LLM sessions, judged by people.

### 4.5 Input format: `transcribe()`

| Lines (author-labelled) | n | Code only: hit / miss / abstain | LLM only | As shipped |
|---|---|---|---|---|
| standard | 13 | 13 / 0 / 0 | 10 / 2 / 1 | 13 / 0 / 0 |
| ambiguous | 11 | 0 / 0 / 11 | 10 / 1 / 0 | 10 / 1 / 0 |
| runs of letters | 8 | 1 / 0 / 7 | 8 / 0 / 0 | 8 / 0 / 0 |
| not maths | 8 | 8 / 0 / 0 | 6 / 2 / 0 | 6 / 2 / 0 |
| all | 40 | 22 / 0 / 18 | 34 / 5 / 1 | 37 / 3 / 0 |

- **On its own, the LLM rewrote two of the student's mistakes into correct maths:** `3x + 4` became `3(x + 4)`, and `3x^2` became `(3x)^2`. Code accepted both, because only brackets changed. The diagnoser would then see no mistake.
- **Code first avoids this**, and calls the model for 26 of 40 lines instead of all 40.

### 4.6 Critic

| Critic (42 author-labelled hints) | Right verdict | Faulty rejected (30) | Good rejected (12) |
|---|---|---|---|
| Code checks only | 18 (43%) | 6 | 0 |
| LLM review only | 39 (93%) | 27 | 0 |
| As shipped | 39 (93%) | 27 | 0 |

- **By fault:** the shipped critic caught every leak, every wrong-maths hint, every off-target hint and every too-advanced hint.
- **Its gap:** it caught only 3 of 6 hints that do the step for the student, such as "First, 1/2 is 3/6 and 1/3 is 2/6. Now add those two fractions."
- **Consistency:** with the cache off, every hint got the same verdict all 3 times.
- **Caveat:** these hints are cleaner than real ones. The week-3 problem of good hints being rejected (T6) does not show up on hand-written good hints.

### 4.7 Slips

| Slip checks | Right (16) | Slips found (8) | Misconceptions kept (8) |
|---|---|---|---|
| on | 14 | 6 | 8 |
| off | 8 | 0 | 8 |

Missed: `3(x + 2) + 2(x + 5) → 3x + 6 + 2x + 5`, the same bracket rule done right once and wrong once, across two brackets in one line; and `7 × 8 + 6 → 54 + 6 → 60`.

### 4.8 Ablations

| # | Ablation | Priority | Result | Status |
|---|---|---|---|---|
| A1 | LLM off | must | Rules and probes alone: has_rule top-1 1.00, known_as_text 0.00 | done |
| A2 | Diagnoser's LLM off | must | Topic accuracy 97% → 90%; no change in misconceptions without Eedi | done (component) |
| A3 | Eedi off | must | Right misconception on Eedi's wrong answers 17% → 3% | done (component) |
| A4 | Probes off (B2) | must | Top-1 1.00 → 0.72 in sessions; 82% → 72% per step | done |
| — | `nodiag` | must | See 4.4: not yet informative | offline done; LLM pending |
| A6 | Critic's LLM review off | could | Right verdicts 93% → 43% | done (component) |
| A8 | Slip checks off | could | Slips found 6/8 → 0/8 | done |
| A9 | T7 groups off | could | No change on 96 steps | done |
| A10 | Rewrites 0 / 1 / 2 | should | 4/5, 5/5, 5/5 approved (n = 5) | partial |
| A5, A7 | Memory off; retrieval off | could | — | not run |
| A11 | Gemini against vLLM | could | Study 12 | earlier |

### 4.9 What a user experiences

| Measure | Value |
|---|---|
| A step code can settle | 0.003 s median, with no model call |
| One fresh model call (median / 90th percentile) | explainer 5.7 / 12.7 s; hinter 6.5 / 14.8 s; critic 5.0 / 10.8 s; transcriber 5.2 / 12.1 s (free tier, from India) |
| An Analyze with a rule-explained mistake | Hinter + critic: about 12 s at the median |
| One with the LLM fallback | About 18 s |
| Cost at paid prices | About $0.0005 per Analyze with a rule; about $0.001 with the fallback |
| Invalid replies | 2 of 749 calls were cut off (`MAX_TOKENS`), both now handled |
| Session token budget | Hit 9 times in 8 LLM sessions, all with students the rules cannot read. Safe-template rate 0.10 in those sessions |

## 5. Failures, with the stage that broke

| # | Case | What happened | Stage |
|---|---|---|---|
| 1 | Typed `3x^2` for `(3x)^2` | LLM-only reading turned it into `(3x)^2`, hiding the mistake | input (LLM) |
| 2 | Typed `hello` | Read as a named quantity `{hello}`; only the page's extra check catches it | input (LLM) |
| 3 | Typed `e^2/a`, meaning `e^2 ÷ a` | Read as `e^(2/a)` | input (LLM) |
| 4 | An Eedi percentage step labelled 329 | `percent_equals_decimal` named the right idea, but the bank does not map the rule to 329, so it scored as a miss | bank mapping |
| 5 | An Eedi step `16 → 16%` | A scientific-notation rule fired on a percentage | rules |
| 6 | `6 ÷ 2 × 3` with the rule "multiplication before division" | Named the near-twin "strictly sequential", the one wrong rule-based diagnosis | rules |
| 7 | "1/2 is 3/6 and 1/3 is 2/6. Now add those." | Approved, although it does the step | critic |
| 8 | `3(x + 2) + 2(x + 5) → 3x + 6 + 2x + 5` | Not called a slip | slips |
| 9 | Students only Eedi describes | 9 budget stops in 8 sessions, so more safe templates | budget |
| 10 | Template hint | Rated "aimed at the real mistake" by the judge | evaluation (judge) |

## 6. Design choices, with the strength of the evidence

| # | Decision | Evidence | Strength |
|---|---|---|---|
| 1 | Code first, at most one LLM call per diagnosis | 4.1, 4.2, 4.5: LLM-only reading rewrites mistakes; LLM-only diagnosis makes 59 wrong claims against 40 | **strong** |
| 2 | Probes for tied rules | 4.2, A4 | moderate: simulated answers follow the rule exactly |
| 3 | Eedi pick only from a verified reading (T6) | 4.1: dropped picks 20% right against 27% kept | weak (small gap) |
| 4 | Near-duplicate groups by people (T7) | 4.1: no effect on 96 steps | none yet |
| 5 | Slips found by code | 4.7: 6/8, no false slips | moderate (hand cases) |
| 6 | The critic as a gate | 4.6: 93%, no false rejections, consistent | moderate (author-made hints) |
| 7 | Hints aimed at the diagnosis | 4.4 | **not shown yet** |
| 8 | Hybrid textbook search | Study 9 | moderate (not re-run) |
| 9 | 20 in-topic Eedi names | 4.1: right name offered for 69% | moderate |
| 10 | Shortlister v2 | 4.1: 3% → 17% with Eedi | **strong** (held-out real data) |
| 11 | Gemini for the final runs | Study 12 | moderate |
| 12 | 20,000-token session budget | 4.9: hit in sessions with text-only students | **evidence against**: raise it, or make the fallback cheaper |
| 13 | One rewrite on the page | 4.4: 5/5 by the second attempt | weak (n = 5) |

## 7. Licences, privacy and safety

- **MalruleLib** is non-commercial and share-alike. Our rules re-implement its ideas without copying code. Whether the repo's own licence must match is still open (`docs/diagnosis_rules.md`).
- **Eedi** data is competition data. It, and every file derived from it, stays out of git. This report quotes two short maths examples from it and no names or questions.
- **The analysis page stores student accounts, attempts and mistakes in SQLite** (`runs/ui/matheasy.db`, out of git). There is no retention policy or consent text yet; both are needed before real students use it.
- **"Suitable for ages 11–14"** rests on the critic's LLM review and the judge. Neither has been checked against people.

## 8. Limitations

- **No real students** have used the tutor. Every learning result is simulated.
- **The format cases, critic hints and slip cases were written and labelled by one author.**
- **The sample sizes are small:** 96 Eedi steps; known_as_text has 8 students and never_seen has 2.
- **The free tier stopped the LLM runs:** only 8 of about 160 sessions and 5 of 24 hinter steps ran.

## 9. Priority levels: what is done

| Level | Item | Status |
|---|---|---|
| must | System description | done |
| must | Component tests (format, diagnosis, critic, slips) | done |
| must | Eedi-split diagnosis test | done |
| must | `nodiag` | offline done; LLM pending |
| must | Reduced baseline run (full, b0, b1, b2) with the LLM | **pending** (8 sessions) |
| must | Ablations with existing switches | done |
| should | Hinter test | 5 of 24 |
| should | Human labels for the judge | not started |
| could | New-switch ablations (memory, retrieval) | not run |
| could | vLLM again | not run |

## 10. To finish

| Part | Requests | When |
|---|---|---|
| Hinter: 19 more steps | about 120 | after the daily reset, about 13:30 IST on 15 November |
| LLM sessions: about 154 | about 1,300 | 3 days on one free key; under an hour for about $1 on a paid key; or one shard per member |
| Human labels: 60–80 hints by two people | none | the team, about 1 hour each |

```
PYTHONPATH=. .venv/bin/python eval/report_eval.py hinter
PYTHONPATH=. .venv/bin/python eval/report_eval.py sessions --conditions full,b0,nodiag,b1,b2,nollmdiag
PYTHONPATH=. .venv/bin/python eval/report_tables.py > runs/report/tables.md
```

## 11. Future work

- **Diagnosis:**
  - retrain the shortlister so the right name ranks higher;
  - show only the LLM's "exact" picks;
  - map more rules to Eedi IDs.
- **Critic:** teach it to reject hints that do the step.
- **Session budget:** raise it, or cut the fallback's tokens.
- **Slips:** split lines of numbers into parts.
- **People:** human labels for the judge, then a small study with real students.
