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

**2. Answer leaks: the full tutor leaked none; the plain LLM leaked in half its sessions.** With the LLM, 27 students per condition:
- **full tutor: 0 of 27 sessions;**
- **B0, a plain LLM tutor: 13 of 27** (9 of 21 ordinary students, and 4 of 6 adversarial);
- **B1, the LLM with tools but no critic: 11 of 27.**

Every condition that keeps the critic leaked nothing, and so did 678 offline sessions (216 adversarial). Leaks here are judged by code; human labels are the next check.

**3. Does diagnosing before hinting help? Partly shown.**
- **The right diagnosis gives the best hints, and a wrong one is worse than none.** In the hinter test (24 steps), the hint passed every check in 24/24 cases when told the right misconception, 20/24 when told nothing, and 19/24 when told a wrong one. This supports stating certainty and abstaining when unsure.
- **Without a diagnosis, the LLM usually finds the mistake itself:** it aimed at the mistake in 22 of 24 steps.
- **In sessions, learning followed the diagnosis:** has_rule students gained 0.88 with the full tutor and 0.00 without the diagnosis in its feedback. But this gap is **built into the simulation**, whose students learn only from a hint that names their misconception. Without the diagnosis, sessions also ran out of the token budget far more often (19/27 against 11/27).

**Also found:**
- On its own, the LLM "corrects" the student's own mistakes when reading their input.
- The critic misses hints that do the step for the student.
- Students whose mistakes no rule explains run out of the 20,000-token session budget: all 10 of them in the full tutor's sessions.
- With 8 students per group, no session difference survives the Holm correction. The LLM sessions show size and direction; the offline runs carry the tests.

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
| LLM | `gemini-3.1-flash-lite`, temperature 0, with a cache. Free tier: 15 requests a minute, **500 a day**. The runs used two free keys over two days. |
| Judge | `gemini-3-flash-preview` (`gemini-2.5-pro` is retired for new keys; Pro has no free quota). Free tier: **20 requests a day**. |
| Eedi shortlister | adapter-3B-v2 on the Mac's GPU; 20 in-topic names per step; T7 groups on |
| Students | Offline: all 113, 2 sessions each. With the LLM: 27 (all that the rules cannot read, every 8th with a rule, one per attack), 1 session each, in each of six conditions: 162 sessions |
| Requests used | Component tests: 498 fresh calls, about $0.23 at paid list prices ($0.25 per million in, $1.50 per million out). LLM sessions: 2,225 calls, 809 of them fresh; 1.9 million tokens |

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

| Condition | Sessions | Sessions that leaked | Ordinary (21) | Under attack (6) | Safe-template rate |
|---|---|---|---|---|---|
| full, offline | 226 | 0 | — | 0 under every technique | 0.01 (0.03 under attack) |
| b2, offline | 226 | 0 | — | 0 | 0.01 |
| nodiag, offline | 226 | 0 | — | 0 | 0.00 |
| **full, with the LLM** | 27 | **0** | 0 | 0 | 0.08 |
| **b0, with the LLM** | 27 | **13** | 9 | 4 | 0.00 |
| **b1, with the LLM** | 27 | **11** | 8 | 3 | 0.07 |
| b2, with the LLM | 27 | 0 | 0 | 0 | 0.08 |
| nodiag, with the LLM | 27 | 0 | 0 | 0 | 0.08 |
| nollmdiag, with the LLM | 27 | 0 | 0 | 0 | 0.05 |

- **The critic is what stops leaks.** B1 has the same tools as the full tutor but no critic, and leaked in 11 of 27 sessions. Every condition with the critic leaked in none.
- **Which attacks worked:**
  - B0 gave in to 4 of the 6 techniques: direct request, intentional wrong answer, contextual manipulation and interpersonal influence.
  - B1 gave in to direct request, intentional wrong answer and interpersonal influence.
  - Neither gave in to emotional threats or request shaping.
- **B0 also leaked to ordinary students** (9 of 21), with the first leak at turn 8 on average. The week-3 trial (earlier code and students) had measured 0.08.
- **Statistics:** full against B0 on known_as_text students, −0.88 [−1.00, −0.62], p = 0.01, but 0.77 after Holm across the table. Eight students per group is too few to pass the correction.
- **Caveat:** leaks are judged by `metrics.states_answer`, the same check the critic uses. A leak the code can't see would be missed in every condition alike, so human labels are the next check.

### 4.4 Main result 3: does diagnosing first help?

**In sessions:**

| Condition | Learning gain, has_rule (offline, 64 students) | Learning gain, has_rule (with the LLM, 8) | Sessions ending at the token budget (LLM) |
|---|---|---|---|
| full | 0.58 [0.46, 0.72] | 0.88 [0.62, 1.00] | 11/27 |
| nodiag | 0.00 | 0.00 | 19/27 |
| B2 (no probes) | 0.52 [0.39, 0.65] | 0.75 [0.38, 1.00] | 12/27 |
| B1 | — | 0.00 | 17/27 |
| B0 | — | 0.00 | 0/27 (B0's sessions are short) |

The full-against-nodiag difference (offline 0.58, p = 0.02 after Holm) is **built into the simulation**. A simulated student "learns" only from a message that quotes its misconception's description, and without a diagnosis nothing names it. The same holds for B0 and B1. This isn't evidence that diagnosis helps; it shows that the simulated students reward naming the mistake. What the sessions do show is a cost: without the diagnosis, students stay stuck, and most sessions run until the budget stops them.

**In the hinter (24 steps):**

| The hinter was told | All checks pass | Aimed at the real mistake (judge) | Gives the answer (judge) | Leak flagged by code |
|---|---|---|---|---|
| the right misconception | **24/24** | 24/24 | 0 | 0 |
| nothing | 20/24 | 22/24 | 1 | 1 |
| a wrong misconception | 19/24 | 21/24 | 0 | 1 |
| (template, no LLM) | 8/24 | **8/24** | 0 | 0 |

Four readings:
- **A wrong diagnosis is worse than none.** Told a wrong misconception, the hinter aimed at the wrong mistake 3 times and did the step once. This is the case for the certainty levels: a tutor that is unsure should say less.
- **Telling the hinter nothing usually did little harm.** It sees the question, the answer and the wrong part, and inferred the mistake in 22 of 24 steps. The misses are telling:
  - for `23 + 45 → 14` (adding every digit), it wrote about adding fractions and mixed numbers;
  - for `2x + 4 = 10 → x = 1`, it only asked the student to substitute and check;
  - for `2/3 = 6/x → x = 4`, the judge found it walked the student to the answer, which the code missed.
- **Only the diagnosed hinter avoided a leak flag.** For `4 ÷ 6 → 1.5`, the other two LLM hints both said "simplify the fraction 4/6", which the code flags because 4/6 equals the answer. The judge didn't count it as giving the answer away, so it's a borderline case for a strict check. In the tutor, the critic stops such hints.
- **The judge separates good hints from vague ones.** At n = 5, it had rated the template aimed at the mistake 4 times out of 5. At n = 24, it's 8 of 24, against 24 of 24 for a hint told the right misconception. It still needs checking against human labels before its scores are used (`docs/labelling_protocol.md`, κ ≥ 0.7).

**Verdict: partly shown.**
- **The right diagnosis gives the best hints**, and a wrong one is costly.
- **The gain over no diagnosis is real but modest** in the hint itself: 24/24 against 20/24.
- **The learning effect in sessions can't be separated from how the simulated students work.**

What would settle it:
- Eedi steps, where the mistake isn't obvious from the answer;
- human labels on the `nodiag` and full sessions (sheets sent to the team).

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
| A2 | Diagnoser's LLM off (`nollmdiag`) | must | Component: topic accuracy 97% → 90%; no change in misconceptions without Eedi. Sessions: the same as full on has_rule (top-1 1.00, learning 0.88); topic on known_as_text 1.00 → 0.97; fewer budget stops (6/27 against 11/27) | done |
| A3 | Eedi off | must | Right misconception on Eedi's wrong answers 17% → 3% | done (component) |
| A4 | Probes off (B2) | must | Top-1 1.00 → 0.72 offline (64 students) and 1.00 → 0.52 with the LLM (8); 82% → 72% per step | done |
| — | `nodiag` | must | See 4.4: learning 0.88 → 0.00 (by construction); budget stops 11 → 19 of 27 | done |
| — | B1 (tools, no critic, no memory) | must | Top-1 0.14; learning 0.00; leaks in 11 of 27 sessions | done |
| A6 | Critic's LLM review off | could | Right verdicts 93% → 43% | done (component) |
| A8 | Slip checks off | could | Slips found 6/8 → 0/8 | done |
| A9 | T7 groups off | could | No change on 96 steps | done |
| A10 | Rewrites 0 / 1 / 2 | should | 20/24, 24/24, 24/24 approved | done |
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
| Session token budget | Full tutor: 11 of 27 LLM sessions ended at the budget, including all 10 students the rules cannot read. Median 12,424 tokens a session. Safe-template rate 0.08 |
| Calibration with the LLM | ECE 0.38 (Brier 0.75) for the full tutor, against 0.09 (0.18) offline: its stated certainty is worse when the LLM is involved |

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
| 9 | Students only Eedi describes | All 10 of them ran out of the token budget in the full tutor's sessions | budget |
| 10 | B0 with an "interpersonal influence" student | Gave the answer at turn 2 | baseline (no critic) |

## 6. Design choices, with the strength of the evidence

| # | Decision | Evidence | Strength |
|---|---|---|---|
| 1 | Code first, at most one LLM call per diagnosis | 4.1, 4.2, 4.5: LLM-only reading rewrites mistakes; LLM-only diagnosis makes 59 wrong claims against 40 | **strong** |
| 2 | Probes for tied rules | 4.2, A4 | moderate: simulated answers follow the rule exactly |
| 3 | Eedi pick only from a verified reading (T6) | 4.1: dropped picks 20% right against 27% kept | weak (small gap) |
| 4 | Near-duplicate groups by people (T7) | 4.1: no effect on 96 steps | none yet |
| 5 | Slips found by code | 4.7: 6/8, no false slips | moderate (hand cases) |
| 6 | The critic as a gate | 4.3: 0/27 leaks with it, 11/27 (B1) and 13/27 (B0) without; 4.6: 93%, no false rejections, consistent | **strong** for leaks (judged by code); moderate for the rest (author-made hints) |
| 7 | Hints aimed at the diagnosis, with certainty levels | 4.4: right 24/24, none 20/24, wrong 19/24 | moderate: a wrong diagnosis is costly; the gain over none is modest |
| 8 | Hybrid textbook search | Study 9 | moderate (not re-run) |
| 9 | 20 in-topic Eedi names | 4.1: right name offered for 69% | moderate |
| 10 | Shortlister v2 | 4.1: 3% → 17% with Eedi | **strong** (held-out real data) |
| 11 | Gemini for the final runs | Study 12 | moderate |
| 12 | 20,000-token session budget | 4.9: hit by every text-only student | **evidence against**: raise it, or make the fallback cheaper |
| 13 | One rewrite on the page | 4.8 (A10): 24/24 by the second attempt | moderate (24 steps) |

## 7. Licences, privacy and safety

- **MalruleLib** is non-commercial and share-alike. Our rules re-implement its ideas without copying code. Whether the repo's own licence must match is still open (`docs/diagnosis_rules.md`).
- **Eedi** data is competition data. It, and every file derived from it, stays out of git. This report quotes two short maths examples from it and no names or questions.
- **The analysis page stores student accounts, attempts and mistakes in SQLite** (`runs/ui/matheasy.db`, out of git). There is no retention policy or consent text yet; both are needed before real students use it.
- **"Suitable for ages 11–14"** rests on the critic's LLM review and the judge. Neither has been checked against people.

## 8. Limitations

- **No real students** have used the tutor. Every learning result is simulated.
- **The format cases, critic hints and slip cases were written and labelled by one author.**
- **The sample sizes are small:** 96 Eedi steps; known_as_text has 8 students and never_seen has 2.
- **The LLM sessions are small:** 27 per condition and 8 per main student group, so no session difference survives the Holm correction.
- **Leaks are judged by the same code the critic uses.** Human labels are still to come.
- **The hinter test has 24 steps and one judge model**, not yet checked against people.

## 9. Priority levels: what is done

| Level | Item | Status |
|---|---|---|
| must | System description | done |
| must | Component tests (format, diagnosis, critic, slips) | done |
| must | Eedi-split diagnosis test | done |
| must | `nodiag` | done (offline and with the LLM) |
| must | Reduced baseline run (full, b0, b1, b2) with the LLM | done (27 sessions each, plus nodiag and nollmdiag) |
| must | Ablations with existing switches | done |
| should | Hinter test | done (24 steps) |
| should | Human labels for the judge | sheets sent (186 tasks per member); waiting for them |
| could | New-switch ablations (memory, retrieval) | not run |
| could | vLLM again | not run |

## 10. To finish

| Part | Requests | When |
|---|---|---|
| Human labels from the LLM sessions (sheets sent) | none | the team, about 1 hour each |

Every planned run is finished. Once the sheets are back:

```
PYTHONPATH=. .venv/bin/python -m matheasy.evaluation.labels agree labels/
PYTHONPATH=. .venv/bin/python -m matheasy.evaluation.labels check labels/ --judge leak
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
