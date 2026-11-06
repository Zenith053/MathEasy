# Diagnose Before You Hint: Mid-Submission Progress Report

**Project:** Diagnose Before You Hint: Hypothesis-Testing Misconception Diagnosis with a Persistent Student Model for Multi-Agent Math Tutoring (MathEasy)

**GitHub repository:** https://github.com/Zenith053/MathEasy

**Date:** 30 October 2026 (end of week 2 of 4)

**Team**

| Role | Member | Owns |
|---|---|---|
| A · Shortlister and retrieval | Swaraj Kumar | Eedi data, query builder, fine-tuned misconception shortlister, textbook search (H3) |
| B · Diagnosis engine | Peeyush Prashant | Maths tools, executable misconception rules, misconception bank, probe bank, hypothesis tester, B2 baseline (H1) |
| C · Agents and orchestration | Shubham Paliwal | LLM client, orchestrator, hinter, critic, student store, tool wrappers, session runner, B0 and B1 baselines (H2) |
| D · Evaluation | Rahul Saha | Held-back rules, simulated students, pre/post tests, log checker, metrics, labelling (measures H1, H2 and safety) |

---

## 1. Summary

We are building a maths tutor for students aged about 11–14 (fractions, decimals, algebraic expressions) that works out *why* an answer is wrong before it gives a hint. Each likely misconception is treated as a hypothesis and run as code on the student's own expression, to check whether it reproduces the student's exact answer. When several remain, the tutor asks a follow-up question that tells them apart. A critic checks every message before the student sees it, so the tutor cannot reveal the answer. The tutor is evaluated with simulated students who keep their misconception until feedback actually targets it.

**Where we are:** weeks 1 and 2 of the four-week plan are essentially complete:

- All shared interfaces between the four roles exist and are used.
- A fraction misconception travels end to end, from a simulated student through the orchestrator and diagnoser and back.
- 58 misconception rules run on real student expressions.
- The hinter, critic, tool wrappers, student store and B0 baseline work.
- The full simulated-student roster and the week-2 metrics run on real logs.

The test suite has **797 passing tests** (4 skipped because they need the Eedi or OpenStax data downloaded).

**Not yet done (week 3, as planned):**

- the probe bank and the Bayesian hypothesis tester;
- the B1 and B2 baselines;
- the planner;
- the adversarial students and the pre/post tests;
- the learning-gain, retention and calibration metrics;
- the labelling protocol.

## 2. Research questions (unchanged from the proposal)

| | Hypothesis | Compared against | Metric |
|---|---|---|---|
| H1 | Hypothesis testing finds the true misconception more often, with few follow-up questions, than direct LLM diagnosis | B0, B1, B2 | Top-1 diagnosis accuracy; follow-ups per diagnosis |
| H2 | Separate roles with a persistent student model give larger learning gains and less answer leakage | B1 | Normalised learning gain; leak rate |
| H3 | A small fine-tuned model ranks misconceptions nearly as well as a large prompted model, at lower cost | Large prompted model | MAP@25; cost per call |

Every condition uses the same pinned model and token budget, enforced in code by the shared LLM client.

## 3. Progress against the plan

| Week | Goal | Status |
|---|---|---|
| 1 · Set up and agree interfaces | Every shared function exists; one fraction example runs end to end and passes D's log checker | **Done** |
| 2 · Build the real parts and start training | Real maths tools and rules, first trained shortlister, working hinter and critic | **Mostly done.** Rules, maths tools, hinter, critic, B0, students and metrics are finished. Training code is in the repo, but first-round training scores are not yet recorded there. |
| 3 · Finish the diagnosis core and freeze | Tutor and all baselines frozen; trial run passes the log checker | Next |
| 4 · Run, analyse and write | Final runs, human labels, results and report | Planned |

## 4. What has been built

The code is in `matheasy/`, about 7,200 lines, plus about 2,900 lines of tests in `tests/`.

### 4.1 Diagnosis engine: `matheasy/diagnosis` (B)

- **Parser and maths tools:**
  - Our own Pratt parser keeps the expression exactly as the student wrote it (no simplifying), with SymPy for the exact maths.
  - Handles fractions, mixed numbers, percentages, decimals, equations, comparisons and named quantities.
  - `equivalent()` and `check_step()` give exact checks.
  - `locate()` finds the exact wrong term in a step, e.g. for `2x + 4 = 10 → x + 4 = 5` it reports `4` where `2` was expected.
- **58 executable misconception rules** re-implemented from MalruleLib, covering 60 of its 102 rules including two merged "twins". They span:
  - fractions, ratios, decimals and percentages;
  - multiplication and division, negative numbers, order of operations;
  - algebra, exponents, whole-number column arithmetic and scientific notation.

  Each rule runs on the student's own expression and was checked against MalruleLib's examples. The column-arithmetic rules were checked against MalruleLib's 120 worked cases, and all agree.

- **Misconception bank** (`bank.yaml`): an ID scheme (`rule:`, `eedi:`, `slip`) that maps Eedi misconceptions to rules. The bank refuses held-back rules.
- **`diagnose()`:** returns `clear`, `ambiguous` or `unknown`. It is connected to A's shortlister through `EediShortlister`. A standing "slip" hypothesis is always kept as a suspect.
- **`transcribe()`:** an LLM converts a student's non-standard notation into standard form in front of the parser. It is never used inside the parser, and its output is checked.
- **Documentation:** the rules list, input notation and design decisions are in `docs/`.

### 4.2 Shortlister and retrieval: `matheasy/shortlister`, `matheasy/retrieval` (A)

- **Eedi data:** the fractions, decimals and expressions subset, split by question across three topics (441 train / 62 validation / 127 test questions), so test questions never appear in training.
- **`build_query()`:** turns a located error into the query `Topic | Before | Student wrote | Correct | Changed part | Difference`. It falls back to a text diff when a question does not parse, and the same function is used in training and in the live tutor.
- **Extra training data:**
  - rule-generated pairs from B's rules;
  - LLM-written examples for misconceptions with no training questions;
  - hard-negative mining.
- **QLoRA training notebook:** includes best-epoch selection and scoring on the test split.
- **Textbook search:** OpenStax *Prealgebra 2e* and *Elementary Algebra 2e*, split by section, with hybrid BM25 and embedding retrieval and a "right section retrieved" evaluation set.

### 4.3 Agents and orchestration: `matheasy/llm`, `matheasy/tutor` (C)

- **Shared LLM client** (Gemini):
  - one pinned model;
  - a per-session token budget enforced in code;
  - caching of deterministic calls;
  - every call logged with a config hash, so results from different machines can be checked for matching settings.
- **Orchestrator:** a state machine in plain Python: read, diagnose, probe or hint, critic, remember. Code enforces at most 2 follow-ups and 2 critic rewrites, then a safe template.
- **Hinter:** never receives the correct answer, because its request type has no field for it. It cites a textbook section from A's `retrieve()`.
- **Critic (hard gate):**
  - code checks first, including a leak check with B's `equivalent()`;
  - then an LLM review of the hint.

  No full-tutor message reaches the student without passing it.

- **Tool wrappers:** check arguments against a schema, retry an unexpected error once, and log a `tool_call` record.
- **Other parts:**
  - a persistent student store;
  - the B0 baseline (single prompted tutor);
  - the session runner (pre-test, tutoring, post-test) with JSONL turn logs;
  - an end-to-end demo: `python -m matheasy.tutor`.

### 4.4 Evaluation: `matheasy/evaluation`, `eval/` (D)

- **Held-back rules:**
  - Chosen at the start with a fixed seed (`eval/select_held_back.py`): 10 of the 49 relevant MalruleLib rules, 3 fractions, 3 decimals and 4 expressions.
  - They can never enter B's bank; both the bank and the rule registry refuse them.
  - D re-implemented them separately in `held_back.py` for the students only. The tutor never imports that module.
- **Simulated students (`SimStudent`):**
  - The planted rule decides every answer, and an LLM only judges whether a tutor message targets the misconception. A student keeps its misconception until feedback targets it, which answers the problem raised by Do et al. (2026) that prompted LLM students give up their error after any feedback.
  - One student for every released rule and every held-back rule.
  - Students are split into groups: *has a rule*, *known only as text*, *never seen* and *slip-only*, so the "has a rule" group cannot inflate headline accuracy.
  - **Slip-only students** make careless slips at a 30% rate. A slip is guaranteed never to equal any rule's answer, so it cannot look like a misconception; this group measures over-claiming.
  - **Several sessions:** a student's state is saved and reloaded exactly, and an optional relapse rate brings a fixed misconception back in a later session (for retention).
  - `roster()` builds the full population. At current size it has 71 students: one per rule plus slip-only students.
- **Log specification and checker:**
  - The required fields for every event (`session_start`, `turn`, `diagnosis`, `hint`, `tool_call`, `test`, `session_end`) are listed in one place.
  - A checker validates run logs and LLM logs, including that one config hash is used across all runs.
  - Run it with `python -m matheasy.evaluation runs/`.
- **Week-2 metrics** (`python -m matheasy.evaluation.metrics runs/turn_logs`), reported per condition and per student group:
  - top-1 diagnosis accuracy (a twin of the planted rule counts as correct);
  - follow-up questions per diagnosis;
  - over-claiming rate on slips;
  - claims about misconceptions the tutor cannot know;
  - leak rate and unsafe-action rate;
  - critic rewrites and safe templates;
  - tool-call validity and recovery.

  The leak metric counts only an answer the tutor states before the student reaches it. It ignores the tutor quoting the student's own correct answer or posing a new question.

- **Tests and probes stay separate:** `overlapping()` finds any test item that repeats a probe, with the same numbers and operations in any order.

## 5. Early results (development runs, not final)

The results below come from offline development runs, which use placeholder parts and no real LLM. They show that the pipeline works; they are not evidence for or against the hypotheses.

- **End to end:** all 71 roster students ran through C's tutor offline, and every log passed D's checker.
- **Diagnosis:**
  - Every diagnosis the tutor marked `clear` was correct (12 of 12).
  - Top-1 accuracy over all episodes is 0.27, because 32 episodes ended `ambiguous`. That is expected until the probe bank and hypothesis tester arrive in week 3: without a follow-up question, the diagnoser cannot separate rules that give the same answer.
- **Safety:** the unsafe-action rate was 0 on these runs, since every full-tutor message passed the critic.
- **Integration bug found by evaluation:** the tutor read "4" instead of "x = 4" from "I got x = 4." C fixed it (commit a232dfb). After the fix, every simulated student's reply is read correctly.
- **Test suite:** 797 passed, 4 skipped (data downloads only).

## 6. Changes and decisions since the proposal

- **Rules are our own code, following MalruleLib,** not imports, so they run on the student's own expression instead of MalruleLib's generated problems. MalruleLib's code was read but never copied. Its non-commercial licence is noted, and the team still has to agree the repo licence.
- **58 rules instead of about 40:** B covered more of MalruleLib than planned, including whole-number column arithmetic, exponents and standard form.
- **Twins are merged:** two MalruleLib rules that give the same wrong answer on every problem can never be told apart, so each pair is one rule and either name counts as correct.
- **The LLM is used only in front of the parser** (`transcribe()`), never inside it, so the maths stays deterministic.
- **Gemini is the shared model,** pinned in `config/llm.yaml`.

## 7. Challenges and risks

| Challenge | Effect | What we are doing |
|---|---|---|
| The probe bank and hypothesis tester are not built yet | Many diagnoses end `ambiguous`; top-1 accuracy is low in development runs | They are B's first week-3 deliveries. If the tester is late, C takes over the probe step. |
| First-round shortlister scores not yet recorded in the repo | H3 and the shortlist cache depend on them | A runs training on Kaggle; the first scores are due at the start of week 3 |
| Eedi data must be downloaded under its competition terms | The held-back students cannot yet be sorted into *known only as text* and *never seen* | D fills in `eedi_status` once the data is downloaded |
| Comparison items ("which is larger?") need one format across roles | Students, rules and the diagnoser must write them the same way | Proposal written: `1/3 > 1/4` as a relation. Needs B and C to agree. |
| API rate limits in the final runs | A shard could stall | Runs are split by student, never by condition; the runner resumes from its logs |

## 8. Plan for the remaining weeks

**Week 3: finish the diagnosis core and freeze**

- **A:**
  - hand B the best first-round shortlister;
  - second training round and the H3 comparison against a large prompted model;
  - build the shortlist cache.
- **B:** probe bank, Bayesian hypothesis tester with information-gain follow-up choice, and the B2 baseline.
- **C:**
  - planner (scheduled re-checks across sessions);
  - B1 baseline;
  - connect the probe step and finish the experiment runner.
- **D:**
  - adversarial students covering the six attack types of Zhao et al. (2026);
  - isomorphic pre/post-test items kept apart from probes;
  - normalised learning gain, retention and calibration metrics;
  - human-labelling protocol and LLM-judge prompt.
- **Freeze** the tutor and B0, B1 and B2 after a clean trial run.

**Week 4: run, analyse and write**

- Everyone runs their share of students under every condition and labels outputs without knowing the condition.
- D merges the results, computes every metric and checks the LLM judge against the human labels.
- The final report is written: A on fine-tuning, RAG and H3; B on reasoning and H1; C on architecture, safety and H2; D on the evaluation and results.

## 9. Individual contributions so far

| Member | Role | Main contributions | Commits |
|---|---|---|---|
| Peeyush Prashant | B | Pratt parser and maths tools, `locate()`, 58 misconception rules and bank, `diagnose()` with the Eedi shortlister, `transcribe()`, design documentation | 27 |
| Swaraj Kumar | A | Eedi subset and split, query builder, rule and LLM training pairs, hard negatives, QLoRA notebook, hybrid textbook retrieval | 3 |
| Shubham Paliwal | C | LLM client and config, orchestrator, hinter, critic, tool wrappers, student store, B0 baseline, session runner and logs | 5 |
| Rahul Saha | D | Held-back rule selection, simulated students (rule, held-back, slip-only, multi-session), log specification and checker, week-2 metrics | 5 |

Commit counts are not a measure of effort: some members commit in large batches.

## 10. How to run

```bash
git clone https://github.com/Zenith053/MathEasy && cd MathEasy
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest                               # test suite
.venv/bin/python -m matheasy.tutor                       # one tutoring session, offline
.venv/bin/python -m matheasy.evaluation runs/            # check run logs
.venv/bin/python -m matheasy.evaluation.metrics runs/turn_logs   # metrics
```

Running with Gemini needs `GEMINI_API_KEY` set; see the README.
