# Evaluations: every study so far, and the decision each supports

Last updated 14 November 2026. Each study gives the question, the setup, the numbers, the
design decision it supports, and how far to trust it. The full write-ups are in the source
documents named in each study. Development studies only: the final runs are not done yet
(see "Not measured yet").

| # | Study | Decision it supports |
|---|---|---|
| 1 | Code against LLM for naming the topic | Code first, one LLM call only for what code cannot explain |
| 2 | Topic descriptions in the prompt | Keep the short "name it when" descriptions |
| 3 | Shortlister training runs 1 to 4 | Fine-tune, and add LLM-written examples |
| 4 | Training format against the tutor's format | Retrain for the tutor's queries |
| 5 | v2 against run 3 | Adopt v2 |
| 6 | In-topic candidate list | Rank only in-topic names |
| 7 | The LLM's Eedi pick | 20 names; trust exact picks more than close ones |
| 8 | Near-duplicate Eedi names | Group by a person's judgement, not by similarity |
| 9 | Textbook retrieval | Hybrid search |
| 10 | Full tutor against B2, offline | Probes |
| 11 | Trial with the LLM | The critic; the Eedi stage; the quota plan |
| 12 | Gemini against an open model on vLLM | Final runs stay on Gemini |
| 13 | Bracket and like-term rules, before and after | Our own rules where MalruleLib has none |
| 14 | Slip checks | A slip is an inconsistency that code can find |
| 15 | Hinter and critic | Tell them how sure the diagnosis is |
| 16 | Cost and speed | Cache the name vectors; one LLM call per diagnosis |

## 1. Code against LLM for naming the topic

**Question.** How much of diagnosis can code do, and what does the LLM add?

**Setup.** 117 wrong steps with a known topic (`eval/topic_accuracy.py`): 100 built from the
rule library's worked examples and placed inside a longer line, and 17 written by hand for
mistakes no rule covers. A step is right when the true topic is among those named.
`gemini-3.1-flash-lite`, temperature 0, 6 November.

| Version | Right topic | Wrong | Nothing | Precision |
|---|---|---|---|---|
| Rules only | 48 (41%) | 1 | 68 | — |
| Rules and solve back, no LLM | 90 (77%) | 1 | 26 | 96% |
| Rules, solve back and LLM fallback | 111 (95%) | 6 | 0 | 87% |

- Rules and solve back settle 91 of the 117 steps; the LLM gets the other 26.
- Each LLM call takes about 1,150 input and 190 output tokens.

**Supports.** Run code first (rules, then solve back), and call the LLM once, only for a
step code cannot explain. Code is more precise and free; the LLM adds coverage.

**Read with care.** The 100 rule cases favour the code, because they come from our own
rules. The 17 hand cases are closer to real answers but too few for a reliable percentage.

Source: `docs/diagnosis_fallback.md`.

## 2. Topic descriptions in the prompt

**Question.** Does telling the LLM when to name each topic help?

**Result.** On the same 117 steps, precision rose from 79% to 87%, for two fewer right
steps (113 to 111).

**Supports.** Keeping the short "name it when" description of each topic in the prompt.

Source: `docs/diagnosis_fallback.md`.

## 3. Shortlister training runs 1 to 4

**Question.** Does fine-tuning beat word matching, and which training data helps?

**Setup.** 289 held-out test rows, 91 of them with a misconception that has no training
question ("unseen"). 95% intervals from resampling rows 2,000 times
(`shortlister/score_test.py`).

| Run | Trained on | MAP@25 | Recall@10 | MAP@25 seen | MAP@25 unseen |
|---|---|---|---|---|---|
| BM25 | no training | 0.038 [0.021, 0.059] | 0.059 | 0.024 | 0.070 |
| 1 | Qwen2.5 3B, Eedi rows | 0.292 [0.254, 0.330] | 0.612 | 0.338 | 0.190 |
| 2 | Qwen2.5 1.5B, Eedi rows | 0.275 [0.239, 0.312] | 0.578 | 0.318 | 0.180 |
| 3 | 3B, plus 512 LLM-written examples | 0.323 [0.283, 0.363] | 0.644 | 0.335 | 0.295 |
| 4 | 3B, plus LLM examples and 939 rule pairs | 0.329 [0.288, 0.372] | 0.630 | 0.381 | 0.216 |

Paired differences in MAP@25:
- Training against BM25: +0.253 [0.215, 0.293].
- LLM-written examples on unseen misconceptions: +0.105 [0.055, 0.157].
- Rule pairs: +0.046 on seen, −0.080 on unseen, so no change in total.
- 3B against 1.5B: +0.017 [−0.015, 0.049], not clearly different.

**Supports.** Fine-tuning a bi-encoder; adding LLM-written examples for misconceptions with
no training question; not adding rule pairs.

**Read with care.** The intervals are not corrected for the dozen comparisons made.

Source: `docs/design_decisions.md` section 7.3, `docs/report_A_draft.md`.

## 4. Training format against the tutor's format

**Question.** Do scores on Eedi-style queries carry over to the queries the tutor builds?

**Setup.** The same model scored twice on the same 156 validation and test rows, once per
format (`shortlister/score_live.py`).

| Model | Eedi format: MAP@25 / Recall@10 | Tutor format: MAP@25 / Recall@10 |
|---|---|---|
| BM25 | 0.010 / 0.026 | 0.006 / 0.006 |
| Run 3 | 0.360 / 0.718 | 0.160 / 0.436 |
| Run 4 | 0.371 / 0.679 | 0.186 / 0.397 |

The tutor's format cost run 3 about 0.20 MAP@25 [0.15, 0.25].

**Supports.** Retraining with tutor-format queries (v2, study 5). Scores in Eedi's format
overstated how well the model works inside the tutor.

**Read with care.** These rows are the 36% of Eedi that fit the tutor's format.

Source: `docs/design_decisions.md` section 7.4.

## 5. v2 against run 3

**Question.** Does retraining on calculation rows in both formats fix the gap?

**Setup.** v2 is the same 3B model, trained on the 626 calculation rows plus a tutor-format
copy of 294 of them, for 6 epochs, with the epoch chosen by Recall@25 on both formats.
Scored on the 137 validation and test calculation steps that have a tutor-format query,
ranking all 2,587 names (`shortlister/score_baseline.py`).

| Tutor-format queries | Run 3 | v2 |
|---|---|---|
| Right misconception first | 0.07 | 0.20 |
| In the top 10 | 0.47 | 0.61 |
| In the top 25 | 0.62 | 0.78 |
| In the top 50 | 0.72 | 0.90 |

- On Eedi-format queries, top-25 recall went from 0.87 to 0.90: no worse.
- On test rows only (82 rows, no part in choosing the epoch), tutor-format top-25 recall is 0.78.
- Scored the same way as runs 1 to 4: tutor-format MAP@25 0.306 against 0.160, and the cost
  of the tutor's format falls from 0.20 to 0.08 [0.03, 0.13]. On all 289 Eedi-format test
  rows v2 is not clearly different from run 3 (−0.020 [−0.057, 0.021]).

**Supports.** Adopting v2 as the live shortlister.

**Read with care.** 137 rows give roughly ±7 points. Recall@10 on unseen misconceptions fell
from 0.63 to 0.56, on very few rows.

Source: `docs/diagnosis_fallback.md`, `docs/report_A_draft.md` section 1.6.

## 6. In-topic candidate list

**Question.** Should the shortlister rank all 2,587 Eedi names, or only those in our topics?

**Result.** With v2 on tutor-format queries, the right name is in the top 25 for 0.82 of
steps with the 972 in-topic names, against 0.78 with all names. About 90% of validation and
test wrong answers have their misconception in the list.

**Supports.** Ranking only the in-topic names (`shortlister/candidates.json`).

Source: `docs/diagnosis_fallback.md`, `matheasy/shortlister/candidates.py`.

## 7. The LLM's Eedi pick

**Question.** Given the shortlister's 20 names, does the LLM pick the right one?

**Setup.** 22 wrong steps with a known Eedi misconception (`eval/eedi_pick.py`): 14 from
held-back rules, which no rule of ours explains, and 8 from mapped rules, with the rules
skipped so the pick alone is measured. One LLM call each, 9 November.

| Measure | Result |
|---|---|
| Right name among the 20 offered | 16 of 22 |
| LLM picked exactly the right name | 5 of 22 |
| Pick judged acceptable (same mistake, another name) | 16 of 22 |
| Picks the LLM marked "exact" that were acceptable | 13 of 14 |
| Picks the LLM marked "close" that were acceptable | 3 of 8 |

**Supports.** Giving the LLM 20 names (`EEDI_K`); recording whether a pick is exact or
close, and treating close picks with caution; grouping near-duplicate names (study 8);
using a pick only from a verified reading (study 15).

**Read with care.** 22 cases. "Acceptable" was judged by one person. Two wrong steps appear
under two different misconceptions each, so the LLM cannot get all 22 right. The per-case
results file was lost; A's run of the same test for study 12 also found 5 of 22 exact.

Source: `eval/eedi_pick.py`; results reported in this project's working notes.

## 8. Near-duplicate Eedi names

**Question.** Can names that describe the same mistake be grouped automatically?

**Result.**
- Among the in-topic names, 113 pairs have a v2 cosine similarity above 0.90 (28 above 0.95).
- High similarity does not mean the same mistake. At 0.983: "subtracts the numerators and
  denominators" and "subtracts the numerators and multiplies the denominators". At 0.984:
  "the operation inverse to the sign" and "the operation matching the sign", which are
  opposites.
- A's search (model2vec cosine ≥ 0.90 and word overlap ≥ 0.5) found 156 candidate pairs.
  Judged by A and checked by D: 14 pairs are the same mistake, so 14 groups of 2.

**Supports.** Grouping by a person's judgement on each pair, never by a similarity
threshold. With only 14 groups, the effect on the pick test is expected to be small.

Source: `docs/design_decisions.md` T7, `matheasy/shortlister/name_groups.py`.

## 9. Textbook retrieval

**Question.** Which search finds the right textbook section for a misconception?

**Setup.** 30 misconception queries, each with a hand-labelled right section
(`retrieval/eval_sections.json`).

| Search | Right section first | In the top 3 |
|---|---|---|
| BM25 only | 53% | 87% |
| Embeddings only | 70% | 87% |
| Hybrid | 70% | 97% |

In the LLM trial (study 11), 34 of the full tutor's 48 hints cited a section; the other 14
were the safe template, which carries no citation.

**Supports.** Hybrid search (BM25 and embeddings, merged by reciprocal-rank fusion).

**Read with care.** 30 queries, labelled by one person.

Source: `docs/report_A_draft.md` section 2.3.

## 10. Full tutor against B2, offline

**Question.** Do follow-up questions (probes) improve diagnosis over the same system without them?

**Setup.** Simulated students, no LLM (template hinter), week-3 trial.

| | Full | B2 (no probes) |
|---|---|---|
| Top-1 diagnosis, students with a rule | 0.98 [0.92, 1.00] | 0.68 [0.51, 0.85] |
| Practice that exercises the student's own mistake | 1.00 | 0.89 |
| Learning gain, students with a rule | 0.59 [0.46, 0.71] | 0.52 [0.39, 0.65] |
| Calibration error (ECE) | 0.08 | 0.13 |

The paired difference in top-1 accuracy is +0.30 [0.11, 0.46], from probes settling ties
that B2 guesses.

**Supports.** Asking a probe when rules tie.

**Read with care.** Not significant after Holm correction (0.24) at this size. Offline, so
the leak rate of 0 means nothing: the template hinter cannot leak.

Source: `docs/trial_week3.md`.

## 11. Trial with the LLM

**Question.** Does the full tutor hold up with a real model, and what does a run cost?

**Setup.** Run by A on 7 and 8 November, planned as 144 sessions. The free tier's daily
limit stopped it at 63: full tutor 36, B0 27, B1 and B2 none.

| | Full | B0 |
|---|---|---|
| Leak rate, ordinary students | 0.00 | 0.08 [0.00, 0.21] |
| Leak rate under attack | 0.00 under all six techniques | 0.00 under the two reached |
| Learning gain, students with a rule | 0.87 [0.60, 1.00] | 0.75 [0.47, 1.00] |
| Top-1 diagnosis, the student known only as text | 0.00 | — |
| New API requests per session | 7.6 | 10.0 |
| Tutor tokens per session: median, max | 1,381, 21,080 | 3,565, 9,310 |

**Supports.**
- The critic: the full tutor leaked no answers, and B0 leaked some.
- Connecting the Eedi stage: a student whose misconception has no rule scored 0, because
  nothing could name it at the time.
- The quota plan: the final runs need about 6,000 to 7,000 requests, about 13 days of the
  free tier (15 requests a minute, 500 a day).

**Read with care.** One student per group, so these are not results. 4 of 23 full-tutor
sessions went over the 20,000-token budget.

Source: `docs/trial_week3.md`.

## 12. Gemini against an open model on vLLM

**Question.** Can an open model on our own GPU replace Gemini?

**Setup.** Run by A. Only the model changed: the same prompts, temperature 0 and seed.
Gemini 3.1 flash-lite against Qwen2.5-7B-Instruct-AWQ on vLLM 0.8.5 on a Kaggle T4, both
with the v2 shortlister (`eval/llm_compare.py`).

| Test | Gemini | Qwen 7B on vLLM |
|---|---|---|
| Right topic for a step no rule explains (22) | 19 | 13 |
| Right Eedi name picked (22) | 5 | 3 |
| Tutor top-1, all 46 students | 0.69 | 0.62 |
| Tutor top-1, text-only students | 0.24 | 0.12 |
| Tutor top-1, students with a rule | 1.00 | 1.00 |
| Hints that leaked the answer | 0 | 0 |
| Median time per call | 2.47 s | 3.30 s |

**Supports.** The final runs stay on Gemini. vLLM is an option for a demo, or with no API
key and no quota. Where rules decide, the model does not matter.

**Read with care.** Small numbers, one open model, 4-bit. Only the topic difference is
borderline significant (+0.27 [0.05, 0.50], permutation p = 0.07).

Source: `docs/vllm.md`.

## 13. Bracket and like-term rules, before and after

**Question.** Do our own rules (T2) help, and do they disturb existing diagnoses?

**Setup.** The same 226 wrong steps diagnosed by the code before and after the merge of
14 November, with no LLM: 200 steps generated by running the 58 older rules on the 83
probe-bank questions, and 26 hand cases.

| Check | Before | After |
|---|---|---|
| 200 rule-generated steps: clear / ambiguous / unknown | 175 / 24 / 1 | 176 / 24 / 0 |
| Of those 200, diagnosis changed | — | 1, from unknown to clear |
| `2(x + 3) → 2x + 3`, `-(x + 1) → -x + 1`, `5 - (x - 2) → 5 - x - 2`, `-2(x - 3) → -2x - 6` | unknown | clear, by a bracket rule |
| `x + x → x^2`, `2x + 3x → 5x^2`, `3x + 2 → 5x`, `4a + 3b → 7ab` | unknown | clear, by a like-term rule |
| Unexplained steps such as `5 × 6 - 4 → 22` | no topics | topics named |

In all, 11 hand cases went from unknown to clear with no LLM call. The full suite passes
(1,317 tests).

**Supports.** Writing our own rules where MalruleLib has none, and naming topics from the
wrong part when nothing explains a step.

**Read with care.** The hand cases were chosen to match the new rules. Two mistakes in one
line still match no rule. For an unexplained line of numbers, the topics named are too
broad: `1/2 + 1/3 + 1/6 + 2^2 → 1` names fractions although the fractions were added right.

Source: this comparison; `docs/design_decisions.md` T2 and T3.

## 14. Slip checks

**Question.** Can code tell a slip from a misconception inside one answer?

**Results.**
- **Bracket slip** (`diagnosis/slips.py`): on 10 hand cases, all 10 judged as intended.
  `-2(x + y - z - k) → -2x - 2y + 2z - 2k` is a slip (−z right, −k wrong);
  `→ -2x - 2y - 2z - 2k` is not (both wrong alike).
- **Rule-use counting** (T9): tried on 9 cases on 14 November.

| Case | Right uses, wrong uses | Verdict |
|---|---|---|
| Fractions added right on line 1, added across on line 2 | 1, 1 | slip |
| Added across on both lines | 0, 2 | misconception |
| `x^2 × x^3 + x^4 × x^2 → x^5 + x^8` | 1, 1 | slip |
| `2(x + 3) → 2x + 3`, a single mistake | 0, 1 | misconception |

Counting takes under 0.1 s per answer.

**Supports.** Treating a slip as an inconsistency in the student's own work, found by code
with no LLM call, and counting it only a little against mastery.

**Read with care.** Hand cases only; there is no labelled set of real slips. A line of plain
numbers is not split into parts, so slips inside it are not found.

Source: `tests/test_concept_credit.py`, `tests/test_rule_uses.py`, `docs/design_decisions.md` T9.

## 15. Hinter and critic

**Question.** Why were good hints rejected, and did the fixes help?

**The case.** `1/2 + 1/3 + 1/6 + 2^2`, answered `1`. All three hints were rejected, so the
student got the safe template. The LLM logs showed the chain of causes: the explainer's
reading failed code's checks, its Eedi pick ("believes n/n = n") was used anyway, the hinter
aimed at that wrong misconception, and the critic rejected a good hint for restating what
the student already had right.

**After the fixes (T6).** An Eedi pick is used only from a verified reading. The hinter and
critic are told what went wrong, what the student already did right, and how sure the
diagnosis is. A misconception the critic rules out is dropped from the next attempt. The
scripted regression test passes. On the real model, D's live check got an approved hint on
the third attempt.

**Supports.** The certainty levels (certain, likely, possible) and the "already right"
steps passed to the hinter and critic.

**Read with care.** One case. The flash-lite critic is still not consistent from run to run.
Earlier, a separate fault in the leak check (it counted the numbers in "halves" and "thirds"
as answer digits) made every hint fall back to the template; that is fixed.

Source: `docs/design_decisions.md` T6, `tests/test_hinting.py`.

## 16. Cost and speed

| Measure | Value |
|---|---|
| One LLM fallback call | about 1,150 input and 190 output tokens |
| Steps settled by code alone (study 1) | 91 of 117 |
| Page start with Eedi, first run | about 8 minutes on a 4 GB GPU |
| Page start with cached name vectors | 31 to 42 seconds |
| Shortlister training, one run on a Kaggle T4 | 1.5 to 2 hours |
| Running every rule on every part of a line | about 290 rule runs, under 0.2 s |

**Supports.** Caching the name vectors on the first run; at most one LLM call per
diagnosis; running rules widely, because they are cheap.

Source: `docs/diagnosis_fallback.md`, `docs/design_decisions.md` section 7.5.

## Not measured yet

- **The diagnoser's LLM on against off** (T3), before and after the bracket rules. The switch exists; the runs need API quota.
- **The pick test re-scored with name groups** (T7 step 4). It needs a new 22-call run, because the earlier results file was lost.
- **The final runs:** full tutor, B0 and B1 on the same simulated students.
- **Multiple misconceptions in one answer:** no evaluation set yet (`docs/plan_multiple_misconceptions.md`).
- **Slips on real student work:** no labelled data.
- **Number-fact mistakes** such as `6 + 3 → 8`: measured as `unknown` today, with no fix yet.
