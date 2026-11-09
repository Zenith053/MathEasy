# Diagnosis when no rule fits: solve back and the LLM fallback

## Why

The tutor checks the question against the student's final answer, so most wrong answers
are a whole line collapsed into one: `7 × 8 + 6 → 60`, `2x + 3 = 11 → x = 7`. The rules
only fit a single change, and `locate()` can only say the whole line should be 62, so most
real wrong answers came out `unknown`.

## How diagnose() works now

```
wrong step
  1. rules on the located change                     code  → clear / ambiguous
  2. solve back (lines of numbers only)              code  → the one operation that went wrong
     rules on that piece                             code  → clear / ambiguous
  3. Eedi names (only with a ranker)                 local model → the 20 closest Eedi misconceptions
     LLM fallback, explain() (only with a session)   LLM   → mistake, explanation, topics, Eedi pick
     the model's working checked step by step        code  → verified / unverified
```

**Solve back** (`matheasy/diagnosis/solve_back.py`). Assume exactly one operation went
wrong. Replace each operation in turn by an unknown and solve for the value it must have
given: `7 × 8 + 6 → 60` gives `7 × 8 → 54` or `56 + 6 → 60`. Each piece is one operation on
numbers, so the rules can run on it, and the hinter sees the piece, not the whole line.

**LLM fallback** (`matheasy/diagnosis/explain.py`). `diagnose()` calls it only when steps
1 and 2 find nothing and the orchestrator passes `session` and `question`.

The model gets:
- the question;
- the line before;
- the student's line;
- the correct line;
- solve back's pieces.

It returns:
- the student's likely working, as whole lines, with its own mistakes marked;
- a short name for the mistake;
- an explanation for the tutor;
- `kind`: misconception, slip or unclear;
- `topics`.

Code then checks the working:
- every line parses;
- the last line is the student's line;
- `check_step()` finds exactly the steps the model marked as mistakes, so an unexplained
  jump onto the student's line fails.

Rules run on each wrong step, and a topic a rule confirms is added for certain.

**Eedi misconceptions** (`matheasy/diagnosis/eedi.py`). With an `eedi_ranker` (the trained
shortlister, from `load_ranker()`), stage 3 first ranks Eedi's misconception names against the
step's query and passes the 20 closest in-topic names to the same LLM call. No extra call is
made. The model always picks the closest name, since a near concept helps a student more than
no answer, and says whether it fits `exact` or is only the `close`st offered
(`Explanation.eedi_fit`). A pick becomes the diagnosis, `eedi:<n>`, so the tutor shows it,
aims the hint at it, searches the textbook with it and records it in the student's memory, as
for a rule. A slip, or a step the model calls unclear, gets no Eedi misconception.

**Topics** (`TOPICS` in `explain.py`). The list is the rule groups, with two changes:
column arithmetic is one topic, and algebra is split into notation, expanding brackets,
collecting like terms and solving equations. Each topic has a "name it when…"
description. The LLM chooses freely, one or more per step, in no order.

Topics are kept even when the working is `unverified`, because the topic is usually right
when the exact working is not. `rule_topic()` gives a rule's topic, so rule and LLM
diagnoses use the same labels, and `Diagnosis.topics` holds them whichever found the
mistake. That lets the student store count mastery per topic.

**Prompt rules learnt from test runs**:
- The student's line is exactly what they meant (`transcribe()` ran first), so it is never
  explained away as a typo.
- Every step is a whole line, never a side sum such as `7 × 8 = 54`.
- The topic is the mistake's, not the question's.

## Results

`eval/topic_accuracy.py` has 117 wrong steps with a known topic:
- 100 built from the rule library's worked examples, placed inside a longer line, such as
  `2/3 + 1/5 + 5 - 3 → 5`;
- 17 written by hand for mistakes no rule covers, including slips inside algebra.

A step counts as right when the true topic is among those named. The model is
`gemini-3.1-flash-lite` at temperature 0, measured on 2026-11-06.

| Version | Right topic | Wrong | Nothing | Precision |
|---|---|---|---|---|
| Before (commit e5425cc): rules only | 48 (41%) | 1 | 68 | — |
| Rules and solve back, no LLM | 90 (77%) | 1 | 26 | 96% |
| Rules, solve back and LLM fallback | 111 (95%) | 6 | 0 | 87% |

Precision is the share of all topics named that are the true topic.

- **Rules and solve back** settle 91 of the 117 steps; the LLM gets the other 26.
- **The LLM** names one topic in 19 of its 26 answers and two in 7.
- **The topic descriptions** raised precision from 79% to 87%, for two fewer right steps
  than the first version of the list (113).
- **Cost:** each LLM call takes about 1,150 input tokens and 190 output tokens.

**Read with care:**
- **The rule cases favour the code.** They are generated from our own rules.
- **The hand cases are closer to real student answers,** but 17 is too few for a reliable
  percentage.
- **A second topic is often only context.** For example, `whole_number_arithmetic` gets
  added to a standard-form mistake. A check in code of which named topics are worth keeping
  is still to do.

**Misses**:
- `3(x + 4) = 21 → x = 17/3`: expanding brackets, named solving equations.
- `x - 15 = 27 → x = 32`: a subtraction slip, named after the question's topic.
- `10 - 3 + 2 + 5 - 3 → 7` and `8 - 2 × 3 + 1 + 5 - 3 → 3`: order of operations, named
  `whole_number_arithmetic`.
- `4 ÷ 6 + 5 - 3 → 7/2`: multiplication and division, named fractions.
- `85 - 23 + 5 - 3 → 54`, from the rules: a column-subtraction bug and an
  order-of-operations rule give the same answer.

## The shortlister: run 3 against v2

The shortlister ranks Eedi names for stage 3. Run 3 (Swaraj's model) was trained on Eedi's
own query format and did much worse on the tutor's maths-only queries. v2 is the same model
(Qwen2.5-3B-Instruct with a QLoRA adapter, LoRA rank 16) retrained on Kaggle on 9 November
2026 with these changes:

| | Run 3 | v2 |
|---|---|---|
| Training rows | all 1,038 Eedi rows | the 626 calculation rows (no word problems, diagrams or word answers) |
| Query format | Eedi's | Eedi's plus a tutor-format copy of 294 rows |
| Best epoch chosen by | MAP@25, Eedi-format val | Recall@25, both formats of val |
| Epochs, batch, query length | 3, 8, 512 tokens | 6, 16, 256 tokens |
| LLM-written examples | yes | yes |

Scored by `shortlister/score_baseline.py` on the 137 val and test calculation steps that
have a tutor-format query, ranking all 2,587 names:

| Tutor-format queries | Run 3 | v2 |
|---|---|---|
| Right misconception first | 0.07 | **0.20** |
| In the top 10 | 0.47 | **0.61** |
| In the top 25 | 0.62 | **0.78** |
| In the top 50 | 0.72 | **0.90** |

- **Eedi-format queries did not get worse:** top 25 went from 0.87 to 0.90.
- **On test rows only,** which played no part in choosing the epoch, v2's tutor-format
  top-25 recall is 0.78 (82 rows): the gain is not from the epoch choice.
- **The in-topic list helps v2 a little:** 0.82 in the top 25 with the 972 in-topic names
  against 0.78 with all names, so the live link keeps it.
- **Read with care:** 137 rows give roughly ±7 points of uncertainty. Recall@10 on
  misconceptions with no training question fell from 0.63 to 0.56, on very few rows.

**Rule-to-Eedi mapping** (`bank.yaml`). 37 of the 58 rules name the Eedi misconceptions they
match. Reviewed on 9 November 2026 against the Eedi names closest in meaning to each rule:
`order_of_operations.ignore_parentheses` now maps to 2488 (brackets ignored) instead of 1507
(left to right), and four rules gained a mapping (729, 684, 1383, 1751). The other 21, mostly
column-arithmetic bugs, have no Eedi name for the same mistake. When the LLM picks an Eedi name
that belongs to exactly one rule, the diagnosis records the rule's ID and keeps the pick in
`explanation.eedi`, so memory counts one mistake under one ID and the rule can re-check it.

v2 is `data/shortlister/adapter-3B-v2` (git-ignored, like all Eedi-derived files) and is what
`load_ranker()` loads. The notebook is `shortlister/train_shortlister.ipynb`.

## Not done

- **Pick accuracy.** How often the LLM picks the right Eedi name from the 20, and how
  accurate `exact` and `close` picks are, has not been measured with the real model.
- **Loading time.** The ranker takes about 8 minutes to load on a 4 GB GPU.
- **Equations.** Solve back covers lines of numbers only; equations and expressions with
  letters go straight to the LLM.
- **Topic check in code.** Decide which of the LLM's topics are worth keeping.
- **More hand cases.** The hand set needs more cases to be reliable.
- **More than one misconception per answer** (future work). Only the first wrong line is
  diagnosed, and two mistakes in one line match no rule. Plan: `plan_multiple_misconceptions.md`.
