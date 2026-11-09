# Plan: finding more than one misconception in an answer

Status: future work, not scheduled (put on hold 2026-11-09). Owner: B (diagnosis). Changes C's orchestrator and planner
and D's evaluation; see "Who is affected".

## The problem

Today `diagnose()` names **one** misconception per answer, and only for the **first** wrong step.

- **Across lines.** The page checks every line against the one before, but diagnoses only the
  first wrong step. A second mistake further down is shown as ✗ and never diagnosed.
- **Within a line.** A rule matches only if it reproduces the *whole* changed part. Two
  mistakes in one line match no rule, so the step falls to the LLM, which returns one
  `mistake`, one Eedi pick and one `kind`.
- **Numbers.** Solve back assumes exactly one wrong operation in a numeric line.

Measured on the current code (rules only, no LLM):

| Step | Each half alone | Together |
|---|---|---|
| `x^2 × x^3 + 2^3 → x^6 + 6` | both caught: `multiply_exponents_when_multiplying_powers`, `multiply_base_by_exponent` | **unknown** |
| `1/2 + 1/3 + 2(x + 3) → 2/5 + 2x + 3` | fractions caught; bracket has no rule | **unknown** |
| `0.3 × 0.2 + 1/2 + 1/3 → 0.6 + 2/5` | fractions caught; decimals not | **unknown** |

So rules we already have are lost as soon as a second mistake sits next to them, and the
answer costs an LLM call that then reports only one of the mistakes.

## What "done" means

For an answer with up to three mistakes, the diagnosis lists each one with its own
misconception, topic, the part of the line it is in, and how it was found. Code finds every
part it can, and the LLM explains only what is left. Answers with one mistake are diagnosed
exactly as well as today.

## Design

### 1. One result, several findings (data model)

```python
@dataclass(frozen=True)
class Finding:
    misconception: Optional[str]   # rule:…, eedi:…, slip, or None (the LLM's reading only)
    topics: tuple[str, ...]
    located: LocatedError          # the part of the line this mistake is in
    line: int                      # which step of the working (1-based)
    found_by: str                  # rule | solve_back | split | slip | llm
    count: int = 1                 # the same misconception in several places of one answer

Diagnosis.findings: tuple[Finding, ...]   # earliest first
```

`Diagnosis.misconception` and `.topics` stay as they are (the first finding, and the union of
all topics), so C's and D's code keeps working unchanged until they switch over.

The same misconception at two places is **one finding with `count = 2`**, not two
misconceptions. Doing it the same way twice is evidence of a belief, the opposite of a slip.

### 2. Across lines: diagnose every wrong step (cheapest, do first)

Each line is already checked against the student's *previous* line, so a mistake carried
forward is not charged twice: `2x = 14 → x = 6` after a wrong earlier line is judged on its
own. Change the page and the orchestrator to call `diagnose()` on **every** step that fails
`check_step`, not only the first.

- **Fixed later.** Line k is wrong and line k+1 is right again against the *question*: the
  student corrected themselves. Record it as a slip, with no hint.
- **LLM budget.** At most one LLM call per answer. If several steps are unknown, send them
  together in one `explain()` call (section 5) instead of one call each.

### 3. Within a line: split it into independent parts (the main gain)

`locate()` already pairs the terms of a sum and blames each wrong monomial on a term
(`x^2 × x^3 + 2^3 → x^6 + 6` gives `wrong_term = "6, x^6"`). Build on that with a new
`split.py`:

1. Take the changed part `source → wrong` (after `locate()` has stripped copied terms).
   Split `source` into its terms in written order, and group neighbours: at most 6 terms,
   so at most 32 groupings.
2. For each group, list what the student could have written for it: the right value, and
   `run_rule(r, group)` for each rule that fires on it. Usually 0 to 3 rules fire.
3. A grouping **explains the step** when one choice per group adds up to `wrong`, *and*
   each group's choice matches the student's own terms (not only the total). The term-level
   check stops two errors from cancelling into a false match.
4. Keep only groupings with the **fewest** mistakes; prefer one mistake over two (as today).
   Every wrong group becomes a finding with `found_by="split"`. A group that is wrong but
   matches no rule is passed on to the LLM as the residual (section 5).

Equations: when both sides changed and no single factor explains the step (the existing
"both sides multiplied" check in `_wrong_term`), treat each side as its own part.

Cost: about 32 groupings × 58 rules per step, all in code. Cap it at 200 rule runs per step
and log when the cap is hit.

### 4. Numbers: solve back with two wrong operations

`solve_back()` replaces one operation with an unknown and solves for it. When that finds no
rule, try **pairs** of operations: for each pair (at most 15 for 6 operations), apply each
rule that fires on each one, evaluate, and compare with the student's number. Keep a pair
only if it is the **only** one that fits; several pairs that all give the same number are
"ambiguous", which leads to a probe, not to a guess. Never try three: a single number cannot
support it.

### 5. The LLM: several mistakes, and only the residual

- **Schema.** Replace the single `mistake / kind / topics / eedi / eedi_fit` with a list
  `mistakes: [{step, mistake, kind, topics, eedi, eedi_fit}]`, at most 3. Keep `steps` and
  `explanation` as they are.
- **Context.** Pass what code already found: "already explained: `2/5` is
  add_numerators_denominators; explain the rest". The model explains the remainder and
  cannot double-count it.
- **Checks (code).** Each listed mistake must point at a step that `check_step` says is
  wrong, and no two mistakes may point at the same step unless the model marks them as two
  mistakes in one move. A failed check makes that mistake `unverified`, not the whole reply.
- **Eedi names.** Rank one list per residual part (k = 20 each, at most 2 parts, so at most
  40 names, about 600 tokens more). One part keeps today's prompt size.

### 6. Slips across the whole answer

Generalise `bracket_slip()`: a finding is a slip when the **same answer** shows the student
doing that operation right elsewhere. Example: fractions added correctly on line 2 and
added across on line 4. For each rule finding, run the rule on the other parts of the
working. If it would have fired somewhere the student was right, call that finding a slip.

### 7. Memory and the planner (C)

- `observe()` once per attempt. Each topic with a finding gets **one** wrong update, even if
  it has two findings, so one answer cannot sink a topic. Slip findings use `SLIP_WEIGHT`.
- `record()` once per misconception finding. Pass `count` so `p_holds` can rise faster for a
  mistake repeated in one answer.
- Parts of a wrong line that were done right (a group that split into its right value) give
  their topics a small credit through `question_topics`, as correct answers do now.
  Optional: do it last, after measuring.

### 8. Hints and the page

- **One hint per turn**, for the **earliest** finding, because later mistakes are often
  caused by it. The tutor's next turn takes the next finding that is still there. The critic
  is unchanged.
- **The page** shows a card per finding next to its line (✗ line 3: *added numerators and
  denominators*, found by rules). The weak-concept chart is unchanged; it just gets more
  data.

## Evaluation (D, with B)

New `eval/multi_mistakes.py`, built from items whose mistakes we already know:

- **Joined lines.** Two single-mistake items from the probe bank and MalruleLib, joined with
  `+` into one line, where the student's line is each rule's output. For example
  `x^2 × x^3 + 2^3 → x^6 + 6`. 40 pairs from different topics, 20 from the same topic.
- **Chains.** 20 multi-line answers with a mistake on two different lines.
- **Held back.** 20 lines where one half uses a held-back rule. Code should find the known
  half and the LLM the other.
- **Regression.** The existing single-mistake set, unchanged.

Metrics:
- recall per finding (each true mistake found);
- precision (no extra findings);
- exact-set accuracy;
- count accuracy (said 2 when there were 2);
- false splits on single-mistake items (must stay near 0);
- LLM calls per answer.

LLM budget: only the held-back group calls Gemini, at most 30 calls in all, as for the Eedi
pick test.

Baseline to beat (today): every joined line in the table above is `unknown`, so code-only
recall on joined lines is close to 0.

## Order of work

| # | Step | Size | Depends on |
|---|---|---|---|
| 1 | `Finding` and `Diagnosis.findings`, with `misconception` kept for compatibility | small | none |
| 2 | Evaluation set and today's baseline numbers | medium | 1 |
| 3 | Diagnose every wrong line; one batched LLM call | small | 1 |
| 4 | Split a line into parts (`split.py`) | medium | 1, 2 to measure |
| 5 | Two-operation solve back | small | 2 |
| 6 | LLM list schema and residual context | medium | 3, 4 |
| 7 | Slips across the answer | small | 3, 4 |
| 8 | Planner, store and hints (with C) | medium | 1, 3 |
| 9 | Page: a card per finding | small | 3 |

After steps 4 and 5, rerun the evaluation. If false splits on single-mistake items go above
2%, tighten the term-level check (Design, section 3) before going on.

## Risks

- **Over-splitting.** One misconception explained as two smaller ones. Mitigations: fewest
  mistakes first, the term-level match, and same-rule merging (`count`).
- **Coincidences.** Two wrong parts that happen to add up to the student's total. Mitigation:
  the term-level match; in numeric lines, a pair must be the only fit.
- **Missing rules.** Splitting only helps where each part has a rule. The bracket case above
  needs the algebra rules already proposed: minus before a bracket, multiplying only the
  first term, and x + x = x². Those are worth adding alongside step 4.
- **Cost.** Rule runs grow with groupings; the caps keep it to well under a second per step.
  LLM calls stay at one per answer.

## Who is affected

- **C (Shubham):** orchestrator, planner and store read `findings` (step 8). Until then,
  nothing breaks: `misconception` still holds the first finding.
- **D (Rahul):** the evaluation set and metrics above; the existing single-mistake metrics
  stay as the regression check.
- **A (Swaraj):** the Eedi ranker is called once per residual part, at most twice per answer.

## Decisions needed

1. At most **3** findings per answer? (Proposed.)
2. One hint per turn, for the earliest finding? (Proposed.)
3. Still at most **one** LLM call per answer? (Proposed.)
