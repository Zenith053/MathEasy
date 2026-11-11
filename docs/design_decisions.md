# Design decisions: reading student maths

Owner: B. Covers how the diagnosis engine reads what a student writes: the parser, the notation standard and the LLM transcriber (`matheasy/diagnosis/maths.py`, `transcribe.py`, `docs/input_notation.md`), and how a located mistake is handed to A's shortlister (`matheasy/diagnosis/interfaces.py`, `matheasy/shortlister/`). Each entry gives the decision, the alternatives considered, and why. Open questions are listed at the end.

## What we are trying to build

**A LeetCode-style maths tutor.** The student submits an answer, and the tutor analyses the submission the way LeetCode analyses code. Decided on 30 October 2026; this replaces the hypothesis plan in the team plan and proposal.

For each wrong answer, the tutor shows the student:

1. **Where it went wrong:** the step or operation that broke.
2. **What the mistake is:** a named mistake and a short explanation.
3. **Which topics it involves:** one or more, such as fractions or solving equations. A student's mistakes build up per topic, so the tutor can see what they have and have not mastered.
4. **What to practise:** similar problems on those topics.

**How diagnosis produces this.** Code decides what it can, and the LLM handles the rest:

| Stage | Who | What it does |
|---|---|---|
| Rules | Code | A rule reproduces the student's exact answer and names the mistake |
| Solve back | Code | For a line of numbers, finds the one operation that went wrong, then runs the rules on it |
| Probe | Code | When rules tie, asks a follow-up question that tells them apart |
| LLM fallback | LLM, checked by code | When no rule fits, explains what the student did and names the topics |

See `docs/diagnosis_fallback.md` for the fallback and its results, and `docs/diagnosis_rules.md` for the rules and the probe bank.

**What changed from the old plan.**

- The H1–H3 hypotheses and the hypothesis tester are dropped. The aim is a useful analysis for the student, not a comparison that proves one design better.
- The LLM is no longer kept away from diagnosis. Rules alone cannot cover real answers, where students combine steps or make several mistakes at once.
- The topic matters more than the exact working. The topic is usually right even when the LLM's reconstruction of the working is not, and it is what the student's memory is built on.

**Not settled yet.**

- Whether the baselines (B0, B1, B2) and the experiment runner stay.
- Whether the Eedi shortlister stays. 33 of B's rules are mapped to Eedi misconceptions (`bank.yaml`), and the trained model and its cache are built (section 7). Neither is connected to `diagnose()` yet. Until one is, students "known only as text" score 0 (`docs/trial_week3.md`).

Sections below written before this decision still mention the hypotheses and baselines.

## 1. Parsing

### 1.1 Our own parser, with SymPy only for the maths

| Considered | Outcome |
|---|---|
| SymPy's parser with `evaluate=False` | Rejected: it reorders terms (`1/2 + 1/3` becomes `1/3 + 1/2`) and flattens `2/3 ÷ 4/5`, so a rule cannot see what the student wrote |
| Python's `ast` after rewriting the text with regular expressions | Used first, then replaced: nine order-sensitive rewrites, and a comparison found three bugs (`25%1` read as 251%, `25%25%` crashing with `TypeError`, `of` after a bracket read as the letters o × f) |
| **Pratt parser** (operator binding powers) | **Chosen**: each piece of notation (fractions, mixed numbers, %, implied ×, `of`, words, π) is one grammar rule rather than a text rewrite; errors give the reason and position |

The tree keeps the student's order and grouping; SymPy is used only to compute values and compare them.

### 1.2 How letters group

| Considered | Outcome |
|---|---|
| A run of letters as one block (`xy` as a unit) | Rejected: made `xy^2` mean (xy)² |
| Each letter separate, side by side meaning × at the same strength as × and ÷ | Rejected: made `a/bc` mean (a/b)·c |
| **Each letter separate; letters side by side bind tighter than × ÷ and looser than ^** | **Chosen**: `xy^2` is x·y², and `a/bc` is a/(bc), as a student means it |

### 1.3 Protecting meaning when the parser changes

A wrong reading that prints back the same way passes printing tests (that is how `a/bc` slipped through), so three checks guard the meaning:

- **Meaning table**: each notation decision checked by value (`test_notation_means`, `test_standard_notation`).
- **Difference report**: every parser change is compared with the last committed parser on about 26,000 generated inputs, and each change in meaning is reviewed before committing.
- **Meaning round trip**: random expression trees are printed and read back; the value must not change (0 failures in about 40,000, including words and π).

## 2. Ambiguous notation

| Considered | Outcome |
|---|---|
| The parser guesses one convention for each ambiguous form | Rejected: every guess is a convention someone reads differently (`(x + 1)^ab` was read as (x + 1)^a·b, but the writer meant (x + 1)^(ab)) |
| Reject every ambiguous form | Rejected as the only answer: students do write `a^bc` meaning a^(bc) |
| **A written standard (`docs/input_notation.md`), plus an LLM that reads what the student meant where the standard is not met** | **Chosen** |

Settled conventions: `a/bc` means a/(bc); `xy^2` means x·y²; a mixed number takes priority over a fraction (`1 / 3 1/2`). A division straight after an exponent (`x^2/3`, `e^2/a`) is always treated as ambiguous, whether the divisor is a number or a letter, because readers split on whether it belongs to the exponent.

`transcribe()` returns the checked text unchanged rather than re-printing it, so nothing can alter a line after it has passed the checks. The printer (`str(node)`) is used only where no typed text exists (rule answers, `locate()` fields, the strict reading offered as a choice), and a test checks that everything it prints is standard notation.

## 3. Where the LLM is used

### 3.1 In front of the parser, never inside it

| Considered | Outcome |
|---|---|
| An LLM as the parser | Rejected: `parse()` runs thousands of times per session (locate, critic, rules, probes, metrics), so cost and speed are impossible; answers would vary between runs; a student's line would reach a prompt on every check; and it would undercut H1, which tests diagnosis done by code |
| **An LLM before the parser, once per flagged line** | **Chosen**: the diagnoser (`parse`, `locate`, `diagnose`) stays pure code |

### 3.2 Rephrase, tag, or classify

| Considered | Outcome |
|---|---|
| Tag spans of the original text | Rejected for notation: models are unreliable at character positions, and tags are lost whenever a line is printed and read again |
| **Rephrase the line with brackets** | **Chosen for ambiguous notation** (`a^bc` → `a^(bc)`), with strict checks (3.5) |
| **Classify each letter run as "word" or "product"** | **Chosen for words** (4): the model picks one of two labels per run and code applies them, so it cannot change anything else |

### 3.3 When the model is called

Only for a line `notation_issue()` flags: the parser cannot read it, it holds an unknown run of letters, or its reading is ambiguous. Standard lines take a fast path with no model call. Only the flagged line is sent, never the whole message.

### 3.4 What the model sees

| Considered | Outcome |
|---|---|
| The full question | **Chosen**: helps judge intent; copying from it is caught by the checks in 3.5 |
| Only the letters the question uses | Considered: less to copy, but less context |
| Nothing but the line | Considered: cannot judge intent |
| The correct answer | **Never**: the model could steer the student's answer towards it |

### 3.5 Checking the model's output

| Considered | Outcome |
|---|---|
| Every number in the output appears somewhere in the message (a set) | Replaced: lets swapped numbers through (`2/5` → `5/2`) |
| **Numbers and letters exactly the student's, in the same order; only brackets, × and spaces may change** | **Chosen**: catches corrected answers, swapped numbers, calculated results and anything copied from the question |

The model's line must also parse. If the model is unsure, the student is offered numbered readings; `resolve()` reads the reply in code, and the tutor asks only once.

## 4. Words as quantities

### 4.1 Representing a word

| Considered | Outcome |
|---|---|
| Rename the word to a letter (`Area` → `A`) | Rejected: two words with the same first letter collide (`Area = Amount + 2` becomes `A = A + 2`), and the student's word is lost from hints and logs |
| **A named quantity in the parser** | **Chosen**: `Area` is one quantity, the same in any case |

### 4.2 Deciding word or letters multiplied (`area` or `xyzw`)

| Considered | Outcome |
|---|---|
| Length alone (4 or more letters is a word) | Rejected: would read `xyzw` as a word |
| The question decides (its variables or words) | Considered, not used |
| **A fixed list of quantity words** (`quantity_words.txt`) | **Chosen**: decided in code with no model call. Generated with an LLM, to be reviewed by a person |
| **The model classifies the remaining runs of 4 or more letters** | **Chosen**: only "word" or "product" per run |

A word decided by the model is written in braces (`{fuel}`), so the decision survives printing and re-reading without the model. Shorter unknown runs (`3ab`, `3abc`) are letters multiplied.

## 5. Scope

| Considered | Outcome |
|---|---|
| Accept sentences and numbers in words ("three and a half times two") | Dropped: harder to check, and outside what the parser handles |
| **Maths expressions only** | **Chosen**: the model reports a sentence as not maths and the student is asked to type maths |

π is the symbol `π`; `22/7` and `3.14` are ordinary numbers.

## 6. Ownership

`transcribe()` is owned by B (it is the notation standard turned into a prompt, and its failures harm the diagnosis). C calls it at the orchestrator's Read stage, inside the session, so its calls are logged and counted in the token budget.

| Considered | Outcome |
|---|---|
| C owns the transcriber (the plan gives the Read stage to C) | Considered |
| **B owns `transcribe()`, C calls it** | **Chosen** |

Every condition (the full tutor and baselines B0, B1, B2) reads the student's input through `transcribe()`. If only the full tutor had it, part of any difference in results could come from reading input better rather than from diagnosis, which would confound H1.

| Considered | Outcome |
|---|---|
| Only the full tutor gets the transcriber | Rejected: confounds the comparison with the baselines |
| **Every condition gets the transcriber** | **Chosen** |

## 7. From a located mistake to suspects: the shortlister query

### 7.1 How it works

Diagnosis needs a short list of suspect misconceptions before it can test any of them. The suspects come from about 2,600 Eedi misconceptions, far more than our rules cover, so A's shortlister ranks them. The steps:

1. **`locate()` (B, `maths.py`) does all the maths.** It narrows the step to the part the student changed and finds the wrong term, returning a `LocatedError`.
2. **`query()` (B, `interfaces.py`) is the only bridge.** `maths.py` and A's code never call each other. `query()` copies the `LocatedError` fields into A's `build_query()`.
3. **`build_query()` (A, `query.py`) only formats.** It joins six fields into one line of text:

   ```
   Topic: fractions | Before: 1/2 + 1/3 | Student wrote: 2/5 | Correct: 5/6 | Changed part: 5/6 -> 2/5 | Difference: student - correct = -13/30; student / correct = 12/25
   ```

   | Field | Where it comes from |
   |---|---|
   | Topic | `topic_of()` on the narrowed part: any letter gives expressions, else any `.` or `%` gives decimals, else fractions |
   | Before, Student wrote, Correct | the step and the correct value, from `locate()` |
   | Changed part | `expected_term -> wrong_term` from `locate()`. Both lines are expanded into monomials (`2x + 6` is `{x: 2, 1: 6}`) and compared, so `2(x + 3) -> 2x + 3` gives `6 -> 3` |
   | Difference | A's `sympy_diff()` on the narrowed part: student − correct, and student ÷ correct when both are numbers |

4. **`shortlist(query, k)` (A) ranks the misconception names** and returns the top k as `eedi:` candidates.
5. **`diagnose()` (B) tests the suspects.** It runs our rule for each suspect that has one, and asks a follow-up question when more than one survives.

### 7.2 The query states facts, not a guess

| Considered | Outcome |
|---|---|
| **The query describes the evidence: where the answer went wrong and by how much** | **Chosen**: it never says what the misconception is. Matching the evidence to a misconception is the model's job, and checking it is the diagnoser's |
| An LLM writes a sentence naming the likely mistake ("added numerators and denominators") and adds it to the query | Not used for now: it costs one more model call per diagnosis, and a wrong guess sends the shortlister confidently the wrong way. That is the kind of guess "diagnose before you hint" sets out to test, not trust. A can try it as an experiment if ranking stays weak |

### 7.3 What the shortlister model learns and searches

The model (A, `train_shortlister.ipynb`) is a **bi-encoder**: one fine-tuned language model turns any text into a vector.

- **Training.** Each labelled Eedi wrong answer gives one pair: its query from `build_query()`, and the name of its misconception. The model learns to place each query's vector close to its own misconception name, and far from the other names in the batch.
- **Live use.** All misconception **names** are embedded once. The student's query is embedded, the names are ranked by how close they are, and the top 25 become the suspects.

| Eedi table | Example | Used |
|---|---|---|
| Questions with labelled wrong answers | `\frac{1}{2}+\frac{1}{3}` → `\frac{2}{5}`, labelled with one misconception | **Training only** |
| Misconception names (about 2,600) | "Adds numerators and adds denominators when adding fractions" | **Searched live** |

The shortlister does **not** look for similar Eedi questions. What it learned from them is stored in the model's weights, and live it searches only the names. So it can rank a misconception that never appeared in training, because it scores the name's text. That is how held-back misconceptions can still be found.

The model is trained but not connected yet: `shortlist()` still uses BM25 word matching over the names, and `diagnose()` uses `RuleShortlister` (every rule is run, and the ones that reproduce the student's answer are kept). `ModelShortlister` (`shortlister/model.py`) and `CachedShortlist` (`shortlister/cache.py`) both plug into `EediShortlister`, if the team keeps the shortlister.

**Test scores of the four runs** (`shortlister/score_test.py`): 289 test rows, 91 of them with a misconception that has no training question ("unseen"). The 95% intervals come from resampling rows 2,000 times.

| Run | Trained on | MAP@25 | Recall@10 | MAP@25 seen | MAP@25 unseen |
|---|---|---|---|---|---|
| BM25 | (no training) | 0.038 [0.021, 0.059] | 0.059 [0.035, 0.087] | 0.024 | 0.070 |
| 1 | Qwen2.5 3B, Eedi rows | 0.292 [0.254, 0.330] | 0.612 [0.557, 0.668] | 0.338 | 0.190 [0.136, 0.250] |
| 2 | Qwen2.5 1.5B, Eedi rows | 0.275 [0.239, 0.312] | 0.578 [0.519, 0.633] | 0.318 | 0.180 |
| 3 | 3B, plus 512 LLM-written examples | 0.323 [0.283, 0.363] | 0.644 [0.592, 0.696] | 0.335 | 0.295 [0.218, 0.373] |
| 4 | 3B, plus LLM examples and 939 rule pairs | 0.329 [0.288, 0.372] | 0.630 [0.578, 0.682] | 0.381 | 0.216 [0.157, 0.281] |

Each change compared on the same rows (paired), MAP@25:

| Change | All rows | Seen | Unseen |
|---|---|---|---|
| Training (run 1 against BM25) | **+0.253 [0.215, 0.293]** | **+0.315** | **+0.120** |
| 3B against 1.5B (run 1 against run 2) | +0.017 [-0.015, 0.049] | +0.020 | +0.010 |
| LLM-written examples (run 3 against run 1) | +0.031 [-0.001, 0.061] | -0.003 | **+0.105 [0.055, 0.157]** |
| Rule pairs (run 4 against run 3) | +0.006 [-0.031, 0.043] | **+0.046 [0.001, 0.091]** | **-0.080 [-0.141, -0.017]** |

- Training is the only change that clearly helps on all rows.
- The LLM-written examples help where they were aimed: misconceptions with no training question. They were written for exactly those names.
- The rule pairs trade unseen misconceptions for seen ones, so they leave the total unchanged.
- The 1.5B model is not clearly worse than the 3B.
- The intervals are not corrected for the dozen comparisons made here. Two results sit right at 0, so they are weak evidence either way: rule pairs on seen rows (the interval only just excludes 0) and LLM examples on all rows (it only just includes 0).

### 7.4 Training and live queries must match

The model only works well on queries shaped like the ones it was trained on. Today the two are built differently, because Eedi's data does not fit `locate()`. Its maths is LaTeX (`\frac{1}{2}`), and many questions are word problems or multiple choice, with no step to narrow down. So in training, `build_query()` fills in the missing fields itself (`fallback_diff()`).

| Field | Training (Eedi) | Live (our `locate()`) |
|---|---|---|
| Topic | our topic, from Eedi's subject: `fractions` | `topic_of()` on the narrowed part: `fractions`. The two agree on 83% of the rows that fit the live format |
| Before, Student wrote, Correct | LaTeX: `\frac{2}{5}` | our notation: `2/5` |
| Changed part | a text diff: `5 -> 2; 6 -> 5` | the wrong term: `5/6 -> 2/5` |
| Difference | `sympy_diff()` on the whole answers | `sympy_diff()` on the narrowed part (the same for most Eedi rows) |

| Considered | Outcome |
|---|---|
| Live queries copy the training format | Possible, but it throws away the sharper wrong-term signal |
| **Training converts the LaTeX of plain-expression Eedi rows to our notation and runs `locate()`. The text diff is kept only for rows we cannot read, and used the same way live** | **Proposed**, not done yet. The count: 527 of 1,472 Eedi rows (36%) fit the live format (`shortlister/live_queries.py`) |

**The construct name in Topic.** It was the training queries' extra clue, and nothing provides it live.

| Considered | Outcome |
|---|---|
| Drop the construct name from training, so both sides use only our topic | **Used in all trained models**: no labels needed |
| Give every tutor question a hand-written skill label in Eedi's style, passed to `diagnose()` and on into the query | **Candidate**: the clue stays, at the cost of labelling the question bank |
| Generate the phrase in code from the narrowed part | Rejected: hand-written rules mapping operations to phrases would be brittle |

The with-and-without comparison has not been run, because no model has been trained with the construct name. A proposes to skip it unless the team wants to label the question bank; A and B to confirm.

**What the mismatch costs.** A trained model was scored twice on the same 156 validation and test rows, once per format (`shortlister/score_live.py`). Run 3 trains on Eedi rows plus 512 LLM-written examples. Run 4 adds 939 live-format pairs generated from 32 of B's rules.

| Model | Training format: MAP@25 / Recall@10 | Live format: MAP@25 / Recall@10 |
|---|---|---|
| BM25 | 0.010 / 0.026 | 0.006 / 0.006 |
| Run 3 | 0.360 / 0.718 | 0.160 / 0.436 |
| Run 4 | 0.371 / 0.679 | 0.186 / 0.397 |

- Live-format queries cost about 0.2 MAP@25 (run 3: 95% interval 0.15 to 0.25), so scores in the training format overstate live performance.
- Run 4's rule pairs did not close the gap, and runs 3 and 4 are close on live queries. On these rows' validation part, run 3 is slightly ahead (0.174 against 0.169), so it is the one used for the cache.
- On all test rows, run 4 is worse on misconceptions with no training question (MAP@25 0.216 against 0.295).
- These rows are the 36% of Eedi that fit the live format, so they may not represent every live query.
- On 199 steps generated by the 32 mapped rules, run 3 ranks one of the rule's Eedi IDs in the top 25 on 73% of steps, and first on 17%. The mapping was done by A and awaits B's review.

### 7.5 The name vectors are cached on the first run

Embedding all the misconception names (927 in the live candidate list) with the 3B model takes about 8 minutes on a 4 GB GPU. This is the slow part of starting the page or the tutor with Eedi on. `ModelShortlister` therefore embeds the names **once, on the first run**, and saves the vectors next to the adapter (`data/shortlister/adapter-3B-v2/name_vectors_<key>.pt`, about 8 MB). Later starts read the file back and take about 30 seconds, mostly loading the model, which is still needed to embed each live query.

- **Recomputed when it must be.** The key is a hash of the adapter's config and results, the name length limit, the device, and the candidate IDs and names. A retrained adapter or a changed name list gets a new file, so stale vectors are never read.
- **Not in git.** The vectors are built from Eedi's misconception names, so like every Eedi-derived file they stay out of the repository (competition rules). Each machine computes them on its first run. To spare a teammate that wait, the file can be shared privately alongside the adapter (the private Kaggle dataset).

## 8. Misconception rules: our own code, following MalruleLib

| Considered | Outcome |
|---|---|
| Import MalruleLib and call its rules | Rejected: MalruleLib is a problem-and-answer generator. Each rule makes its own problem from templates and gives the wrong answer to that problem. It has no way to take an expression a student wrote (`1/2 + 1/3` inside a longer line) and say what the misconception predicts, which is the question diagnosis asks. Its rules also read the problem's text and metadata, so they cannot run on our parsed tree; and importing its code would bring its share-alike non-commercial licence into our repo |
| **Write our own rules, keeping MalruleLib's names and wrong answers** | **Chosen**: each rule follows MalruleLib's `calculate_expected_answer()`, but runs on any expression the student writes and returns `None` where the misconception says nothing. The names keep IDs shared with the literature and with D's held-back list, and the held-back rules can be refused in code. The column-arithmetic rules were checked against the 120 worked cases in MalruleLib's own tests |

MalruleLib's code was read, never copied or run. Rules that give the same answer on every problem (twins) are merged into one rule, so that a diagnosis is never permanently ambiguous. Coverage and conventions are in `docs/diagnosis_rules.md`.

## Open questions

| Question | Who decides |
|---|---|
| Answers using π, 22/7 or 3.14: state the value in each question, or accept all three | B |
| Fixed readings for `1/2x` and `x/2y`, or always ask | B |
| Do baselines B0 and B1 also get the transcriber, so comparisons stay fair? | Team |
| Clarification questions logged separately, so they do not count as diagnostic follow-ups in H1 metrics | C and D |
| Human review of `quantity_words.txt` before the final runs | B |
| D's `item_key()` handles the new `Word` and `Mixed` nodes | D |
| Training and live shortlister queries in one format (7.4), including where the topic comes from | A and B |

## To do

Decided, but not started. Anyone in the team can pick an item up.

| # | Item | Decided |
|---|---|---|
| T1 | No MalruleLib rule is held back | 2026-11-09 (team leader, final) |
| T2 | Rules for bracket and like-term mistakes | 2026-11-09 |
| T3 | Testing with only the diagnoser's LLM off | 2026-11-09 |
| T4 | B0 on LangChain/LangGraph | 2026-11-09 |
| T5 | B1 on LangGraph | 2026-11-09 |
| T6 | Every hint rejected: wrong misconception given to the hinter, and a critic that is too strict | 2026-11-09 |
| T7 | Group near-duplicate Eedi names | 2026-11-09 |
| T8 | The LLM's own "slip" verdict becomes a slip diagnosis | 2026-11-09 |
| T9 | Slips from counting each rule's right and wrong uses | 2026-11-09 |

### T1. No MalruleLib rule is held back

**Decision (final).** Every MalruleLib rule may now go into the bank. Until now, D set aside 10 rules (`eval/held_back_rules.json`) so the evaluation could test diagnosis on misconceptions the rules never saw:

| Group | Held back until now |
|---|---|
| fractions | `fractions.denominator_comparison_error`, `fractions.subtract_across`, `ratios_proportions.each_fraction_digit_is_ratio` |
| decimals | `decimals.right_align_decimals`, `decimals.shorter_is_larger`, `percentages.percentage_as_index` |
| expressions | `algebra.change_side_change_sign`, `negative_numbers.multiplication_rule_for_addition`, `negative_numbers.negative_swaps_operation`, `negative_numbers.two_negatives_always_positive` |

**What has to change.**
- **Code that refuses them.** `diagnosis/rules/__init__.py` (`held_back_rules()`, `HeldBackRuleError`) and `diagnosis/bank.py` refuse these names. Remove the check, or empty the list.
- **The 10 rules themselves.** They were never written, because they were refused. Each needs a rule function that follows MalruleLib's `calculate_expected_answer()` (decision 8), a `bank.yaml` entry with its Eedi ID, a topic, and tests. A rule that ties with an existing rule needs probe items (`eval/build_probes.py`).
- **Evaluation code that uses the held-back set.** Update or retire these: `evaluation/held_back.py`, `metrics.py`, `students.py`, `prepost.py`, `adversarial.py`, `log_fields.py`, `eval/select_held_back.py`, `eval/match_eedi.py`, `eval/eedi_pick.py`, and their tests (`test_evaluation*.py`, `test_metrics.py`, `test_students_week2.py`, `test_b1.py`, `test_diagnosis_rules.py`).
- **"Unseen misconception" results need a new source.** The held-back rules were how we measured the Eedi and LLM stage on mistakes no rule covers. Use instead mistakes no rule covers at all, for example Eedi misconceptions with no MalruleLib counterpart, or the bracket cases in T2 measured before their rules are added.
- **Docs.** `docs/diagnosis_rules.md`, `docs/Mid_Submission_Report.md` and this file (sections 7.3 and 8) mention the held-back set.

### T2. Rules for bracket and like-term mistakes

**The problem.** Our 58 rules re-implement MalruleLib (decision 8), and MalruleLib has no rule for the most common bracket mistakes. Its whole algebra folder is six rules: `cancel_across_equals`, `change_side_change_sign`, `distribute_over_non_distributive`, `divide_one_term_only`, `forget_negative_division` and `variable_letter_has_value`. `distribute_over_non_distributive` is a different mistake: spreading an operation that does not spread, such as a square root over a sum. Measured on the current code:

| Student's step | Today |
|---|---|
| `2(x + 3) → 2x + 3` | `unknown` |
| `-(x + 1) → -x + 1` | `unknown` |
| `2(x + 3) - (x + 1) → 2x + 3 - x + 1` | `unknown` (two mistakes in one line) |

**What that costs.**
- **With the LLM on,** each such step costs one LLM call, and the result depends on the model's reading and on the Eedi pick. These steps should be found exactly and for free.
- **With the LLM off,** nothing is found. The step has no misconception and no topic, the student's memory does not change, and the hint is the general template (see T3).

**The rules to add.** Each takes the student's expression and returns the answer the mistake predicts, or `None` when it does not apply, like every other rule (`diagnosis/rules/`). Use `rewrite()` from `rules/_common.py`, so a rule also fires on a bracket inside a longer line.

| Proposed name | The mistake | Example: question → what the rule predicts |
|---|---|---|
| `brackets.multiply_first_term_only` | multiplies only the first term inside the bracket | `2(x + 3)` → `2x + 3`; `3(a - b + 4)` → `3a - b + 4` |
| `brackets.minus_first_term_only` | a minus in front of a bracket changes only the first term's sign | `-(x + 1)` → `-x + 1`; `5 - (x - 2)` → `5 - x - 2` |
| `brackets.negative_multiplier_first_term_only` | multiplies by the size of a negative number but applies its sign only to the first term | `-2(x - 3)` → `-2x - 6` (right: `-2x + 6`) |
| `like_terms.add_letters_as_powers` | adds like terms by multiplying the letters | `x + x` → `x^2`; `2x + 3x` → `5x^2` |
| `like_terms.combine_unlike_terms` | adds a number to a letter term | `3x + 2` → `5x`; `4a + 3b` → `7ab` |

**How they must fit with what exists.**
- **Names.** These are our own rules, not MalruleLib's. Decision 8 says the rules keep MalruleLib's names, so update it: rules with no MalruleLib counterpart use the groups `brackets.` and `like_terms.`, which MalruleLib does not use, and are marked as ours. Check that `rule_id()` accepts the new groups.
- **Topics.** Add the groups to `_GROUP_TOPIC` in `diagnosis/explain.py`: `brackets` → `expanding_brackets`, `like_terms` → `collecting_like_terms`. `test_every_rule_has_a_topic_in_the_list` checks this.
- **Eedi.** Give each rule its Eedi ID in `bank.yaml` (Eedi has names for most of these mistakes; `eval/match_eedi.py` helps find them), so an Eedi pick of the same mistake is recorded as the rule (one ID per mistake).
- **The slip check.** `diagnosis/slips.py` stays as it is. A rule describes a *consistent* mistake: `2(x + y + z) → 2x + y + z` (no term after the first multiplied) is `multiply_first_term_only`. The slip check describes an *inconsistent* one: `−2(x + y − z − k) → −2x − 2y + 2z − 2k` (−z right, −k wrong) is a slip. Rules run first, and the slip check only runs when no rule fits, so the two do not overlap. Add one test of each kind.
- **Ties and probes.** If a new rule gives the same answer as an existing rule on some question, the two tie. `eval/build_probes.py` must find a probe that tells them apart, or they are twins (decision 8).
- **Tests.** Follow `tests/test_diagnosis_rules.py`. For each rule: the examples above, a case where it returns `None`, and the whole step diagnosed through `diagnose()`.

**What "done" means.** The first two steps in the first table are diagnosed `clear` by a rule, with no LLM call, and the full suite passes. The third has two mistakes in one line, so no single rule fits it (multiple misconceptions are future work). With the LLM on, though, each step of the model's reconstructed working is then matched by a rule.

### T3. Testing with only the diagnoser's LLM off

**Why.** One switch (`--llm` in the page and the runner) turns the LLM on or off for everything: reading the student's lines, diagnosis (the explainer, and through it the Eedi pick), the hinter and the critic. To learn what the LLM adds *to diagnosis alone*, turn off only the diagnoser and keep everything else the same. This comparison is rules, solve back and the slip check, against all of those plus the LLM.

**What to build.**
- A switch, for example `--no-llm-diagnosis`, in `ui/__main__.py` and `tutor/runner.py`. It calls `diagnose()` with `session=None`, while the transcriber, hinter and critic keep their session. With the LLM off for diagnosis, the Eedi ranker is unused (it only supplies the LLM's options), so it need not be loaded.
- **A known gap to fix first.** When no rule matches and the diagnoser's LLM is off, the diagnosis has no topics. The wrong answer then changes nothing in the student's memory: no mastery update, nothing on the chart, no Revise card and no similar problems. Fix: when nothing names a topic, take the topics from the wrong part that `locate()` found, using `question_topics()` (`diagnosis/topics.py`), which correct answers already use for credit. For example, `2(x + 3)` gives `expanding_brackets` and `algebraic_notation`. For a line turned into one number, use solve back's wrong piece: `5 × 6 → 26` gives `whole_number_arithmetic`.

**What to measure, with the diagnoser's LLM on and off.**
- share of wrong steps diagnosed `clear`, `ambiguous` and `unknown`;
- diagnosis accuracy on steps whose misconception is known (rule-generated items);
- topics named per wrong step, and how often memory changed;
- hint critic pass rate, and how often the safe template was used;
- LLM calls and tokens per session.

Run it before and after T2, since the bracket rules should move many `unknown` steps to `clear`.

### T4. B0 on LangChain/LangGraph

B0 (`tutor/baselines.py`, `B0Tutor`) is one prompted tutor: each turn it reads the student's line, makes one LLM call, and sends the message. In LangGraph it is a small `StateGraph`: a reading node, a tutor node (the LLM call) and a delivery node, with the conversation kept in the graph's state.

- **Every model call stays on our client.** Calls must go through our `LLMClient` session (`state.llm.call`), wrapped as a LangChain chat model if LangChain is used, so that the per-session token budget, the cache and the run logs stay exactly as they are. Do not use LangChain's own Gemini wrapper: it skips all three, and the conditions would no longer be comparable.
- **Same outputs as now.** Keep B0's prompt, schema and log records (`diagnosis` with `stated_misconception`, and `hint`), so the evaluation metrics read it unchanged. Port B0's tests and re-run the B0 baseline.
- Pin the `langchain-core` and `langgraph` versions in `requirements.txt`.

### T5. B1 on LangGraph

Rebuild `B1Agent`'s tool loop as a LangGraph `StateGraph`: an agent node (the LLM call), a tool node, and the message step, joined by a conditional edge on the reply's `action`, with at most `MAX_TOOL_CALLS` tool calls. The same rules as T4 apply:
- every model call goes through our `LLMClient` session;
- every tool call goes through `ToolBox.call`, which the tool-call metric reads;
- the versions are pinned.

Port `tests/test_b1.py` and re-run the B1 baseline. Only the baselines use LangGraph; the full tutor and diagnosis stay as plain code.

### T6. Every hint rejected: wrong misconception given to the hinter, and a critic that is too strict

**The case** (page, LLM and Eedi on). The question is `1/2 + 1/3 + 1/6 + 2^2` (answer 5), and the student wrote `1`: the fractions are right (they sum to 1), but `2^2 = 4` was dropped. The hinter wrote three hints, and the LLM critic rejected all three, so the student got the safe template. Attempt 2 was a good hint:

> You correctly identified that the fractions add up to 1, but you still need to include the value of the exponent. Remember that 2^2 means 2 multiplied by itself. What is the total value when you add 1 to the result of 2^2?

The critic rejected it with "gives away the result of the fraction addition", "gives away the result of the exponent calculation" and "performs the final addition for the student". Attempts 1 and 3 were about "any number divided by itself is one" and were rightly rejected as not relevant.

**What happened, from the LLM logs (`runs/llm_logs/ui.jsonl`) and the attempt saved in the page database.**
1. **The explainer misread the step, and code caught it.** Its reconstructed working marked `1/2 + 1/3 + 1/6 + 4 → 1 + 4` as the mistake, which is a correct step, and marked `1 + 4 → 1`, which is the real mistake, as right. `_reading()` therefore returned `unverified` ("step … is right, but the model says it is a mistake").
2. **The Eedi pick was used anyway (a bug in diagnosis).** `diagnose()` takes the explainer's Eedi pick whenever `kind == "misconception"` (`diagnosis/diagnose.py`, the `elif explanation.eedi is not None …` branch), without checking that the explanation is `verified`. So the step was recorded as `eedi:413`, "Believes that n/n = n, instead of 1". That misconception was unrelated, and even contradicted by the student's work, but it was passed to the hinter and to the critic as the diagnosis.
3. **The hinter was told to aim at that misconception.** Attempts 1 and 3 did, and the critic correctly rejected them (rule 3: aimed at the misconception, but it does not fit). Attempt 2 ignored it and addressed the real mistake.
4. **The critic's rule 2 is read too strictly.** It forbids "work out the correct result of the step". The critic only sees the question, the correct answer, the student's answer, the misconception and the hint. It cannot see that the student already has `1` for the fractions, so restating the student's own right result looks to it like giving the answer away. Explaining what `2^2` means (not its value) was counted as a leak too.
5. **The rewrite loop cannot recover.** Each rewrite gets the critic's reasons, but the wrong misconception stays in the hinter's prompt. So after `MAX_REWRITES` (2) it ends on the safe template.

**What to fix.**
1. **Use an Eedi pick only from a verified explanation.** In `diagnose()`, require `explanation.status == "verified"` before an Eedi pick becomes the diagnosis. Consider requiring `eedi_fit == "exact"` too: the pick test (`eval/eedi_pick.py`) found exact picks acceptable 13 of 14 times, but close picks only 3 of 8. Otherwise the misconception is `None`, and the hinter gets "not known".
2. **Give the critic what the student got right.** Add "What went wrong" (the located part, as the hinter already gets) and, when known, the parts of the student's work that are right, to the critic's prompt. Reword rule 2: the hint must not give the final answer or the corrected result of the *wrong* part. It may restate what the student already worked out correctly, and explain what a notation means (`2^2` means 2 × 2) without giving its value.
3. **Tell the hinter how sure the diagnosis is.** Pass the misconception's source and confidence in `HintRequest`: a rule reproduces the step exactly, or an Eedi pick (exact or close), or none. When it is not certain, the hinter should aim at the located mistake and treat the misconception as a possibility only. The critic's rule 3 should then not require the hint to aim at an uncertain misconception.
4. **Let a rewrite drop the misconception.** If the critic says the misconception is not relevant, the next attempt should be written without it ("help the student check their own work"), instead of being told to aim at it again.

**Test.** Add this case as a regression test with scripted model replies:
- the explainer's unverified working above must not produce an Eedi diagnosis;
- attempt 2's hint must pass a critic given the new inputs.

Then check it once on the real model, which costs at most about 6 calls.

### T7. Group near-duplicate Eedi names

**Why.** The LLM picks one Eedi name from the shortlister's 20 (`EEDI_K`). Eedi has many names that describe the same mistake in slightly different words. They take up places in the 20, split the model's choice, and make a right pick look wrong in evaluation. In the pick test (`eval/eedi_pick.py`, 22 cases), the right name was in the list 16 times, but the pick matched it exactly only 5 times, while 16 picks were judged acceptable. Much of that gap is near-duplicates.

**Measured, and why grouping by similarity alone is wrong.** Using the saved name vectors (927 live candidates), 113 pairs have a cosine similarity above 0.90, touching 143 names (above 0.95: 28 pairs). But high similarity does not mean the same mistake. Even pairs above 0.98 are often different, or opposite, beliefs:

| Similarity | Pair | Same mistake? |
|---|---|---|
| 0.977 | "Carries out operations from left to right regardless of priority order" / "…, unless brackets are used" | yes |
| 0.954 | ratio: "uses the difference as the total" / "…as the total for one side instead" | yes |
| 0.983 | subtracting fractions: "subtracts the numerators and denominators" / "subtracts the numerators and multiplies the denominators" | **no** |
| 0.984 | eliminating a variable: "the operation inverse to the sign" / "the operation matching the sign" | **no, opposites** |

Merging by a similarity threshold would merge different misconceptions, which is worse than not grouping.

**What to do.**
1. **Find candidate pairs.** Use vector similarity above 0.90, plus large wording overlap.
2. **Judge each pair** "same mistake or not", and have a second person check the list. Do this by hand (about 113 pairs) or with an LLM in batches, then check by hand.
3. **Use the groups.**
   - Write them to a group file. It stays out of git, like every Eedi-derived file.
   - In `eedi.options()`, show one name per group and give the freed places to the next names.
   - A pick counts as its group, both in the diagnosis and in evaluation.
   - `bank.for_eedi()` checks every member of a group, so a rule mapped to one member owns the group.
4. **Re-score** the 22 existing pick results counting group matches (no API calls). Re-running the pick test live is optional (22 calls).

Time: about 1 to 1.5 hours.

**Status, 11 November (A): steps 1 and 2 done, waiting for a second check.**
- **The tool.** `shortlister/name_groups.py` lists the candidate pairs, builds the groups and provides `group_of()`.
- **Candidate pairs.** The search used model2vec vectors, the CPU model `retrieve()` uses, because the v2 vectors need a GPU. A pair needs cosine ≥ 0.90 *and* word overlap ≥ 0.5, which gives 156 pairs among the 972 in-topic names.
- **Judged by A: 15 pairs are the same mistake, so 15 groups of 2.** The other 141 are opposites ("rounds up" / "rounds down"), the same mistake in a different operation (lining up digits when adding / when subtracting), or different mistakes in one topic.
- **Rule for subsets.** When one name is a narrower case of the other ("adds or subtracts from inside brackets" / "subtracts from inside brackets"), the pair is counted as the same.
- **Not done.** The second check, step 3 (using the groups in `eedi.options()`, scoring and `bank.for_eedi()`, which is B's code), and step 4.
- **What it is likely to change.** Some near-duplicates have little wording in common and are missed by this search, so B's v2-based count of 113 similar pairs is not comparable. With only 15 groups, the effect on the pick test is likely to be small.

### T8. The LLM's own "slip" verdict becomes a slip diagnosis

**Why.** The explainer returns `kind`: `misconception` or `slip`. Today a verified explanation with `kind == "slip"` gives *no* misconception at all. The diagnosis is `unknown`, so the tutor treats the mistake as unexplained, rather than as a careless slip. Example: `7 × 8 + 6 → 60` (a times-table slip, 7 × 8 taken as 54). Code already turns two kinds of evidence into a slip:
- the bracket check (`diagnosis/slips.py`);
- wrong steps in the LLM's working that are all bracket slips.

The LLM's own verdict is not used.

**What to do.** In `diagnose()`, after the checks on the LLM's working: if the explanation is `verified`, `kind == "slip"`, and no rule reproduces any of its wrong steps, then the diagnosis is `slip`. Add a trace stage `slip` whose reason is the explainer's `mistake` text, so the page's "Why a slip" card shows it. An unverified explanation changes nothing. The slip handling already exists: mastery moves by a quarter of a normal mistake (`SLIP_WEIGHT`), the mistake is not recorded as a habit, and the hint is general.

**Check.** A test with a scripted model reply: verified with `kind: slip`, unverified with `kind: slip`, and verified with `kind: slip` where a rule reproduces a step (the rule wins). Time: about 15 to 20 minutes.

### T9. Slips from counting each rule's right and wrong uses

**Why.** A misconception is applied the same way every time; a slip is not. If the same student does an operation right in one place and wrong in another *in the same answer*, the wrong one is probably a slip. Today only brackets are checked this way (`slips.py`). This item generalises the idea to all the rules.

**The idea: count chances, not mistakes.** For every part of every step, each rule that would give a wrong answer there is a *chance* to make that mistake. Every rule returns `None` where its mistake does not apply, so there is no table of which rule goes with which operation: running all the rules on a part finds its chances. Measured: about 290 rule runs per line, in under 0.2 s. Each chance then becomes:
- **right**, when the student worked that part out and got it right: the step is correct, or the part is outside the part `locate()` found wrong. A part copied unchanged to the next line was not worked out, and is not counted;
- **wrong**, when the rule reproduces what the student wrote for that part.

Per rule, over the whole answer:

| Right | Wrong | Verdict |
|---|---|---|
| ≥ 1 | 1 | slip |
| 0 | ≥ 1 | misconception |
| any | ≥ 2 | misconception (repeated) |

**What works today, tried on the current code.**

| Case | Example | Works? |
|---|---|---|
| Across steps | line 1 `1/2 + 1/3 + 1/4 → 5/6 + 1/4` (right); line 2 `5/6 + 1/4 → 6/10` (wrong) | yes: line 2 matches `fractions.add_numerators_denominators`, and the same rule fires on `1/2 + 1/3` in line 1 |
| Same line, letter terms | `x^2 × x^3 + x^4 × x^2 → x^5 + x^8` | yes: `locate()` sets aside `x^5` as right, and the rule catches `x^4 × x^2 → x^8` |
| Same line, numbers | `1/2 + 1/3 + 1/4 + 1/5 → 5/6 + 2/9` | no: numbers combine, so the line is not split into parts (it needs the line-splitting in `docs/plan_multiple_misconceptions.md`) |
| A mistake no rule covers | the sign error in `−2 × −k` | no: the bracket check stays for it, and T2's rules would add more chances |

**Three details the trial showed.**
1. **Count each chance once, at the smallest part.** Rules apply wherever they fit inside an expression, so `multiply_base_by_exponent` also fires on a whole line that contains `2^3`.
2. **Skip typed fractions.** `1/2` is a number, not a division, but `divide_larger_by_smaller_always` treats it as one. Solve back already skips fractions this way.
3. **Equation rules** (`divide_one_term_only` and others) fire on the whole equation. They count once per line.

**Where it goes.**
- `rule_uses(chain)` in `diagnosis/slips.py`, called once per answer by the page (`ui/__main__.py`) and the orchestrator.
- Optionally, keep the counts in the student's memory too. Then a suspected misconception fades only after a right answer where that rule actually had a chance to fire, instead of after any right answer in its topic, as now.

No LLM call is needed. Time: about half a day, with tests.

## Future work

| Item | Why it is not done now | Plan |
|---|---|---|
| **Multiple misconceptions in one answer** | Only the first wrong line is diagnosed, and two mistakes in one line match no rule, so the LLM reports only one of them. Example: `x^2 × x^3 + 2^3 → x^6 + 6` is `unknown`, although each half alone is caught by a rule. Put on hold on 2026-11-09 | `docs/plan_multiple_misconceptions.md` |
