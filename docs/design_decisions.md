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
