# Design decisions: reading student maths

Owner: B. Covers how the diagnosis engine reads what a student writes: the parser, the notation standard and the LLM transcriber (`matheasy/diagnosis/maths.py`, `transcribe.py`, `docs/input_notation.md`), and how a located mistake is handed to A's shortlister (`matheasy/diagnosis/interfaces.py`, `matheasy/shortlister/`). Each entry gives the decision, the alternatives considered, and why. Open questions are listed at the end.

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

Until the model is trained, `shortlist()` uses BM25 word matching over the names, and `diagnose()` uses `RuleShortlister` (every rule is run, and the ones that reproduce the student's answer are kept).

### 7.4 Training and live queries must match

The model only works well on queries shaped like the ones it was trained on. Today the two are built differently, because Eedi's data does not fit `locate()`. Its maths is LaTeX (`\frac{1}{2}`), and many questions are word problems or multiple choice, with no step to narrow down. So in training, `build_query()` fills in the missing fields itself (`fallback_diff()`).

| Field | Training (Eedi) | Live (our `locate()`) |
|---|---|---|
| Topic | our topic plus Eedi's construct name, e.g. `fractions - Add fractions with different denominators` | `fractions` only |
| Before, Student wrote, Correct | LaTeX: `\frac{2}{5}` | our notation: `2/5` |
| Changed part | a text diff: `5 -> 2; 6 -> 5` | the wrong term: `5/6 -> 2/5` |
| Difference | `sympy_diff()` on the whole answers | `sympy_diff()` on the narrowed part (the same for most Eedi rows) |

| Considered | Outcome |
|---|---|
| Live queries copy the training format | Possible, but it throws away the sharper wrong-term signal |
| **Training converts the LaTeX of plain-expression Eedi rows to our notation and runs `locate()`. The text diff is kept only for rows we cannot read, and used the same way live** | **Proposed**, to agree with A before the full training run. First step: count how many Eedi rows are plain expressions |

**The construct name in Topic.** In training it is the model's strongest clue, and nothing provides it live.

| Considered | Outcome |
|---|---|
| Drop the construct name from training, so both sides use only our topic | **Candidate**: no labels needed, but a weaker clue |
| Give every tutor question a hand-written skill label in Eedi's style, passed to `diagnose()` and on into the query | **Candidate**: the clue stays, at the cost of labelling the question bank |
| Generate the phrase in code from the narrowed part | Rejected: hand-written rules mapping operations to phrases would be brittle |

To decide between the two candidates, A trains with and without the construct name and compares MAP@25 on validation.

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
