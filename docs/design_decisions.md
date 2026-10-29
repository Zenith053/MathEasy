# Design decisions: reading student maths

Owner: B. Covers how the diagnosis engine reads what a student writes: the parser, the notation standard and the LLM transcriber (`matheasy/diagnosis/maths.py`, `transcribe.py`, `docs/input_notation.md`). Each entry gives the decision, the alternatives considered, and why. Open questions are listed at the end.

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

## Open questions

| Question | Who decides |
|---|---|
| Answers using π, 22/7 or 3.14: state the value in each question, or accept all three | B |
| Fixed readings for `1/2x` and `x/2y`, or always ask | B |
| Do baselines B0 and B1 also get the transcriber, so comparisons stay fair? | Team |
| Clarification questions logged separately, so they do not count as diagnostic follow-ups in H1 metrics | C and D |
| Human review of `quantity_words.txt` before the final runs | B |
| D's `item_key()` handles the new `Word` and `Mixed` nodes | D |
