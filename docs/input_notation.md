# Standard maths notation

Owner: B. Every maths line that reaches `parse()` should be written in this notation: student answers after the LLM tagger has rewritten them, D's test items and probes, and anything our own code prints. The parser reads every form below the same way every time; forms outside the standard are either rejected or given one fixed reading, which may not be what the writer meant.

The rule for the LLM tagger: rewrite the student's message into one line of this notation, using brackets wherever a group is meant. Never compute or correct anything; only transcribe.

## Rules

| # | Rule | Standard | Not standard → write instead |
|---|---|---|---|
| 1 | One statement per line, at most one `=`, `<`, `>`, `<=`, `>=` | `x + 3 = 5` | `x = 2, y = 3` → two lines |
| 2 | Variables are single letters; letters side by side multiply | `3ab`, `2xy`, `A = 30 × 20` | `area = 446` → `A = 446` |
| 3 | An exponent of more than one number or letter goes in brackets | `x^2`, `2^n`, `a^(bc)`, `(x + 1)^(xy)` | `a^bc` → `a^(bc)` |
| 4 | Nothing written straight after an exponent | `yx^2`, `x^2 × y` | `x^2y` → `x^2 × y` |
| 5 | A power applies only to the letter, number or bracket just before it | `xy^2` is x·y², `(xy)^2` is x²y² | |
| 6 | A divisor of more than one number or letter goes in brackets | `a/(bc)`, `x/2`, `2x/3`, `1/2 × x` | `a/bc` → `a/(bc)`; `1/2x` → `1/2 × x` or `x/2`; `x/2y` → `xy/2` |
| 7 | A fraction in an exponent goes in brackets | `x^(2/3)`, `x^2 ÷ 3` | `x^2/3` → `x^(2/3)` or `x^2 ÷ 3` |
| 8 | A fraction of whole numbers, or a mixed number, is one number | `2/3 ÷ 4/5`, `6 ÷ 2/3`, `3 1/2` | |
| 9 | `%` follows a number; `of` after a number or percentage means × | `25%`, `50% of 20`, `1/2 of 12`, `25% × x` | `25% x` → `25% × x` |
| 10 | Signs | `-x`, `6 ÷ -2`, `x - (-3)` | |
| 11 | Symbols | `+ - × ÷ * / ^`, brackets, `= < > <= >=`, decimals with a point | `,` as a separator, words, other symbols |

## What the parser does with forms outside the standard

These are not standard, so neither the tagger nor D's items should produce them. Today the parser accepts some of them with one fixed reading:

| Written | Read as | Standard form |
|---|---|---|
| `a/bc` | a/(bc) | `a/(bc)` |
| `1/2x` | x/2 | `1/2 × x` |
| `x/2y` | (x/2)·y | `xy/2` or `x/(2y)`, whichever is meant |
| `a^bc`, `(x + 1)^ab` | a^b·c, (x + 1)^a·b | `a^(bc)` or `a^b × c` |
| `x^2y` | x²·y | `x^2 × y` |
| `x^2/3` | x^(2/3) | `x^(2/3)` or `x^2 ÷ 3` |
| `area` | a·r·e·a (a product of letters) | a single letter, e.g. `A` |
| `x = 2, y = 3`, `x = y = 2` | rejected | one statement per line |

## Checking a line

`parse(line)` raises `ParseError`, with the reason and position, for anything it cannot read. `str(parse(line))` prints the line back in standard notation; printing and reading back never changes the value (tested on about 40,000 generated expressions).

The tests in `tests/test_diagnosis_maths.py` (`test_standard_notation`) check every standard example in this document by value.
