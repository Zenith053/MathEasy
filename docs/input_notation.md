# Standard maths notation

Owner: B. Every maths line that reaches `parse()` should be written in this notation: student answers after `transcribe()`, D's test items and probes, and anything our own code prints. The parser reads every form below the same way every time; forms outside the standard are either rejected or given one fixed reading, which may not be what the writer meant.

Students type maths, not sentences. Sentences and numbers in words ("three and a half times two", "one over fifth") are not accepted: the student is asked to type the answer as maths.

## Rules

| # | Rule | Standard | Not standard → write instead |
|---|---|---|---|
| 1 | One statement per line, at most one `=`, `<`, `>`, `<=`, `>=` | `x + 3 = 5` | `x = 2, y = 3` → two lines |
| 2 | Single letters are variables; letters side by side multiply | `3ab`, `2xy`, `A = 30 × 20` | |
| 3 | A quantity word (from `matheasy/diagnosis/quantity_words.txt`, any case) is one quantity; any other word is written in braces | `Area = 2πR`, `2 × Area`, `{fuel} × 3` | `fuel × 3` → `{fuel} × 3` |
| 4 | Letters written together, 4 or more, that are not a quantity word are letters multiplied, but the writer must make it clear | `x × y × z × w`, `{fuel}` | `xyzw` → `x × y × z × w`, or `{xyzw}` if it is one quantity |
| 5 | π is the symbol `π`; `22/7` and `3.14` are ordinary numbers | `2πr`, `πr^2`, `22/7 × 7^2` | |
| 6 | An exponent of more than one number or letter goes in brackets | `x^2`, `2^n`, `a^(bc)`, `(x + 1)^(xy)` | `a^bc` → `a^(bc)` |
| 7 | Nothing written straight after an exponent | `yx^2`, `x^2 × y` | `x^2y` → `x^2 × y` |
| 8 | A power applies only to the letter, number or bracket just before it | `xy^2` is x·y², `(xy)^2` is x²y² | |
| 9 | Letters written together are one term, so a divisor of letters needs no brackets; a divisor that mixes numbers and letters goes in brackets | `a/bc` (means a/(bc)), `a/(bc)`, `x/2`, `2x/3`, `1/2 × x` | `1/2x` → `1/2 × x` or `x/2`; `x/2y` → `xy/2` or `x/(2y)` |
| 10 | A division straight after an exponent goes in brackets if it belongs to the exponent, or is written with ÷ if it does not | `x^(2/3)`, `e^(2/a)`, `x^2 ÷ 3`, `e^2 ÷ a` | `x^2/3` → `x^(2/3)` or `x^2 ÷ 3`; `e^2/a` → `e^(2/a)` or `e^2 ÷ a` |
| 11 | A fraction of whole numbers, or a mixed number, is one number | `2/3 ÷ 4/5`, `6 ÷ 2/3`, `3 1/2` | |
| 12 | `%` follows a number; `of` after a number or percentage means × | `25%`, `50% of 20`, `1/2 of 12`, `25% × x` | `25% x` → `25% × x` |
| 13 | Signs | `-x`, `6 ÷ -2`, `x - (-3)` | |
| 14 | Symbols | `+ - × ÷ * / ^`, brackets, `= < > <= >=`, `π`, `{ }` around a word, decimals with a point | `,` as a separator, sentences, other symbols |

A word can never stand right next to a letter or bracket: write `Area × x` and `{cost} × (n + 1)`. A number may stand before a word: `2Area` means 2 × Area.

## How a student's line is read (`transcribe()`)

1. **Standard already?** If `notation_issue(line)` finds nothing, the line is used as it is. No model call.
2. **Word or letters multiplied?** For each unknown run of 4 or more letters (`xyzw`, `fuel`), the model answers only "word" or "product". Code checks it answered exactly the runs asked, then writes the answer into the line: `{fuel}`, or `x × y × z × w`.
3. **Ambiguous notation?** For `a^bc`, `x/2y`, `1/2x`, `x^2/3`, `e^2/a` and lines the parser cannot read, the model rewrites the line with brackets to say what the student meant, or says the line is not maths. Code accepts the rewrite only if every number and letter is the student's, in the same order: only brackets, `×` and spaces may change. If the model is unsure, the student is asked "did you mean 1) … or 2) …?" and `resolve()` reads the reply.

The model sees the question and the student's line, never the correct answer.

## What the parser does with forms outside the standard

These are not standard, so neither `transcribe()` nor D's items should produce them. Today the parser accepts some of them with one fixed reading:

| Written | Read as | Standard form |
|---|---|---|
| `1/2x` | x/2 | `1/2 × x` |
| `x/2y` | (x/2)·y | `xy/2` or `x/(2y)`, whichever is meant |
| `a^bc`, `(x + 1)^ab` | a^b·c, (x + 1)^a·b | `a^(bc)` or `a^b × c` |
| `x^2y` | x²·y | `x^2 × y` |
| `x^2/3` | x^(2/3) | `x^(2/3)` or `x^2 ÷ 3` |
| `e^2/a` | e²/a | `e^(2/a)` or `e^2 ÷ a` |
| `xyzw`, `fuel` | letters multiplied | `x × y × z × w`, or `{fuel}` for one quantity |
| `x = 2, y = 3`, `x = y = 2` | rejected | one statement per line |

## Checking a line

`parse(line)` raises `ParseError`, with the reason and position, for anything it cannot read. `notation_issue(line)` says why a line is not standard, or `None`. `str(parse(line))` prints the line back in standard notation; printing and reading back never changes the value (tested on about 40,000 generated expressions, including words and π).

The tests in `tests/test_diagnosis_maths.py` (`test_standard_notation`, `test_notation_means`) and `tests/test_transcribe.py` check the examples in this document.
