"""
Query formatter for the shortlister (Role A).

build_query() turns one located error into the text the shortlister ranks
misconceptions against:

  Topic: ... | Before: ... | Student wrote: ... | Correct: ... | Changed part: ... | Difference: ...

The same function builds the training queries from Eedi and the live queries in
the tutor, so the two never drift apart. In the tutor, B's locate() supplies the
changed part and the difference. When they are not given (Eedi rows, or an
answer locate() cannot handle), they come from fallback_diff(): the SymPy
difference when both answers parse, and a plain text diff when they do not.

Usage: python -m matheasy.shortlister.query
  -> reports how many Eedi wrong answers parse, and writes data/eedi/queries.csv
"""

import re
import warnings
from difflib import SequenceMatcher
from pathlib import Path

import sympy
from sympy.parsing.latex import parse_latex

UNPARSED = "does not parse"


def clean(text):
    """Strip LaTeX delimiters, colours and spacing commands; keep the maths."""
    text = str(text)
    text = re.sub(r"!\[(.*?)\]\([^)]*\)", r"[image: \1]", text)
    text = re.sub(r"\\color\{\w+\}", "", text)
    text = re.sub(r"\\[()\[\]]", " ", text)
    text = re.sub(r"\\[;,!:]|\\left|\\right", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse(text):
    """Best-effort SymPy value of an answer, or None. Only used for the difference."""
    s = clean(text)
    # Words ("Variable", "Both Tom and Katie") would parse as products of letters.
    if re.search(r"(?<![\\a-zA-Z])[a-zA-Z]{2,}", re.sub(r"\\[a-zA-Z]+", " ", s)):
        return None
    s = re.sub(r"£|\\pounds|\\\$|\\%|%", "", s)  # units: both answers share them
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)  # 300,000
    s = re.sub(r"^\\(bigcirc|bigstar|square|triangle)\s*=", "", s).strip()  # "\bigcirc = -9"
    # Mixed numbers: the parser reads "1 \frac{1}{2}" as 1/2.
    s = re.sub(r"(\d+)\s*(\\frac\{\d+\}\{\d+\})", r"(\1+\2)", s)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e = parse_latex(s, backend="lark")
    except Exception:
        return None
    if not isinstance(e, (sympy.Expr, sympy.Equality)):
        return None
    # Decimals as exact rationals, so 0.3 - 0.25 is 1/20, not 0.04999...
    return e.replace(lambda x: x.is_Float, lambda x: sympy.Rational(str(x)))


def text_diff(student, correct):
    """The tokens that differ, as 'correct -> student' edits."""
    tok = lambda s: re.findall(r"\d+\.?\d*|\\[a-zA-Z]+|[a-zA-Z]+|\S", clean(s))
    a, b = tok(correct), tok(student)
    if len(a) == len(b):  # same shape: compare position by position, so swaps read as swaps
        return "; ".join(f"{x} -> {y}" for x, y in zip(a, b) if x != y) or "(same text)"
    edits = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op != "equal":
            edits.append(f"{' '.join(a[i1:i2]) or '(nothing)'} -> {' '.join(b[j1:j2]) or '(nothing)'}")
    return "; ".join(edits) or "(same text)"


def sympy_diff(s, c):
    """How the student's value differs from the correct one, or None if not comparable."""
    if isinstance(s, sympy.Equality) and isinstance(c, sympy.Equality):
        if s.lhs != c.lhs:
            return None
        s, c = s.rhs, c.rhs
    if isinstance(s, sympy.Equality) or isinstance(c, sympy.Equality):
        return None
    try:
        d = sympy.simplify(s - c)
        if d == 0:
            return "equivalent to the correct answer"
        out = f"student - correct = {sympy.sstr(d)}"
        if s.is_number and c.is_number and c != 0:
            out += f"; student / correct = {sympy.sstr(sympy.nsimplify(s / c))}"
        return out
    except Exception:  # e.g. a parsed list "3, 5" cannot be subtracted
        return None


def fallback_diff(student, correct):
    """(changed part, difference) when locate() is not available."""
    changed = text_diff(student, correct)
    s, c = parse(student), parse(correct)
    if s is None or c is None:
        return changed, UNPARSED
    return changed, sympy_diff(s, c) or UNPARSED


def build_query(topic, before, student, correct, changed=None, difference=None):
    """The shortlister query for one wrong step. Pass changed/difference from locate() when known."""
    if changed is None or difference is None:
        fb_changed, fb_difference = fallback_diff(student, correct)
        changed = fb_changed if changed is None else changed
        difference = fb_difference if difference is None else difference
    fields = [("Topic", topic), ("Before", before), ("Student wrote", student),
              ("Correct", correct), ("Changed part", changed), ("Difference", difference)]
    return " | ".join(f"{name}: {clean(value)}" for name, value in fields)


def eedi_query(row):
    """Query for one row of data/eedi/pairs.csv."""
    return build_query(
        topic=f"{row['Topic']} - {row['ConstructName']}",
        before=row["QuestionText"],
        student=row["WrongAnswerText"],
        correct=row["CorrectAnswerText"],
    )


if __name__ == "__main__":
    import pandas as pd

    data = Path(__file__).resolve().parents[2] / "data" / "eedi"
    pairs = pd.read_csv(data / "pairs.csv")
    pairs["Query"] = pairs.apply(eedi_query, axis=1)
    pairs["StudentParses"] = pairs.WrongAnswerText.map(lambda t: parse(t) is not None)
    pairs["CorrectParses"] = pairs.CorrectAnswerText.map(lambda t: parse(t) is not None)
    pairs["Parsed"] = ~pairs.Query.str.endswith(f"Difference: {UNPARSED}")

    by_q = pairs.groupby("QuestionId").agg(Topic=("Topic", "first"), Parsed=("Parsed", "all"))
    print("Share that parse (both answers parse and SymPy can compare them):\n")
    print(pd.DataFrame({
        "wrong answers": pairs.groupby("Topic").Parsed.mean(),
        "questions (all wrong answers)": by_q.groupby("Topic").Parsed.mean(),
    }).map("{:.0%}".format).to_string(), "\n")
    print(f"overall: {pairs.Parsed.mean():.0%} of {len(pairs)} wrong answers, "
          f"{by_q.Parsed.mean():.0%} of {len(by_q)} questions")
    print(f"student answer parses: {pairs.StudentParses.mean():.0%}, "
          f"correct answer parses: {pairs.CorrectParses.mean():.0%}")

    out = data / "queries.csv"
    pairs[["QuestionId", "Split", "Topic", "WrongAnswer", "MisconceptionId",
           "MisconceptionName", "Parsed", "Query"]].to_csv(out, index=False)
    print(f"-> {out}")
