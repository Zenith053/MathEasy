"""
Eedi rows rewritten as live-tutor queries (Role A, docs/design_decisions.md 7.4).

Training queries keep Eedi's LaTeX and a text diff; live queries use our notation
and the wrong term from locate(). This converts the rows that fit: a maths part of
the question and the wrong answer must both parse, and locate() must find the
step. The query then comes from query() in matheasy/diagnosis/interfaces.py, as in
the tutor. Word problems and similar rows have no step to locate, so they are left
out. Output is Eedi-derived, so it stays in data/ (git-ignored).

Usage: python -m matheasy.shortlister.live_queries
  -> data/eedi/live_queries.csv  (QuestionId, Split, Topic, WrongAnswer, MisconceptionId, Query)
"""

import re
from pathlib import Path

import pandas as pd

from matheasy.diagnosis import ParseError, locate
from matheasy.diagnosis.interfaces import query
from matheasy.diagnosis.maths import parse

from .query import clean

DATA = Path(__file__).resolve().parents[2] / "data" / "eedi"
COLUMNS = ["QuestionId", "Split", "Topic", "WrongAnswer", "MisconceptionId", "Query"]
MATHS_PART = re.compile(r"\\\((.*?)\\\)", flags=re.S)
FRAC = re.compile(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}")
SYMBOLS = {"\\times": "×", "\\div": "÷", "\\cdot": "×", "\\%": "%"}


def to_notation(latex):
    """Eedi LaTeX as our notation: \\frac{1}{2} + \\frac{1}{3} -> 1/2 + 1/3."""
    s = clean(latex)
    for _ in range(4):  # nested fractions, innermost first
        s = FRAC.sub(lambda m: f"({m[1]})/({m[2]})", s)
    s = re.sub(r"\((\d+)\)/\((\d+)\)", r"\1/\2", s)
    for tex, symbol in SYMBOLS.items():
        s = s.replace(tex, symbol)
    s = re.sub(r"\^\{([^{}]*)\}", r"^(\1)", s)
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)  # 300,000
    return s.replace("{", "").replace("}", "").strip()


def _readable(text):
    try:
        parse(text)
        return True
    except ParseError:
        return False


def live_query(question, wrong_answer):
    """The query the tutor would build for this question and wrong answer, or None if it does not fit."""
    wrong = to_notation(wrong_answer)
    if not _readable(wrong):
        return None
    for part in MATHS_PART.findall(str(question)):
        before = to_notation(part)
        if _readable(before):
            error = locate(before, wrong)
            if error is not None:
                return query(error)
    return None


def build(pairs):
    rows = pairs.assign(Query=[live_query(r.QuestionText, r.WrongAnswerText) for r in pairs.itertuples()])
    return rows.dropna(subset=["Query"])[COLUMNS]


if __name__ == "__main__":
    pairs = pd.read_csv(DATA / "pairs.csv")
    live = build(pairs)
    print(f"{len(live)} of {len(pairs)} rows fit the live format ({len(live) / len(pairs):.0%})")
    print(pd.crosstab(live.Topic, live.Split, margins=True).to_string())
    out = DATA / "live_queries.csv"
    live.to_csv(out, index=False)
    print(f"-> {out}")
