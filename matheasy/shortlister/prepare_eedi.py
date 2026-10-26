"""
Build the Eedi subset and its train/val/test split (Role A, week 1).

Keeps the questions on fractions, decimals and algebraic expressions, using the
same topic grouping as D's held-back rules (ratios with fractions; percentages,
rounding and standard form with decimals; negatives, BIDMAS and equations with
expressions). The split is by question, stratified by topic and seeded, so test
questions never appear in training and anyone can re-run it.

Eedi's raw data stays private (competition rules): it is read from data/eedi/raw/
and the outputs go to data/eedi/, both git-ignored. Only the list of question IDs
per split is written to the repo, in matheasy/shortlister/eedi_split.json.

Setup: accept the competition rules on your own Kaggle account, then
  kaggle competitions download -c eedi-mining-misconceptions-in-mathematics -p data/eedi/raw
  (and unzip it there)

Usage: python -m matheasy.shortlister.prepare_eedi
  -> data/eedi/questions.csv  one row per question, with topic and split
  -> data/eedi/pairs.csv      one row per wrong answer that has a misconception label
  -> matheasy/shortlister/eedi_split.json
"""

import json
import random
from pathlib import Path

import pandas as pd

SEED = 20261016  # date of the team plan, same as D's draw
SPLIT = {"train": 0.7, "val": 0.1, "test": 0.2}

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "eedi" / "raw"
OUT = ROOT / "data" / "eedi"

# Eedi SubjectName -> topic. Every subject not listed here is left out.
# Quadratics (double-bracket factorising, algebraic fractions) are out of tutor
# scope, as in D's eval/select_held_back.py.
TOPICS = {
    "fractions": [
        "Adding and Subtracting Fractions",
        "Multiplying Fractions",
        "Dividing Fractions",
        "Fractions of an Amount",
        "Ordering Fractions",
        "Converting Mixed Number and Improper Fractions",
        "Simplifying Fractions",
        "Equivalent Fractions",
        "Writing Ratios",
        "Sharing in a Ratio",
        "Direct Proportion",
        "Indirect (Inverse) Proportion",
    ],
    "decimals": [
        "Place Value",
        "Multiplying and Dividing with Decimals",
        "Adding and Subtracting with Decimals",
        "Ordering Decimals",
        "Converting between Fractions and Decimals",
        "Rounding to Decimal Places",
        "Recurring Decimals to Fractions",
        "Percentages of an Amount",
        "Converting between Fractions and Percentages",
        "Converting between Decimals and Percentages",
        "Percentage Increase and Decrease",
        "Rounding to Significant Figures",
        "Standard Form",
    ],
    "expressions": [
        "Writing Expressions",
        "Substitution into Formula",
        "Simplifying Expressions by Collecting Like Terms",
        "Expanding Single Brackets",
        "Expanding Double Brackets",
        "Multiplying Terms",
        "Factorising into a Single Bracket",
        "BIDMAS",
        "Adding and Subtracting Negative Numbers",
        "Multiplying and Dividing Negative Numbers",
        "Ordering Negative Numbers",
        "Linear Equations",
        "Rearranging Formula and Equations",
    ],
}


def load():
    train = pd.read_csv(RAW / "train.csv")
    names = pd.read_csv(RAW / "misconception_mapping.csv")
    subject_topic = {s: t for t, subjects in TOPICS.items() for s in subjects}
    missing = set(subject_topic) - set(train.SubjectName)
    assert not missing, f"subjects not in Eedi: {missing}"
    q = train[train.SubjectName.isin(subject_topic)].copy()
    q["Topic"] = q.SubjectName.map(subject_topic)
    return q, names


def split(questions):
    """Shuffle each topic's question IDs and cut them by the SPLIT shares."""
    rng = random.Random(SEED)
    out = {}
    for topic in TOPICS:
        ids = sorted(questions.loc[questions.Topic == topic, "QuestionId"])
        rng.shuffle(ids)
        n_train = round(len(ids) * SPLIT["train"])
        n_val = round(len(ids) * SPLIT["val"])
        for name, part in [("train", ids[:n_train]),
                           ("val", ids[n_train:n_train + n_val]),
                           ("test", ids[n_train + n_val:])]:
            out.setdefault(name, []).extend(part)
    return {name: sorted(ids) for name, ids in out.items()}


def to_pairs(questions, names):
    """One row per wrong option that Eedi labels with a misconception."""
    rows = []
    for _, r in questions.iterrows():
        for opt in "ABCD":
            mid = r[f"Misconception{opt}Id"]
            if opt == r.CorrectAnswer or pd.isna(mid):
                continue
            rows.append({
                "QuestionId": r.QuestionId,
                "Split": r.Split,
                "Topic": r.Topic,
                "SubjectName": r.SubjectName,
                "ConstructName": r.ConstructName,
                "QuestionText": r.QuestionText,
                "CorrectAnswer": r.CorrectAnswer,
                "CorrectAnswerText": r[f"Answer{r.CorrectAnswer}Text"],
                "WrongAnswer": opt,
                "WrongAnswerText": r[f"Answer{opt}Text"],
                "MisconceptionId": int(mid),
            })
    pairs = pd.DataFrame(rows)
    return pairs.merge(names, on="MisconceptionId", how="left")


def report(questions, pairs, ids):
    print(f"{len(questions)} questions, {len(pairs)} labelled wrong answers, "
          f"{pairs.MisconceptionId.nunique()} misconceptions "
          f"(of {len(pd.read_csv(RAW / 'misconception_mapping.csv'))} in Eedi)\n")
    print(pd.crosstab(questions.Topic, questions.Split, margins=True)
          [["train", "val", "test", "All"]].to_string(), "\n")
    seen = set(pairs.loc[pairs.Split == "train", "MisconceptionId"])
    for name in ["val", "test"]:
        p = pairs[pairs.Split == name]
        unseen = ~p.MisconceptionId.isin(seen)
        print(f"{name}: {len(p)} pairs, {unseen.sum()} ({unseen.mean():.0%}) "
              f"have a misconception never seen in train")
    overlap = set(ids["train"]) & (set(ids["val"]) | set(ids["test"]))
    assert not overlap, f"questions in more than one split: {overlap}"
    print("no question appears in more than one split")


if __name__ == "__main__":
    questions, names = load()
    ids = split(questions)
    split_of = {qid: name for name, qids in ids.items() for qid in qids}
    questions["Split"] = questions.QuestionId.map(split_of)
    pairs = to_pairs(questions, names)

    OUT.mkdir(parents=True, exist_ok=True)
    questions.to_csv(OUT / "questions.csv", index=False)
    pairs.to_csv(OUT / "pairs.csv", index=False)
    path = Path(__file__).with_name("eedi_split.json")
    path.write_text(json.dumps({"seed": SEED, "shares": SPLIT, "topics": TOPICS,
                                **ids}, indent=2) + "\n")
    report(questions, pairs, ids)
    print(f"-> {OUT / 'questions.csv'}\n-> {OUT / 'pairs.csv'}\n-> {path}")
