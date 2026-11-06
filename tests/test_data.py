"""Split, shortlist() and retrieve() against the built data. Skipped when data/ is not built."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPLIT = ROOT / "matheasy" / "shortlister" / "eedi_split.json"

EEDI = ROOT / "data" / "eedi"
PASSAGES = ROOT / "data" / "openstax" / "passages.jsonl"
needs_eedi = pytest.mark.skipif(not (EEDI / "raw" / "misconception_mapping.csv").exists(),
                                reason="Eedi data not downloaded (python -m matheasy.shortlister.prepare_eedi)")
needs_openstax = pytest.mark.skipif(not PASSAGES.exists(),
                                    reason="passages not built (python -m matheasy.retrieval.openstax)")


def test_split_has_no_question_in_two_splits():
    split = json.loads(SPLIT.read_text())
    train, val, test = set(split["train"]), set(split["val"]), set(split["test"])
    assert not (train & val) and not (train & test) and not (val & test)


def test_split_excludes_quadratics_like_d():
    subjects = [s for v in json.loads(SPLIT.read_text())["topics"].values()
                for s in v]
    assert "Factorising into a Double Bracket" not in subjects
    assert "Simplifying Algebraic Fractions" not in subjects


def test_metrics():
    from matheasy.shortlister.shortlist import map_at_k, recall_at_k
    ranked = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert map_at_k(ranked, [1, 5, 0], k=25) == pytest.approx((1 + 1 / 2 + 0) / 3)
    assert recall_at_k(ranked, [1, 6, 0], k=2) == pytest.approx(1 / 3)


@needs_eedi
def test_shortlist_returns_k_ranked_eedi_misconceptions():
    from matheasy.shortlister import shortlist
    out = shortlist("Topic: fractions - add fractions | Student wrote: 2/5 | adds numerators and denominators", k=25)
    assert len(out) == 25
    ids, names, scores = zip(*out)
    assert all(isinstance(i, int) for i in ids)
    assert list(scores) == sorted(scores, reverse=True)


@needs_openstax
def test_retrieve_finds_the_fractions_section():
    from matheasy.retrieval import retrieve
    out = retrieve("Adds the numerators and adds the denominators when adding fractions", k=3)
    assert len(out) == 3
    assert {"ref", "text", "book", "section", "score"} <= set(out[0])
    assert any("Fractions" in p["section_title"] for p in out)
    assert out[0]["ref"].startswith(("Prealgebra 2e, ", "Elementary Algebra 2e, "))


def test_hard_negatives_are_siblings_first_and_never_the_answer():
    import pandas as pd
    from matheasy.shortlister.hard_negatives import mine
    names = pd.DataFrame({"MisconceptionId": [1, 2, 3, 4],
                          "MisconceptionName": ["Adds the numerators and adds the denominators",
                                                "Adds the numerators and the denominators",  # near-duplicate of 1
                                                "Multiplies the numerators and the denominators",
                                                "Thinks a longer decimal is bigger"]})
    pairs = pd.DataFrame({"QuestionId": [7, 7], "WrongAnswer": ["A", "B"], "MisconceptionId": [1, 4]})
    negs = mine(pairs, names).set_index("WrongAnswer").NegativeIds.str.split()
    assert negs["A"][0] == "4"                      # sibling option of the same question comes first
    assert "1" not in negs["A"] and "2" not in negs["A"]   # never the answer or a rewording of it
    assert "3" in negs["A"]


def test_hard_negatives_skip_more_general_names():
    import pandas as pd
    from matheasy.shortlister.hard_negatives import mine
    names = pd.DataFrame({"MisconceptionId": [1, 2, 3],
                          "MisconceptionName": ["Ignores negative sign when expanding bracket",
                                                "Ignores the negative sign",
                                                "Only multiplies the first term in the expansion of a bracket"]})
    pairs = pd.DataFrame({"QuestionId": [7], "WrongAnswer": ["A"], "MisconceptionId": [1]})
    assert mine(pairs, names).NegativeIds[0].split() == ["3"]


def test_uk_maths_words_get_us_words_added():
    from matheasy.retrieval.retrieve import americanise
    q = americanise("Only multiplies the first term when expanding the bracket")
    assert q.startswith("Only multiplies") and "parentheses" in q and "distribute" in q
    assert americanise("adds fractions") == "adds fractions"


@needs_openstax
def test_hybrid_retrieve_finds_distributive_property_for_brackets():
    from matheasy.retrieval import retrieve
    out = retrieve("Only multiplies the first term inside the bracket, 3(x + 2) = 3x + 2 expressions", k=3)
    assert any(p["section_title"] in ("Distributive Property", "Properties of Real Numbers", "Multiply Polynomials")
               for p in out)


@needs_openstax
def test_hybrid_beats_or_matches_bm25_on_right_section_set():
    from matheasy.retrieval.retrieve import right_section_rate
    assert right_section_rate(3, "hybrid")[0] >= right_section_rate(3, "bm25")[0]


def test_is_weak_flags_only_an_unrelated_best_passage():
    from matheasy.retrieval.retrieve import is_weak

    assert is_weak([{"similarity": 0.2}, {"similarity": 0.3}])
    assert not is_weak([{"similarity": 0.2}, {"similarity": 0.8}])
    assert not is_weak([{"similarity": None}])  # BM25 mode: unknown, not weak
    assert not is_weak([])
