"""T3: turning off only the diagnoser's LLM, and topics for a step nothing explains."""

import pytest

from matheasy.diagnosis import diagnose
from matheasy.tutor import runner
from matheasy.ui.__main__ import Tutor
from matheasy.ui.db import Database


def test_without_llm_drops_the_session_and_the_eedi_ranker():
    seen = {}

    def fake_diagnose(before, after, **kwargs):
        seen.update(kwargs)
        return "diagnosis"

    off = runner.without_llm(fake_diagnose)
    assert off("1/2 + 1/3", "2/5", session=object(), eedi_ranker=object(), question="q") == "diagnosis"
    assert seen == {"session": None, "question": "q"}


def test_the_runner_wraps_diagnosis_only_when_asked(tmp_path):
    from matheasy.tutor.turnlog import TurnLogger
    from matheasy.tutor.store import InMemoryStore
    log = TurnLogger("t", "hash", tmp_path)
    on = runner.make_tutor("full", log, InMemoryStore(), llm=False, retrieve=None)
    off = runner.make_tutor("full", log, InMemoryStore(), llm=False, retrieve=None, llm_diagnosis=False)
    assert on.diagnose_fn is diagnose
    assert off.diagnose_fn.__name__ == "diagnose_without_llm"


def test_eedi_needs_the_diagnosers_llm():
    with pytest.raises(SystemExit):
        runner.main(["t", "--llm", "--eedi", "--no-llm-diagnosis"])


def test_the_page_reports_the_switch(tmp_path):
    tutor = Tutor(llm=False, eedi=False, db=Database(tmp_path / "t.db"), llm_diagnosis=False)
    assert tutor.status()["llm_diagnosis"] is False


@pytest.mark.parametrize("before, after, topics", [
    ("2(x + 3) - (x + 1)", "2x + 3 - x + 1", ("algebraic_notation", "expanding_brackets")),  # two mistakes, no rule
    ("5 × 6", "26", ("whole_number_arithmetic",)),               # a number fact: solve back's piece
    ("17 × 8 + 6", "140", ("whole_number_arithmetic",)),     # beyond the tables: 7 × 8 + 6 -> 60 is a fact slip now
    ("x - 15 = 27", "x = 32", ("algebraic_notation", "solving_equations")),
])
def test_a_step_nothing_explains_still_names_its_topics(before, after, topics):
    d = diagnose(before, after)
    assert d.status == "unknown" and d.misconception is None
    assert d.topics == topics
    assert d.trace[-1]["stage"] == "topics"


def test_a_diagnosed_step_keeps_the_rules_topic():
    d = diagnose("1/2 + 1/3", "2/5")
    assert d.topics == ("fractions",) and all(t["stage"] != "topics" for t in d.trace)
