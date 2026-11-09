"""Linking a step no rule explains to an Eedi misconception, and what the tutor does with it."""

import json

import pytest
import yaml

from matheasy.diagnosis import diagnose, locate
from matheasy.diagnosis.eedi import options
from matheasy.llm import FakeBackend, LLMClient, load_config
from matheasy.llm.config import DEFAULT_CONFIG_PATH
from matheasy.shortlister.candidates import candidate_ids
from matheasy.tutor.analysis import mistake
from matheasy.tutor.hinter import LLMHinter
from matheasy.tutor.state import HintRequest, Problem, StudentMemory
from matheasy.tutor.store import InMemoryStore

STEP = ("2x + 3 = 11", "x = 7")   # no rule reproduces it
IN = sorted(candidate_ids())[:3]   # three in-topic Eedi IDs
OUT = 0                            # not in our topics
assert OUT not in candidate_ids()


def ranker(query, k=25):
    """A fake shortlister: one out-of-topic name first, then three in-topic ones."""
    return [(OUT, "out of topic", 0.9)] + [(m, f"name {m}", 0.8 - i / 10) for i, m in enumerate(IN)]


@pytest.fixture
def model(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["provider"] = "fake"
    raw["cache"]["dir"] = str(tmp_path / "cache")
    raw["logging"]["dir"] = str(tmp_path / "logs")
    path = tmp_path / "llm.yaml"
    path.write_text(yaml.safe_dump(raw))
    queue = []
    backend = FakeBackend(lambda request: queue.pop(0))
    client = LLMClient(load_config(path), backend=backend, run_id="test")

    def say(eedi, kind="misconception", fit="exact"):
        queue.append(json.dumps({"steps": [{"line": "2x = 11 + 3", "mistake": True}, {"line": "2x = 14", "mistake": False},
                                           {"line": "x = 7", "mistake": False}],
                                 "mistake": "kept the sign", "explanation": "...", "topics": ["solving_equations"],
                                 "eedi": eedi, "eedi_fit": fit, "kind": kind}))

    return client.session("student-1", "full"), backend, say


def test_options_keep_only_in_topic_names_best_first():
    assert options(locate(*STEP), ranker) == [(f"eedi:{m}", f"name {m}") for m in IN]
    assert len(options(locate(*STEP), ranker, k=2)) == 2


def test_a_picked_name_becomes_the_diagnosis(model):
    session, backend, say = model
    say(2)
    d = diagnose(*STEP, session=session, eedi_ranker=ranker)
    assert d.status == "clear" and d.misconception == f"eedi:{IN[1]}" and d.beliefs == {f"eedi:{IN[1]}": 1.0}
    assert d.topics == ("solving_equations",)
    prompt = backend.requests[0].messages[-1].content
    assert "Possible misconceptions:" in prompt and f"2. name {IN[1]}" in prompt and "out of topic" not in prompt


def test_a_close_pick_is_still_the_diagnosis_and_says_so(model):
    # A near concept helps the student more than no answer; the fit tells the tutor how sure it is.
    session, _, say = model
    say(1, fit="close")
    d = diagnose(*STEP, session=session, eedi_ranker=ranker)
    assert d.misconception == f"eedi:{IN[0]}" and d.explanation.eedi_fit == "close"


@pytest.mark.parametrize("pick", [0, 99, -1, True, "2"])
def test_a_bad_pick_leaves_the_step_unknown(model, pick):
    session, _, say = model
    say(pick)
    d = diagnose(*STEP, session=session, eedi_ranker=ranker)
    assert d.status == "unknown" and d.misconception is None and d.explanation.eedi is None
    assert d.explanation.eedi_fit is None


def test_a_slip_is_not_given_an_eedi_misconception(model):
    session, _, say = model
    say(1, kind="slip")
    assert diagnose(*STEP, session=session, eedi_ranker=ranker).status == "unknown"


def test_without_a_ranker_no_names_are_offered(model):
    session, backend, say = model
    say(0)
    diagnose(*STEP, session=session)
    assert "Possible misconceptions" not in backend.requests[0].messages[-1].content


# ---------------------------------------------------------------- what the tutor does with it

def test_the_store_records_the_eedi_id_for_later():
    store, memory = InMemoryStore(), StudentMemory("student-1")
    store.record(memory, f"eedi:{IN[0]}")
    store.record(memory, f"eedi:{IN[0]}")
    assert memory.times_diagnosed[f"eedi:{IN[0]}"] == 2  # a repeat: a habit, not a one-off


def test_the_textbook_is_searched_and_the_hint_aimed_with_the_eedi_name():
    asked = []
    hinter = LLMHinter(lambda query, k=3: asked.append(query) or [])
    hinter.passages(HintRequest("fractions", "1/2 + 1/3", "2/5", "Adds numerators and adds denominators", None))
    assert asked == ["Adds numerators and adds denominators fractions"]


def test_the_analysis_shows_the_eedi_name(model):
    session, _, say = model
    say(1)
    d = diagnose(*STEP, session=session, eedi_ranker=ranker)
    names = pytest.importorskip("matheasy.tutor.describe")._eedi_names()
    if not names:
        pytest.skip("no Eedi data on this machine")
    assert mistake(d, Problem("p1", "expressions", "2x + 3 = 11", "x = 4")) == names[IN[0]]
