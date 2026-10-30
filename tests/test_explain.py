import json

import pytest
import yaml

from matheasy.diagnosis import diagnose, explain, locate, rule_names
from matheasy.diagnosis.explain import TOPICS, rule_topic
from matheasy.diagnosis.solve_back import solve_back
from matheasy.llm import FakeBackend, LLMClient, load_config
from matheasy.llm.config import DEFAULT_CONFIG_PATH

SIGN_KEPT = [("2x = 11 + 3", True), ("2x = 14", False), ("x = 7", False)]  # 2x + 3 = 11 -> x = 7


@pytest.fixture
def model(tmp_path):
    """A scripted model: queue its replies with say(...); returns (session, backend, say)."""
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["provider"] = "fake"
    raw["cache"]["dir"] = str(tmp_path / "cache")
    raw["logging"]["dir"] = str(tmp_path / "logs")
    path = tmp_path / "llm.yaml"
    path.write_text(yaml.safe_dump(raw))
    queue = []
    backend = FakeBackend(lambda request: queue.pop(0))
    client = LLMClient(load_config(path), backend=backend, run_id="test")

    def say(steps, kind="misconception", mistake="kept the sign", explanation="...", topics=("solving_equations",)):
        """steps: [(line, the model calls it a mistake)]."""
        queue.append(json.dumps({"steps": [{"line": l, "mistake": m} for l, m in steps], "mistake": mistake,
                                 "explanation": explanation, "topics": list(topics), "kind": kind}))

    return client.session("student-1", "full"), backend, say


# ---------------------------------------------------------------- topics

def test_every_rule_has_a_topic_in_the_list():
    missing = {rule_topic(r) for r in rule_names()} - set(TOPICS)
    assert not missing, f"rule groups with no topic in TOPICS: {missing}"


def test_every_topic_named_is_kept_even_when_unverified(model):
    session, _, say = model
    say([("2x = 14", True), ("x = 6", True)],
        topics=["solving_equations", "negative_numbers", "solving_equations", "astrology"])
    e = explain(locate("2x + 3 = 11", "x = 7"), session=session)
    assert e.status == "unverified"
    assert set(e.topics) == {"solving_equations", "negative_numbers"}  # once each; unknown labels dropped


def test_no_known_topic_is_other(model):
    session, _, say = model
    say(SIGN_KEPT, topics=[])
    assert explain(locate("2x + 3 = 11", "x = 7"), session=session).topics == ("other",)


def test_a_rule_on_a_verified_wrong_step_adds_its_topic(model):
    # The model names only the question's topic; the rule on 5/6 + 1/6 -> 6/12 is certain.
    session, _, say = model
    say([("5/6 + 1/6", False), ("6/12", True)], topics=[])
    e = explain(locate("1/2 + 1/3 + 1/6", "6/12"), session=session)
    assert e.status == "verified" and e.topics == ("fractions",)


# ---------------------------------------------------------------- solve_back

def test_solve_back_finds_each_operation_that_alone_explains_the_number():
    pieces = solve_back("7 × 8 + 6", "60")
    assert {(p.source, p.wrong, p.expected) for p in pieces} == {("7 × 8", "54", "56"), ("56 + 6", "60", "62")}


def test_solve_back_keeps_the_students_own_line_for_the_last_operation():
    assert ("5/6 + 1/6", "6/12") in {(p.source, p.wrong) for p in solve_back("1/2 + 1/3 + 1/6", "6/12")}


@pytest.mark.parametrize("before, after", [("2x + 3 = 11", "x = 7"), ("3^2 × 3^4 - 10", "9^6 - 10")])
def test_solve_back_only_reads_a_line_of_numbers_turned_into_one_number(before, after):
    assert solve_back(before, after) == []


def test_a_rule_on_the_wrong_piece_diagnoses_without_the_model():
    d = diagnose("1/2 + 1/3 + 1/6", "6/12")
    assert d.status == "clear" and d.misconception == "rule:fractions.add_numerators_denominators"
    assert (d.located.source, d.located.wrong) == ("5/6 + 1/6", "6/12")  # the hinter sees the piece


# ---------------------------------------------------------------- explain

def test_the_wrong_step_is_found_by_code(model):
    session, _, say = model
    say(SIGN_KEPT)
    e = explain(locate("2x + 3 = 11", "x = 7"), session=session)
    assert e.status == "verified" and e.mistake == "kept the sign"
    assert [(w.before, w.after, w.rule) for w in e.wrong_steps] == [("2x + 3 = 11", "2x = 11 + 3", None)]


def test_an_unexplained_jump_onto_the_students_line_is_unverified(model):
    # Solved it right, then "x = 5 -> x = 5/4" with no mistake owned up to.
    session, _, say = model
    say([("2x = 10", False), ("x = 5", False), ("x = 5/4", False)])
    e = explain(locate("5x - 2 = 3x + 8", "x = 5/4"), session=session)
    assert e.status == "unverified" and "'x = 5' -> 'x = 5/4' is wrong" in e.reason


def test_a_chain_that_does_not_reach_the_students_line_is_unverified(model):
    session, _, say = model
    say([("2x = 14", True), ("x = 6", True)])
    e = explain(locate("2x + 3 = 11", "x = 7"), session=session)
    assert e.status == "unverified" and "not the student's line" in e.reason


def test_a_step_that_does_not_parse_is_unverified(model):
    session, _, say = model
    say([("2x = eleven plus 3", True), ("x = 7", False)])
    assert explain(locate("2x + 3 = 11", "x = 7"), session=session).status == "unverified"


def test_the_model_may_say_it_cannot_tell(model):
    session, _, say = model
    say([], kind="unclear", mistake="", explanation="")
    assert explain(locate("2x + 3 = 11", "x = 7"), session=session).status == "unclear"


def test_the_model_is_given_the_pieces(model):
    session, backend, say = model
    say([("54 + 6", True), ("60", False)], kind="slip")
    d = diagnose("7 × 8 + 6", "60", session=session)
    assert "7 × 8 gave 54 (should be 56)" in backend.requests[0].messages[-1].content
    assert d.explanation.status == "verified"
    assert [(w.before, w.after) for w in d.explanation.wrong_steps] == [("7 × 8 + 6", "54 + 6")]


def test_diagnose_calls_the_model_only_for_unknown_steps(model):
    session, backend, say = model
    say(SIGN_KEPT)
    d = diagnose("2x + 3 = 11", "x = 7", question="Solve 2x + 3 = 11", session=session)
    assert d.status == "unknown" and d.explanation.status == "verified"
    assert d.to_log()["explanation"]["steps"] == ["2x = 11 + 3", "2x = 14", "x = 7"]
    assert diagnose("1/2 + 1/3", "2/5", session=session).explanation is None  # a rule reproduces it: no call
    assert len(backend.requests) == 1
    assert diagnose("2x + 3 = 11", "x = 7").explanation is None  # no session: no call
