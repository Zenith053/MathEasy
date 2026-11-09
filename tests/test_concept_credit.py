"""What a student gets credit for: every concept a correct answer uses, probe answers too, and
an Eedi misconception's suspicion fading as the student answers its topic correctly."""

import json

import pytest
import yaml

from matheasy.diagnosis import diagnose, load_bank
from matheasy.diagnosis.explain import TOPICS
from matheasy.diagnosis.topics import question_topics
from matheasy.llm import FakeBackend, LLMClient, load_config
from matheasy.llm.config import DEFAULT_CONFIG_PATH
from matheasy.tutor import Orchestrator, TurnLogger, TutorState, make_problem
from matheasy.tutor import planner
from matheasy.tutor.critic import TutorCritic
from matheasy.tutor.placeholders import TemplateHinter
from matheasy.tutor.state import StudentMemory
from matheasy.tutor.store import InMemoryStore


@pytest.mark.parametrize("question, topics", [
    ("1/2 + 1/3", {"fractions"}),
    ("12 ÷ 3", set()),                     # a division, not a fraction
    ("0.3 × 0.2", {"decimals"}),
    ("2 + 3 × 4", {"order_of_operations"}),
    ("4x + 2x - 3 + 5", {"algebraic_notation", "collecting_like_terms"}),
    ("2(x + 3) = 14", {"algebraic_notation", "expanding_brackets", "solving_equations"}),
    ("5 - (x + 1)", {"algebraic_notation", "expanding_brackets"}),
    ("(3 × 10^4) × (2 × 10^5)", {"scientific_notation"}),
    ("2/3 = 6/x", {"algebraic_notation", "fractions", "ratios_proportions"}),
])
def test_question_topics_reads_the_concepts_off_the_question(question, topics):
    assert set(question_topics(question)) == topics
    assert set(question_topics(question)) <= set(TOPICS)


def test_a_right_answer_credits_a_concept_no_rule_covers():
    memory = StudentMemory("s1")
    planner.answered(memory, make_problem("p1", "4x + 2x - 3 + 5"), "6x + 2", True)
    assert memory.p_knows["collecting_like_terms"] > planner.KNOWS_AT_START


def test_a_wrong_answer_is_not_charged_to_every_visible_concept():
    memory = StudentMemory("s1")
    planner.answered(memory, make_problem("p1", "4x + 2x - 3 + 5"), "6x^2 + 2", False, topics=["collecting_like_terms"])
    assert "algebraic_notation" not in memory.p_knows  # only the diagnosed topic (and the rules' topics) lose


def test_eedi_suspicion_fades_with_right_answers_in_its_topic_only():
    memory = StudentMemory("s1")
    planner.diagnosed(memory, "eedi:1088", topics=["fractions"])
    start = memory.p_holds["eedi:1088"]
    planner.answered(memory, make_problem("p1", "0.3 × 0.2"), "0.06", True)    # another topic: no change
    assert memory.p_holds["eedi:1088"] == start
    for i, q in enumerate(["1/2 + 1/3", "2/5 + 1/4", "3/4 - 1/3", "1/6 + 1/3", "2/7 + 1/2"]):
        planner.answered(memory, make_problem(f"p{i}", q), "x", True)
    assert memory.p_holds["eedi:1088"] < planner.FIXED_BELOW < start  # several right answers: looks fixed


def test_an_eedi_pick_that_belongs_to_one_rule_is_recorded_as_that_rule(tmp_path):
    rule = load_bank().for_eedi(1804)
    assert rule == ["rule:fractions.multiply_across_for_division"]
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw.update(provider="fake")
    raw["cache"]["dir"], raw["logging"]["dir"] = str(tmp_path / "c"), str(tmp_path / "l")
    (tmp_path / "llm.yaml").write_text(yaml.safe_dump(raw))
    reply = {"steps": [{"line": "x = 7", "mistake": True}], "mistake": "m", "explanation": "e",
             "topics": ["fractions"], "eedi": 1, "eedi_fit": "exact", "kind": "misconception"}
    client = LLMClient(load_config(tmp_path / "llm.yaml"), backend=FakeBackend(lambda r: json.dumps(reply)), run_id="t")
    d = diagnose("2x + 3 = 11", "x = 7", session=client.session("s", "full"),
                 eedi_ranker=lambda q, k=25: [(1804, "When dividing fractions, multiplies across", 0.9)])
    assert d.misconception == rule[0] and d.explanation.eedi == "eedi:1804"


def test_probe_answers_count_towards_the_probes_concepts(tmp_path):
    tutor = Orchestrator(TemplateHinter(), TutorCritic(), InMemoryStore(),
                         TurnLogger("test", config_hash="abc", log_dir=tmp_path))
    state = TutorState("sess", "full", "s1", tutor.store.load("s1"))
    tutor.start(state, make_problem("p1", "2/3 ÷ 4/5"))
    reply = tutor.respond(state, "8/15")          # two rules tie: the tutor asks a probe
    assert reply.kind == "probe"
    probe = state.probe.question
    tutor.respond(state, "35" if probe == "7 ÷ 0.2" else "1")
    assert probe in state.memory.worked           # the probe answer reached the student model


@pytest.mark.parametrize("before, after, slip", [
    ("-2(x + y - z - k)", "-2x - 2y + 2z - 2k", True),    # -z right, -k wrong: a slip
    ("-2(x + y - z - k)", "-2x - 2y - 2z - 2k", False),   # both minus terms wrong alike: a misconception
    ("3(x + y + z + 4)", "3x + 3y + 3z + 4", False),      # the number has no term like it
    ("-(a - b + c - d)", "-a + b - c - d", True),
    ("3k - 2(x + y - z - k)", "3k - 2x - 2y + 2z - 2k", False),   # k outside too: cannot pin it on one term
    ("-2(x + y - z - k) = 4", "-2x - 2y + 2z - 2k = 4", True),
])
def test_an_operation_done_right_on_all_terms_but_one_is_a_slip(before, after, slip):
    from matheasy.diagnosis import SLIP
    from matheasy.diagnosis.slips import bracket_slip
    assert bool(bracket_slip(before, after)) == slip
    d = diagnose(before, after)
    assert (d.misconception == SLIP) == slip
    if slip:
        assert d.topics == ("expanding_brackets",) and any(t["stage"] == "slip" for t in d.trace)
