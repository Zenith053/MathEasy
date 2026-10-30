import dataclasses
import json
import re

import pytest

from matheasy.diagnosis import Candidate, diagnose
from matheasy.evaluation import SimStudent, check_files
from matheasy.llm import FakeBackend, LLMClient, LLMOutputError, load_config
from matheasy.tutor import HintRequest, Orchestrator, TurnLogger, TutorState, make_problem, run_session
from matheasy.tutor import describe, hinter as hinter_module
from matheasy.tutor.critic import TutorCritic, leaks_answer
from matheasy.tutor.hinter import LLMHinter
from matheasy.tutor.placeholders import TemplateHinter
from matheasy.tutor.store import InMemoryStore

RULE = "fractions.add_numerators_denominators"
PROBLEM = make_problem("p1", "1/2 + 1/3")
PASSAGES = [
    {"ref": "Prealgebra 2e, 4.5 Add and Subtract Fractions with Different Denominators", "text": "To add fractions..."},
    {"ref": "Prealgebra 2e, 4.4 Add and Subtract Fractions with Common Denominators", "text": "Fractions with..."},
]
REQUEST = HintRequest("fractions", "1/2 + 1/3", "2/5", "Adds the numerators and adds the denominators.",
                      "1/2 + 1/3 became 2/5")


def fake_retrieve(query, k=3):
    return PASSAGES[:k]


def keyword_judge(message, misconception):
    """Counts a message as targeted if it names the method: 'added the numerators and added the denominators'."""
    return "numerators and added the denominators" in message or \
        re.split(r"[:,]", misconception)[0].lower() in message.lower()


def scripted(hint=None, citation=PASSAGES[0]["ref"], critic=(True, [])):
    """A fake model that answers as the hinter or the critic, depending on the system prompt."""
    verdicts = iter(critic) if isinstance(critic, list) else None

    def reply(request):
        if request.system == hinter_module.SYSTEM:
            text = hint or ("You added the numerators and added the denominators. "
                            "What must the denominators be before you add?")
            if "rejected your previous hint" in request.messages[-1].content:
                text = "Think about what the denominator means. " + text  # a rewrite says something new
            return json.dumps({"hint": text, "citation": citation})
        approved, problems = next(verdicts) if verdicts else critic
        return json.dumps({"approved": approved, "problems": problems})
    return reply


@pytest.fixture
def llm(tmp_path):
    def make(reply, session_tokens=20000):
        cfg = dataclasses.replace(load_config(), log_dir=tmp_path / "llm", cache_dir=tmp_path / "cache",
                                  session_tokens=session_tokens)
        return LLMClient(cfg, backend=FakeBackend(reply), run_id="t")
    return make


def llm_records(client):
    return [json.loads(line) for line in client.log_path.read_text().splitlines()]


# --- misconception wording ----------------------------------------------------------

def test_misconception_text_for_each_kind_of_id(tmp_path, monkeypatch):
    assert describe.misconception_text(f"rule:{RULE}").startswith("Adds the numerators")
    assert describe.misconception_text("slip") is None
    assert describe.misconception_text("nonsense") is None

    names = tmp_path / "misconception_mapping.csv"
    names.write_text("MisconceptionId,MisconceptionName\n1287,Adds denominators when adding fractions \n")
    monkeypatch.setattr(describe, "EEDI_NAMES", names)
    describe._eedi_names.cache_clear()
    try:
        assert describe.misconception_text("eedi:1287") == "Adds denominators when adding fractions"
        assert describe.misconception_text("eedi:9") is None
    finally:
        describe._eedi_names.cache_clear()


def test_eedi_diagnosis_no_longer_crashes_the_tutor(tmp_path):
    eedi_only = lambda error, k=25: [Candidate("eedi:999999", 9.0), Candidate("eedi:42", 2.0)]
    hinter = TemplateHinter()
    tutor = Orchestrator(hinter, TutorCritic(), InMemoryStore(), TurnLogger("t", "h", tmp_path),
                         diagnose_fn=lambda *a, **k: diagnose(*a, shortlister=eedi_only, **k))
    state = TutorState("s", "full", "st", tutor.store.load("st"))
    tutor.start(state, PROBLEM)
    reply = tutor.respond(state, "2/5")
    assert reply.kind == "hint"  # unknown wording: a general hint, not a KeyError
    assert "Look again at your answer" in reply.text


# --- hinter --------------------------------------------------------------------------

def test_hinter_prompt_has_misconception_and_sections_but_no_answer(llm):
    client = llm(scripted())
    hint = LLMHinter(fake_retrieve).hint(REQUEST, llm=client.session("s", "full"))

    assert hint.citations == (PASSAGES[0]["ref"],)
    [request] = client.backend.requests
    prompt = request.messages[-1].content
    assert "Adds the numerators and adds the denominators" in prompt and PASSAGES[1]["ref"] in prompt
    assert "5/6" not in prompt and "5/6" not in request.system
    assert request.json_schema["properties"]["citation"]["enum"] == [p["ref"] for p in PASSAGES]


def test_hinter_rewrite_includes_critic_reasons(llm):
    client = llm(scripted())
    LLMHinter(fake_retrieve).hint(REQUEST, ["It gives away the answer (5/6)."], llm=client.session("s", "full"))
    assert "It gives away the answer (5/6)." in client.backend.requests[0].messages[-1].content


def test_hinter_rejects_an_invented_citation(llm):
    client = llm(scripted(citation="Some other book, 9.9"))
    with pytest.raises(LLMOutputError, match="not offered"):
        LLMHinter(fake_retrieve).hint(REQUEST, llm=client.session("s", "full"))


def test_hinter_without_retriever_has_no_citation_field(llm):
    client = llm(scripted())
    hint = LLMHinter().hint(REQUEST, llm=client.session("s", "full"))
    assert hint.citations == ()
    assert "citation" not in client.backend.requests[0].json_schema["properties"]


# --- critic --------------------------------------------------------------------------

@pytest.mark.parametrize("text, answer, leaked", [
    ("So the answer is 5/6.", "5/6", "5/6"),
    ("That makes 10/12 in total.", "5/6", "10/12"),
    ("Half is 0.5 here.", "1/2", "0.5"),
    ("Write it as 50%.", "1/2", "50%"),
    ("It simplifies to 2x+6.", "2x + 6", "2x + 6"),
    ("Look at 1/2 + 1/3 again.", "5/6", None),
    ("So x must be 3.", "x = 3", "3"),  # an equation's value on its own gives it away
    ("Then x = 3.", "x = 3", "x = 3"),
    ("Take 4 from both sides of 2x + 4 = 10.", "x = 3", None),
])
def test_leak_check(text, answer, leaked):
    assert leaks_answer(text, answer) == leaked


def test_code_checks_block_before_any_llm_call(llm):
    client = llm(scripted())
    verdict = TutorCritic().review("The answer is 5/6.", "hint", PROBLEM, request=REQUEST,
                                   llm=client.session("s", "full"))
    assert not verdict.passed and "gives away the answer" in verdict.reasons[0]
    assert client.backend.requests == []


def test_questions_and_probes_get_no_llm_review(llm):
    client = llm(scripted(critic=(False, ["bad"])))
    session = client.session("s", "full")
    assert TutorCritic().review("Work out 1/2 + 1/3", "question", PROBLEM, llm=session).passed
    assert TutorCritic().review("Before we go on, what is 1/3 + 1/5?", "probe", PROBLEM, llm=session).passed
    assert client.backend.requests == []


def test_llm_review_sees_the_answer_and_returns_its_problems(llm):
    client = llm(scripted(critic=(False, ["The hint does the step for the student."])))
    verdict = TutorCritic().review("Find a common denominator first.", "hint", PROBLEM, request=REQUEST,
                                   llm=client.session("s", "full"))
    assert verdict.reasons == ("The hint does the step for the student.",)
    prompt = client.backend.requests[0].messages[-1].content
    assert "Correct answer: 5/6" in prompt and REQUEST.misconception in prompt


# --- the whole tutor on the LLM client ----------------------------------------------------

def make_tutor(tmp_path):
    return Orchestrator(LLMHinter(fake_retrieve), TutorCritic(), InMemoryStore(),
                        TurnLogger("t", load_config().hash, tmp_path / "turns"))


def test_full_session_with_llm_hinter_and_critic(tmp_path, llm):
    client = llm(scripted())
    tutor = make_tutor(tmp_path)
    result = run_session(tutor, SimStudent("s1", RULE, judge=keyword_judge), [PROBLEM],
                         session_id="sess-0", condition="full", session_index=0,
                         llm_session=client.session("sess-0", "full"))

    hint = next(u.text for u in result.state.transcript if u.kind == "hint")
    assert hint.endswith(f"(See {PASSAGES[0]['ref']}.)")
    roles = [r["role"] for r in llm_records(client)]
    assert roles == ["hinter", "critic"]
    end = [json.loads(l) for l in tutor.log.path.read_text().splitlines() if '"session_end"' in l][0]
    assert end["tokens_used"] > 0
    assert check_files([tutor.log.path, client.log_path]) == []


def test_critic_rejection_drives_a_rewrite(tmp_path, llm):
    client = llm(scripted(critic=[(False, ["Too vague."]), (True, [])]))
    tutor = make_tutor(tmp_path)
    state = TutorState("s", "full", "st", tutor.store.load("st"), llm=client.session("s", "full"))
    tutor.start(state, PROBLEM)
    assert tutor.respond(state, "2/5").kind == "hint"
    roles = [r["role"] for r in llm_records(client)]
    assert roles == ["hinter", "critic", "hinter", "critic"]
    assert "Too vague." in client.backend.requests[2].messages[-1].content


def test_tight_budget_ends_in_safe_template(tmp_path, llm):
    client = llm(scripted(), session_tokens=30)
    tutor = make_tutor(tmp_path)
    result = run_session(tutor, SimStudent("s1", RULE, judge=keyword_judge), [PROBLEM],
                         session_id="sess-0", condition="full", session_index=0,
                         llm_session=client.session("sess-0", "full"))
    assert result.ended_by == "budget"
    assert result.state.transcript[-1].kind == "safe_template"
    assert check_files([tutor.log.path]) == []
