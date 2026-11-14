"""T6 in docs/design_decisions.md: every hint rejected because the hinter was told to aim at a
misconception the student's work rules out, and the critic could not see what the student got right."""

import dataclasses
import json

import pytest

from matheasy.diagnosis import Diagnosis, Explanation, WrongStep, locate
from matheasy.llm import FakeBackend, LLMClient, load_config
from matheasy.tutor import Orchestrator, TurnLogger, TutorState, describe, hinter as hinter_module, make_problem
from matheasy.tutor import critic as critic_module
from matheasy.tutor.critic import TutorCritic
from matheasy.tutor.hinter import LLMHinter
from matheasy.tutor.hinting import certainty, hint_request, right_steps, write_hint
from matheasy.tutor.store import InMemoryStore

QUESTION = "1/2 + 1/3 + 1/6 + 2^2"
PROBLEM = make_problem("p", QUESTION)  # answer 5; the student wrote 1, dropping the 2^2
WRONG = "Remember that any number divided by itself is 1. Which fractions here are a number over itself?"
GOOD = ("You correctly found that the fractions add up to 1, but you still need to include the "
        "exponent. Remember that 2^2 means 2 multiplied by itself. What do you get when you add that?")


@pytest.fixture(autouse=True)
def eedi_names(monkeypatch):
    monkeypatch.setattr(describe, "_eedi_names", lambda: {413: "Believes that n/n = n, instead of 1"})


def explanation(status="unverified", fit="close", steps=("1/2 + 1/3 + 1/6 + 4", "1 + 4", "1"), wrong=("1",)):
    return Explanation(status, "dropped a term", "left out 2^2", "misconception", ("fractions",), steps,
                       tuple(WrongStep("x", w, None) for w in wrong), eedi="eedi:413", eedi_fit=fit)


def eedi_diagnosis(e):
    """What diagnose() returned in the T6 case: the explainer's Eedi pick as the diagnosis."""
    return Diagnosis("clear", "eedi:413", {"eedi:413": 1.0}, None, locate(QUESTION, "1"), "full",
                     ({"stage": "decide", "status": "clear", "misconception": "eedi:413", "eedi": "eedi:413"},), e,
                     ("fractions",))


def scripted(critic_replies):
    """A fake model: the hinter writes WRONG while told the misconception, GOOD without it."""
    verdicts = iter(critic_replies)

    def reply(request):
        if request.system == hinter_module.SYSTEM:
            told = "Misconception: Believes" in request.messages[-1].content
            return json.dumps({"hint": WRONG if told else GOOD})
        return json.dumps(next(verdicts))
    return reply


@pytest.fixture
def llm(tmp_path):
    cfg = dataclasses.replace(load_config(), log_dir=tmp_path / "llm", cache_dir=tmp_path / "cache")
    return lambda reply: LLMClient(cfg, backend=FakeBackend(reply), run_id="t")


# --- how sure the diagnosis is ---------------------------------------------------------------

def test_certainty_of_each_kind_of_diagnosis():
    from matheasy.diagnosis import diagnose
    assert certainty(diagnose("1/2 + 1/3", "2/5")) == "certain"  # a rule reproduces it
    assert certainty(eedi_diagnosis(explanation())) == "possible"  # the unverified pick in T6
    assert certainty(eedi_diagnosis(explanation("verified", "close"))) == "possible"
    assert certainty(eedi_diagnosis(explanation("verified", "exact"))) == "likely"
    assert certainty(None) is None


def test_right_steps_come_only_from_a_verified_working():
    d = eedi_diagnosis(explanation("verified"))
    assert right_steps(d) == ("1/2 + 1/3 + 1/6 + 2^2 → 1/2 + 1/3 + 1/6 + 4", "1/2 + 1/3 + 1/6 + 4 → 1 + 4")
    # Unverified: the working is only the model's guess at what the student did (the student wrote just
    # "1"), so none of it is passed on as the student's own right work. Tried and reverted 2026-11-11:
    # passing code-checked steps of an unverified guess told the hinter the student had worked out
    # 2^2 = 4, and its hints then gave that away.
    assert right_steps(eedi_diagnosis(explanation())) == ()


def test_the_critic_may_let_a_hint_ask_the_student_to_work_something_out():
    assert "a question is not an answer" in critic_module.SYSTEM


# --- the T6 case ---------------------------------------------------------------------------------

def test_a_misconception_the_critic_rules_out_is_dropped_from_the_rewrite(llm):
    client = llm(scripted([
        {"approved": False, "problems": ["The student's 1 shows they know n/n = 1."], "misconception_fits": False},
        {"approved": True, "problems": [], "misconception_fits": True},
    ]))
    request = hint_request(PROBLEM, "1", eedi_diagnosis(explanation()), right=("1/2 + 1/3 + 1/6 → 1",))
    assert request.certainty == "possible" and request.misconception.startswith("Believes that n/n = n")

    written = write_hint(LLMHinter(), TutorCritic(), request, PROBLEM, llm=client.session("s", "full"))

    assert written.hint.text == GOOD and written.dropped
    assert [a["hint"] for a in written.attempts] == [WRONG]
    first_hint, first_review, second_hint, _ = [r.messages[-1].content for r in client.backend.requests]
    assert "How sure: only possible" in first_hint and "Already right: 1/2 + 1/3 + 1/6 → 1" in first_hint
    assert "What went wrong: " in first_review and "How sure: only possible" in first_review
    assert "Already right: 1/2 + 1/3 + 1/6 → 1" in first_review
    assert "Misconception: not known" in second_hint and "How sure" not in second_hint


def test_the_tutor_does_not_show_a_ruled_out_misconception(tmp_path, llm):
    client = llm(scripted([
        {"approved": False, "problems": ["It does not fit."], "misconception_fits": False},
        {"approved": True, "problems": [], "misconception_fits": True},
    ]))
    log = TurnLogger("t", "h", tmp_path)
    tutor = Orchestrator(LLMHinter(), TutorCritic(), InMemoryStore(), log,
                         diagnose_fn=lambda *a, **k: eedi_diagnosis(explanation()))
    state = TutorState("s", "full", "st", tutor.store.load("st"), llm=client.session("s", "full"))
    tutor.start(state, PROBLEM)
    reply = tutor.respond(state, "1")
    assert reply.kind == "hint" and GOOD in reply.text and "The mistake:" not in reply.text
    [hint] = [json.loads(line) for line in log.path.read_text().splitlines() if '"hint"' in line]
    assert (hint["certainty"], hint["dropped_misconception"], hint["rewrites"]) == ("possible", True, 1)


def test_the_critic_prompt_says_what_may_be_restated():
    assert "may restate what the student already worked out correctly" in critic_module.SYSTEM
    assert "2^2 means 2 × 2" in critic_module.SYSTEM
    assert critic_module.SCHEMA["required"] == ["approved", "problems", "misconception_fits"]


def test_a_certain_misconception_is_kept_even_if_a_rewrite_is_needed(llm):
    from matheasy.diagnosis import diagnose
    client = llm(lambda r: json.dumps({"hint": "Are the pieces the same size?"}) if r.system == hinter_module.SYSTEM
                 else json.dumps({"approved": False, "problems": ["Too vague."], "misconception_fits": True}))
    request = hint_request(make_problem("p", "1/2 + 1/3"), "2/5", diagnose("1/2 + 1/3", "2/5"))
    written = write_hint(LLMHinter(), TutorCritic(), request, make_problem("p", "1/2 + 1/3"),
                         llm=client.session("s", "full"))
    assert written.hint is None and not written.dropped and len(written.attempts) == 3
    assert all("How sure: certain" in r.messages[-1].content for r in client.backend.requests
               if r.system == hinter_module.SYSTEM)


def test_the_page_can_allow_fewer_rewrites(llm):
    client = llm(lambda r: json.dumps({"hint": "Are the pieces the same size?"}) if r.system == hinter_module.SYSTEM
                 else json.dumps({"approved": False, "problems": ["Too vague."], "misconception_fits": True}))
    request = hint_request(make_problem("p", "1/2 + 1/3"), "2/5", diagnose_fraction())
    written = write_hint(LLMHinter(), TutorCritic(), request, make_problem("p", "1/2 + 1/3"),
                         llm=client.session("s", "full"), max_rewrites=1)
    assert written.hint is None and len(written.attempts) == 2      # the default allows 3


def diagnose_fraction():
    from matheasy.diagnosis import diagnose
    return diagnose("1/2 + 1/3", "2/5")
