import dataclasses
import json

import pytest

from matheasy.diagnosis import Diagnosis, Explanation, Transcription, WrongStep, locate
from matheasy.llm import FakeBackend, LLMCallError, LLMClient, load_config
from matheasy.tutor import B0Tutor, Orchestrator, TurnLogger, TutorState, make_problem
from matheasy.tutor import orchestrator
from matheasy.tutor.critic import TutorCritic
from matheasy.tutor.analysis import mistake
from matheasy.tutor.orchestrator import RETYPE, SAFE_TEMPLATE, read_answer
from matheasy.tutor.store import InMemoryStore

PROBLEM = make_problem("p1", "1/2 + 1/3")
SESSION = object()  # stands in for the tutor's LLM session; transcribe is faked below


class CodeChecksOnly(TutorCritic):
    def review(self, text, kind, problem, *, llm=None, **kwargs):
        return super().review(text, kind, problem, **kwargs)


class RecordingHinter:
    def __init__(self):
        self.requests = []

    def hint(self, request, feedback=None, **_):
        from matheasy.tutor import Hint
        self.requests.append(request)
        return Hint("Check how you combined the denominators.")


def tutor_and_state(tmp_path, diagnose_fn=None):
    hinter = RecordingHinter()
    kwargs = {"diagnose_fn": diagnose_fn} if diagnose_fn else {}
    tutor = Orchestrator(hinter, CodeChecksOnly(), InMemoryStore(), TurnLogger("t", "h", tmp_path), **kwargs)
    state = TutorState("s", "full", "st", tutor.store.load("st"), llm=SESSION)
    tutor.start(state, PROBLEM)
    return tutor, state, hinter


def fake_transcribe(result):
    def transcribe(text, *, question, session):
        assert session is SESSION
        if isinstance(result, Exception):
            raise result
        return dataclasses.replace(result, original=text)
    return transcribe


UNSURE = Transcription("unsure", None, "", used_llm=True, reason="ambiguous", choices=("2/5", "5/2"))


# --- the Read stage --------------------------------------------------------------------

def test_offline_read_is_the_literal_reading():
    assert read_answer("I got 2/5.", "1/2 + 1/3").line == "2/5"
    assert read_answer("no idea", "1/2 + 1/3").status == "unreadable"


def test_standard_notation_needs_no_model_call(tmp_path):
    cfg = dataclasses.replace(load_config(), log_dir=tmp_path / "llm", cache_dir=tmp_path / "cache")
    client = LLMClient(cfg, backend=FakeBackend(), run_id="t")
    reading = read_answer("I got x = 4.", "x + 3 = 7", client.session("s", "full"))
    assert (reading.status, reading.line, reading.used_llm) == ("ok", "x = 4", False)
    assert client.backend.requests == []


def test_an_unsure_reading_is_asked_about_once(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "transcribe", fake_transcribe(UNSURE))
    tutor, state, hinter = tutor_and_state(tmp_path)
    ask = tutor.respond(state, "2 / 5 maybe")
    assert ask.kind == "question" and ask.text == "Did you mean 1) 2/5 or 2) 5/2? Reply with the number."
    assert tutor.respond(state, "1").kind == "hint"  # resolve() read "1" as 2/5, a wrong answer
    assert hinter.requests[0].student_answer == "2/5"

    tutor.respond(state, "?")
    assert tutor.respond(state, "no").text == RETYPE  # still unclear after asking once: no second question


@pytest.mark.parametrize("status", ["not_maths", "unfaithful", "unreadable"])
def test_unusable_readings_ask_to_retype(tmp_path, monkeypatch, status):
    monkeypatch.setattr(orchestrator, "transcribe", fake_transcribe(Transcription(status, None, "", used_llm=True)))
    tutor, state, _ = tutor_and_state(tmp_path)
    assert tutor.respond(state, "help me").text == RETYPE


def test_an_llm_error_while_reading_sends_the_safe_template(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "transcribe", fake_transcribe(LLMCallError("down")))
    tutor, state, _ = tutor_and_state(tmp_path)
    assert tutor.respond(state, "help me").text == SAFE_TEMPLATE
    assert state.transcript[-2].text == "help me"


def test_a_transcribed_reading_is_logged_with_the_diagnosis(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "transcribe", fake_transcribe(
        Transcription("ok", "2/5", "", used_llm=True)))
    tutor, state, _ = tutor_and_state(tmp_path)
    tutor.respond(state, "two fifths")
    [d] = [json.loads(line) for line in tutor.log.path.read_text().splitlines() if '"diagnosis"' in line]
    assert d["read"]["line"] == "2/5"


def test_b0_sees_how_its_input_was_read(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "transcribe", lambda text, **_: dataclasses.replace(UNSURE, original=text))
    replies = []

    def reply(request):
        replies.append(request.messages[-1].content)
        return json.dumps({"message": "Which did you mean?", "student_is_correct": False,
                           "student_misconception": ""})
    cfg = dataclasses.replace(load_config(), log_dir=tmp_path / "llm", cache_dir=tmp_path / "cache")
    client = LLMClient(cfg, backend=FakeBackend(reply), run_id="t")
    b0 = B0Tutor(TurnLogger("t", "h", tmp_path))
    state = TutorState("s", "b0", "st", b0.store.load("st"), llm=client.session("s", "b0"))
    b0.start(state, PROBLEM)
    b0.respond(state, "2 / 5 maybe")
    assert "Student: 2 / 5 maybe (could mean: 2/5 or 5/2)" in replies[0]


# --- B's diagnose() arguments and LLM fallback ---------------------------------------------

def unknown_with(explanation):
    def diagnose_fn(before, after, **kwargs):
        diagnose_fn.kwargs = kwargs
        return Diagnosis("unknown", None, {}, None, locate(before, after), "full", (), explanation)
    return diagnose_fn


def explanation(status="verified", kind="misconception", mistake="added tops and bottoms",
                why="thinks fractions add like whole numbers", rule=None):
    return Explanation(status, mistake, why, kind, ("fractions",), ("2/5",),
                       (WrongStep("1/2 + 1/3", "2/5", rule),))


def test_diagnose_gets_the_question_and_the_session(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "transcribe", fake_transcribe(Transcription("ok", "2/5", "", False)))
    fn = unknown_with(None)
    tutor, state, hinter = tutor_and_state(tmp_path, fn)
    tutor.respond(state, "2/5")
    assert fn.kwargs["question"] == "1/2 + 1/3" and fn.kwargs["session"] is SESSION
    assert hinter.requests[0].misconception is None  # no explanation: a general hint


@pytest.mark.parametrize("e, words", [
    (explanation(), "added tops and bottoms: thinks fractions add like whole numbers"),
    (explanation(why="the answer should be 5/6"), "added tops and bottoms"),  # gives the answer away: dropped
    (explanation(mistake="got 5/6 wrong", why="it is 5/6"), None),
    (explanation(status="unverified"), None),  # the model's guess only
    (explanation(kind="slip"), None),
])
def test_the_hinter_gets_a_verified_explanation_without_the_answer(e, words):
    assert mistake(unknown_with(e)("1/2 + 1/3", "2/5"), PROBLEM) == words


def test_a_rule_that_reproduces_the_wrong_step_names_it():
    e = explanation(rule="rule:fractions.add_numerators_denominators")
    assert mistake(unknown_with(e)("1/2 + 1/3", "2/5"), PROBLEM).startswith("Adds the numerators")


def test_explained_words_reach_the_hinter(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "transcribe", fake_transcribe(Transcription("ok", "2/5", "", False)))
    tutor, state, hinter = tutor_and_state(tmp_path, unknown_with(explanation()))
    tutor.respond(state, "2/5")
    assert hinter.requests[0].misconception.startswith("added tops and bottoms")
