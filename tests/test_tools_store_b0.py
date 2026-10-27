import dataclasses
import json

import pytest

from matheasy.evaluation import SimStudent, check_files
from matheasy.llm import FakeBackend, LLMClient, load_config
from matheasy.tutor import (
    B0Tutor, InMemoryStore, JsonStudentStore, NoMemoryStore, SAFE_TEMPLATE, Tool, ToolBox, TurnLogger, TutorState,
    make_problem, run_session, standard_tools,
)
from matheasy.tutor.baselines import SYSTEM as B0_SYSTEM

RULE = "fractions.add_numerators_denominators"
PROBLEM = make_problem("p1", "1/2 + 1/3")


def judge(message, misconception):
    return "added the numerators and added the denominators" in message


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def log(tmp_path):
    return TurnLogger("t", load_config().hash, tmp_path / "turns")


@pytest.fixture
def state(log):
    return TutorState("s", "b1", "st", NoMemoryStore().load("st"))


# --- tool wrappers ---------------------------------------------------------------------

def test_valid_tool_call_runs_and_is_logged(log, state):
    tools = ToolBox(standard_tools(), log)
    result = tools.call(state, "check_step", {"before": "1/2 + 1/3", "after": "2/5"})
    assert (result.ok, result.value) == (True, False)
    assert tools.call(state, "run_rule", {"rule": RULE, "item": "1/2 + 1/3"}).value == "2/5"
    assert tools.call(state, "locate", {"before": "1/2 + 1/3", "after": "2/5"}).value["wrong"] == "2/5"
    [first, *_] = read(log.path)
    assert (first["event"], first["tool"], first["args_valid"], first["ok"], first["is_retry"]) == \
        ("tool_call", "check_step", True, True, False)


@pytest.mark.parametrize("name, args, problem", [
    ("check_step", {"before": "1/2"}, "missing argument(s): after"),
    ("check_step", {"before": "1/2", "after": "2/5", "x": 1}, "unknown argument(s): x"),
    ("check_step", {"before": 1, "after": "2/5"}, "before must be string"),
    ("nope", {}, "unknown tool"),
    ("check_step", ["1/2", "2/5"], "arguments must be an object"),
])
def test_bad_arguments_are_caught_before_running(log, state, name, args, problem):
    result = ToolBox(standard_tools(), log).call(state, name, args)
    assert not result.ok and problem in result.error
    [record] = read(log.path)
    assert record["args_valid"] is False and record["ok"] is False


def test_bad_input_is_not_retried(log, state):
    tools = ToolBox(standard_tools(), log)
    result = tools.call(state, "check_step", {"before": "1/2 +", "after": "2/5"})
    assert not result.ok and "ParseError" in result.error
    assert tools.call(state, "run_rule", {"rule": "no.such_rule", "item": "1/2 + 1/3"}).ok is False
    assert [r["is_retry"] for r in read(log.path)] == [False, False]


def test_unexpected_error_is_retried_once(log, state):
    calls = []

    def flaky(x):
        calls.append(x)
        if len(calls) == 1:
            raise TimeoutError("slow")
        return x

    schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
    result = ToolBox([Tool("flaky", "", schema, flaky)], log).call(state, "flaky", {"x": "hi"})
    assert (result.ok, result.value, len(calls)) == (True, "hi", 2)
    assert [(r["ok"], r["is_retry"]) for r in read(log.path)] == [(False, False), (True, True)]


def test_retrieve_tool_and_declarations(log, state):
    passages = [{"ref": "Prealgebra 2e, 4.5", "text": "x" * 2000, "score": 3.0}]
    tools = ToolBox(standard_tools(lambda q, k=3: passages[:k]), log)
    [p] = tools.call(state, "retrieve", {"query": "adding fractions"}).value
    assert p == {"ref": "Prealgebra 2e, 4.5", "text": "x" * 700}
    names = [d["name"] for d in tools.declarations()]
    assert names == ["check_step", "equivalent", "locate", "run_rule", "retrieve"]


def test_tool_call_records_pass_ds_checker(log, state):
    log.log(state, "session_start", session_index=0, student_group="has_rule", planted=[], attack=None)
    tools = ToolBox(standard_tools(), log)
    tools.call(state, "check_step", {"before": "1/2 + 1/3", "after": "2/5"})
    tools.call(state, "check_step", {"before": "1/2"})
    assert check_files([log.path]) == []


# --- student store -----------------------------------------------------------------------

def test_json_store_survives_a_restart(tmp_path):
    path = tmp_path / "students.json"
    store = JsonStudentStore(path)
    store.record(store.load("s1"), f"rule:{RULE}")
    store.record(store.load("s1"), f"rule:{RULE}")

    again = JsonStudentStore(path)
    assert again.load("s1").times_diagnosed == {f"rule:{RULE}": 2}
    assert again.misconception_prior("s1", f"rule:{RULE}") == 0.8
    assert again.misconception_prior("s2", f"rule:{RULE}") is None


def test_no_memory_store_forgets():
    store = NoMemoryStore()
    memory = store.load("s1")
    store.record(memory, "slip")
    assert store.load("s1").times_diagnosed == {} and store.misconception_prior("s1", "slip") is None
    assert InMemoryStore().misconception_prior("s1", "slip") is None


# --- B0 baseline --------------------------------------------------------------------------

def b0_model(turns):
    """A fake B0: replies from `turns` in order, as JSON."""
    replies = iter(turns)

    def reply(request):
        assert request.system == B0_SYSTEM
        message, correct, stated = next(replies)
        return json.dumps({"message": message, "student_is_correct": correct, "student_misconception": stated})
    return reply


@pytest.fixture
def llm(tmp_path):
    def make(reply, session_tokens=20000):
        cfg = dataclasses.replace(load_config(), log_dir=tmp_path / "llm", cache_dir=tmp_path / "cache",
                                  session_tokens=session_tokens)
        return LLMClient(cfg, backend=FakeBackend(reply), run_id="t")
    return make


def run_b0(log, client):
    return run_session(B0Tutor(log), SimStudent("s1", RULE, judge=judge), [PROBLEM],
                       session_id="sess-0", condition="b0", session_index=0,
                       llm_session=client.session("sess-0", "b0"))


def test_b0_session_runs_and_logs_pass_ds_checker(log, llm):
    client = llm(b0_model([
        ("It looks like you added the numerators and added the denominators. Can pieces of different sizes "
         "be added like that?", False, "Adds numerators and adds denominators."),
        ("Yes, that's right.", True, ""),
    ]))
    result = run_b0(log, client)

    assert [u.kind for u in result.state.transcript if u.speaker == "tutor"] == ["question", "hint", "feedback"]
    assert result.ended_by == "solved"
    records = read(log.path)
    assert all(r["critic_passed"] is None for r in records if r["event"] == "turn")
    [diagnosis] = [r for r in records if r["event"] == "diagnosis"]
    assert (diagnosis["status"], diagnosis["mode"]) == ("unknown", "b0")
    assert diagnosis["stated_misconception"] == "Adds numerators and adds denominators."
    assert records[-1]["tokens_used"] > 0
    assert check_files([log.path, client.log_path]) == []


def test_b0_never_sees_the_answer_and_sees_the_conversation(log, llm):
    client = llm(b0_model([("Check how you added.", False, ""), ("Check again.", False, ""),
                           ("Well done.", True, "")]))
    run_b0(log, client)
    prompts = [r.messages[-1].content for r in client.backend.requests]
    assert not any("5/6" in p for p in prompts[:1])  # the student hasn't said it yet
    assert "Student: I got 2/5." in prompts[0] and "Tutor: Check how you added." in prompts[1]
    assert [r["role"] for r in read(client.log_path)] == ["b0_tutor"] * 3


def test_b0_budget_exhausted_sends_safe_template(log, llm):
    client = llm(b0_model([("Think again about it.", False, "")] * 3), session_tokens=10)
    result = run_b0(log, client)
    assert result.ended_by == "budget"
    assert result.state.transcript[-1].text == SAFE_TEMPLATE
    assert check_files([log.path]) == []


def test_b0_needs_an_llm_session(log):
    state = TutorState("s", "b0", "st", NoMemoryStore().load("st"))
    with pytest.raises(ValueError, match="LLM session"):
        B0Tutor(log).start(state, PROBLEM)
