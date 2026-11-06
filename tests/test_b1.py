import dataclasses
import json

import pytest

from matheasy.diagnosis import held_back_rules
from matheasy.evaluation import SimStudent, check_files
from matheasy.llm import FakeBackend, LLMClient, load_config
from matheasy.tutor import B1Agent, SAFE_TEMPLATE, ToolBox, TurnLogger, make_problem, run_session, standard_tools
from matheasy.tutor.baselines import MAX_TOOL_CALLS

RULE = "fractions.add_numerators_denominators"
PROBLEM = make_problem("p1", "1/2 + 1/3")
PASSAGES = [{"ref": "Prealgebra 2e, 4.5", "text": "To add fractions with different denominators..."}]


def judge(message, misconception):
    return "same size" in message


def tool(name, args):
    return {"action": "tool", "tool": name, "args": args if isinstance(args, str) else json.dumps(args)}


def message(text, correct=False, rule="", stated="", citation=""):
    return {"action": "message", "message": text, "student_is_correct": correct, "misconception_rule": rule,
            "student_misconception": stated, "citation": citation}


def agent_model(steps):
    """A fake B1: each step in order, with every schema field filled in."""
    replies = iter(steps)
    blank = message("")

    def reply(request):
        return json.dumps({**blank, "action": "tool", **next(replies)})
    return reply


@pytest.fixture
def run(tmp_path):
    def go(steps, session_tokens=20000):
        cfg = dataclasses.replace(load_config(), log_dir=tmp_path / "llm", cache_dir=tmp_path / "cache",
                                  session_tokens=session_tokens)
        client = LLMClient(cfg, backend=FakeBackend(agent_model(steps)), run_id="t")
        log = TurnLogger("t", cfg.hash, tmp_path / "turns")
        agent = B1Agent(log, ToolBox(standard_tools(lambda q, k=3: PASSAGES[:k]), log))
        result = run_session(agent, SimStudent("s1", RULE, judge=judge), [PROBLEM], session_id="sess-0",
                             condition="b1", session_index=0, llm_session=client.session("sess-0", "b1"))
        records = [json.loads(line) for line in log.path.read_text().splitlines()]
        client.turn_log = log.path
        return result, records, client
    return go


def events(records, name):
    return [r for r in records if r["event"] == name]


def test_b1_uses_tools_names_a_rule_and_logs_pass_ds_checker(run):
    result, records, client = run([
        tool("check_step", {"before": "1/2 + 1/3", "after": "2/5"}),
        tool("run_rule", {"rule": RULE, "item": "1/2 + 1/3"}),
        tool("retrieve", {"query": "adding fractions"}),
        message("Can you add pieces that are not the same size?", rule=RULE,
                stated="Adds numerators and denominators.", citation="Prealgebra 2e, 4.5"),
        message("Yes, that's right.", correct=True),
    ])
    assert result.ended_by == "solved"
    [diagnosis] = events(records, "diagnosis")
    assert (diagnosis["status"], diagnosis["misconception"], diagnosis["mode"]) == ("clear", f"rule:{RULE}", "b1")
    assert [t["tool"] for t in events(records, "tool_call")] == ["check_step", "run_rule", "retrieve"]
    [hint] = events(records, "hint")
    assert hint["citations"] == ["Prealgebra 2e, 4.5"] and hint["text"].endswith("(See Prealgebra 2e, 4.5.)")
    assert all(t["critic_passed"] is None for t in events(records, "turn"))

    second = client.backend.requests[1].messages
    assert second[-1].content == 'Tool result: {"result": false}'  # the check_step result came back
    assert "5/6" not in client.backend.requests[0].messages[0].content
    assert check_files([client.log_path, client.turn_log]) == []


def test_bad_arguments_are_logged_and_a_repeat_is_a_retry(run):
    _, records, _ = run([
        tool("check_step", "{before: 1/2 + 1/3}"),  # not JSON
        tool("check_step", {"before": "1/2 + 1/3", "after": "2/5"}),
        message("Look at the denominators: are the pieces the same size?"),
        message("Well done.", correct=True),
    ])
    first, second = events(records, "tool_call")
    assert (first["args_valid"], first["ok"], first["is_retry"]) == (False, False, False)
    assert (second["args_valid"], second["ok"], second["is_retry"]) == (True, True, True)


@pytest.mark.parametrize("named", ["no.such_rule", sorted(held_back_rules())[0], ""])
def test_a_name_that_isnt_a_bank_rule_is_unknown(run, named):
    _, records, _ = run([message("Are the pieces the same size?", rule=named, stated="Adds across."),
                         message("Yes.", correct=True)])
    [diagnosis] = events(records, "diagnosis")
    assert (diagnosis["status"], diagnosis["misconception"]) == ("unknown", None)
    assert diagnosis["stated_rule"] == (named or None)


def test_an_unretrieved_citation_is_dropped(run):
    _, records, _ = run([message("Are the pieces the same size?", citation="Some book, 9.9"),
                         message("Yes.", correct=True)])
    assert events(records, "hint")[0]["citations"] == []


def test_too_many_tool_calls_end_in_the_safe_template(run):
    result, records, client = run([tool("check_step", {"before": "1/2", "after": "1/2"})] * (MAX_TOOL_CALLS + 1)
                                  + [message("Yes.", correct=True)] * 8)
    assert result.state.transcript[2].text == SAFE_TEMPLATE
    assert "last tool call" in client.backend.requests[MAX_TOOL_CALLS].messages[-1].content
    assert events(records, "hint")[0]["safe_template"] is True


def test_budget_exhausted_sends_safe_template(run):
    result, records, _ = run([tool("check_step", {"before": "1/2", "after": "1/2"})] * 20, session_tokens=10)
    assert result.ended_by == "budget" and result.state.transcript[-1].text == SAFE_TEMPLATE
