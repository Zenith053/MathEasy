import json

import pytest
import yaml

from matheasy.diagnosis import check_step, diagnose
from matheasy.evaluation import (
    LLMFeedbackJudge, SimStudent, check_files, check_record, check_records, correct_answer, item_key,
    overlapping, student_group,
)
from matheasy.llm import FakeBackend, LLMClient, load_config
from matheasy.llm.config import DEFAULT_CONFIG_PATH

ADDS_ACROSS = "fractions.add_numerators_denominators"


def never(message, misconception):
    return False


def mentions_denominators(message, misconception):
    return "denominator" in message


# ---------------------------------------------------------------- the student

def test_answers_by_its_rule():
    s = SimStudent("s1", ADDS_ACROSS, judge=never)
    r = s.respond("Work out 1/2 + 1/3", item="1/2 + 1/3")
    assert (r.answer, r.correct_answer, r.correct, r.followed_rule) == ("2/5", "5/6", False, True)
    assert r.text == "I got 2/5."


def test_answers_correctly_where_the_rule_does_not_apply():
    s = SimStudent("s1", ADDS_ACROSS, judge=never)
    r = s.respond("Work out 2/3 × 3/4", item="2/3 × 3/4")
    assert (r.answer, r.correct, r.followed_rule) == ("1/2", True, False)


def test_keeps_misconception_until_feedback_targets_it():
    s = SimStudent("s1", ADDS_ACROSS, judge=mentions_denominators)
    r = s.respond("Not quite. Check your work and try 1/4 + 1/5.", item="1/4 + 1/5")
    assert r.feedback_targeted is False and r.answer == "2/9" and r.holds_misconception
    r = s.respond("You can't add the denominators: rewrite both over 20 first. Try 1/4 + 1/5.",
                  item="1/4 + 1/5")
    assert r.feedback_targeted is True and r.answer == "9/20" and not r.holds_misconception
    r = s.respond("Next: 1/2 + 1/3", item="1/2 + 1/3")
    assert r.feedback_targeted is None and r.answer == "5/6"  # fixed students are not judged again


def test_hints_to_fix():
    s = SimStudent("s1", ADDS_ACROSS, judge=mentions_denominators, hints_to_fix=2)
    s.respond("Look at the denominators.")
    assert s.holds_misconception
    s.respond("The denominators must match before you add.")
    assert not s.holds_misconception


def test_student_with_no_misconception():
    s = SimStudent("s1", None)
    r = s.respond("1/2 + 1/3?", item="1/2 + 1/3")
    assert (r.answer, r.correct, r.feedback_targeted, s.group) == ("5/6", True, None, "slip_only")
    with pytest.raises(ValueError):
        SimStudent("s2", ADDS_ACROSS)  # a misconception needs a judge


def test_held_back_rule_brings_its_own_code():
    # B's runner refuses held-back rules, so D passes the rule in.
    subtract_across = lambda item: "2/1" if item == "5/4 - 3/3" else None
    s = SimStudent("s1", "fractions.subtract_across", judge=never, predict=subtract_across,
                   description="Subtracts numerators and denominators: 5/4 - 3/3 = 2/1.")
    assert s.respond("5/4 - 3/3?", item="5/4 - 3/3").answer == "2/1"
    assert s.misconception_id == "rule:fractions.subtract_across"


def test_student_groups():
    assert student_group(ADDS_ACROSS) == "has_rule"
    assert student_group("fractions.subtract_across") == "unchecked"  # eedi_status not filled in yet
    assert student_group(None) == "slip_only"


def test_correct_answer():
    assert correct_answer("1/2 + 1/3") == "5/6"
    assert correct_answer("0.25 + 0.3") == "0.55"
    assert correct_answer("0.1 ÷ 0.3") == "1/3"  # does not terminate: stays a fraction
    assert correct_answer("2(x + 3)") == "2*x + 6"


@pytest.fixture
def llm(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["provider"] = "fake"
    raw["cache"]["dir"] = str(tmp_path / "cache")
    raw["logging"]["dir"] = str(tmp_path / "logs")
    path = tmp_path / "llm.yaml"
    path.write_text(yaml.safe_dump(raw))
    backend = FakeBackend(lambda req: json.dumps({"targets": "denominator" in req.messages[-1].content}))
    return LLMClient(load_config(path), backend=backend, run_id="test"), backend


def test_llm_judge(llm):
    client, backend = llm
    judge = LLMFeedbackJudge(client.session("sess-1", "full", budget_tokens=None))
    s = SimStudent("s1", ADDS_ACROSS, judge=judge)
    assert s.respond("Try again.").feedback_targeted is False
    assert s.respond("Make the denominators the same first.").feedback_targeted is True
    request = backend.requests[-1]
    assert "Adds the numerators and adds the denominators" in request.system  # the rule's docstring
    assert request.json_schema is not None


# ---------------------------------------------------------------- the week-1 path, without C's orchestrator

def records_for_one_session(student, condition="full"):
    base = {"ts": "2026-10-26T12:00:00.000+00:00", "run_id": "run-1", "session_id": "sess-1",
            "student_id": student.student_id, "condition": condition, "config_hash": "abc123"}
    reply = student.respond("Work out 1/2 + 1/3", item="1/2 + 1/3")
    assert not check_step(reply.item, reply.answer)
    d = diagnose(reply.item, reply.answer, student_id=student.student_id)
    return [
        {**base, "event": "session_start", "session_index": 0, "student_group": student.group,
         "planted": [student.misconception_id], "attack": None},
        {**base, "event": "turn", "turn": 0, "tutor_text": "Work out 1/2 + 1/3", "critic_passed": True,
         "item": reply.item, "correct_answer": reply.correct_answer, "student_text": reply.text,
         "student_answer": reply.answer, "student_correct": reply.correct,
         "followed_rule": reply.followed_rule, "holds_misconception": reply.holds_misconception},
        {**base, "event": "diagnosis", "turn": 0, "follow_ups": 0, **d.to_log()},
        {**base, "event": "session_end", "turns": 1, "ended_by": "max_turns", "tokens_used": 0,
         "holds": [student.misconception_id]},
    ]


def test_student_through_diagnoser_passes_the_checker():
    student = SimStudent("student-001", ADDS_ACROSS, judge=never)
    records = records_for_one_session(student)
    assert records[2]["misconception"] == student.misconception_id  # diagnosed the planted misconception
    assert check_records((f"line {i}", r) for i, r in enumerate(records)) == []


# ---------------------------------------------------------------- the checker

@pytest.fixture
def good():
    return records_for_one_session(SimStudent("student-001", ADDS_ACROSS, judge=never))


def test_missing_and_mistyped_fields(good):
    turn = dict(good[1])
    del turn["tutor_text"]
    turn["turn"] = True
    assert check_record(turn) == ["turn: turn should be int, got True", "turn: missing tutor_text"]


def test_bad_values(good):
    start = {**good[0], "condition": "B0", "student_group": "easy", "planted": ["fractions.x"]}
    problems = check_record(start)
    assert len(problems) == 3 and "not a misconception ID" in problems[2]
    assert check_record({"event": "chat"}) == ["unknown event 'chat'; expected one of " + str(sorted(
        ["session_start", "turn", "diagnosis", "hint", "tool_call", "test", "session_end"]))]
    clear_without_id = {**good[2], "misconception": None}
    assert "exactly when status is clear" in check_record(clear_without_id)[0]


def test_full_tutor_message_must_pass_the_critic(good):
    assert "without passing the critic" in check_record({**good[1], "critic_passed": False})[0]
    assert check_record({**good[1], "condition": "b0", "critic_passed": None}) == []


def test_held_back_rule_in_a_diagnosis_is_a_leak(good):
    held = frozenset({"fractions.subtract_across"})
    leak = {**good[2], "misconception": "rule:fractions.subtract_across",
            "beliefs": {"rule:fractions.subtract_across": 1.0}}
    assert all("leaked into the bank" in p for p in check_record(leak, held))
    planted = {**good[0], "planted": ["rule:fractions.subtract_across"]}
    assert check_record(planted, held) == []  # students may hold it


def test_cross_record_checks(good):
    other_hash = {**good[3], "config_hash": "zzz999"}
    problems = check_records([("a:1", good[1]), ("a:2", good[0]), ("a:3", other_hash)])
    assert problems[0] == "a:1: turn before session_start of session 'sess-1'"
    assert "more than one config_hash" in problems[1]


def test_check_files_reads_llm_logs_too(tmp_path, good):
    runs = tmp_path / "runs"
    (runs / "llm_logs").mkdir(parents=True)
    (runs / "run-1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in good))
    llm_call = {"event": "call", "config_hash": "abc123", "session_id": "sess-1"}
    (runs / "llm_logs" / "run-1.jsonl").write_text(json.dumps(llm_call) + "\n")
    assert check_files([runs]) == []
    (runs / "llm_logs" / "run-1.jsonl").write_text(json.dumps({**llm_call, "config_hash": "new"}) + "\n")
    assert "more than one config_hash" in check_files([runs])[0]


# ---------------------------------------------------------------- tests vs probes

def test_items_match_in_any_order():
    assert item_key("1/3 + 1/4") == item_key("1/4 + 1/3")
    assert item_key("3/4 - 1/4") == item_key("1/4 - 3/4")
    assert item_key("0.30 + 0.2") == item_key("0.2 + 0.3")
    assert item_key("2/4 + 1/3") != item_key("1/2 + 1/3")  # the student sees different numbers
    assert item_key("1/3 + 1/4") != item_key("1/3 × 1/4")


def test_overlapping():
    probes = ["1/3 + 1/3", "1/4 + 1/5"]
    assert overlapping(["1/5 + 1/4", "2/7 + 1/7"], probes) == [("1/5 + 1/4", "1/4 + 1/5")]
    assert overlapping(["2/7 + 1/7"], probes) == []
