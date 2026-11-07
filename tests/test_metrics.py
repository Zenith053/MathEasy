"""D's week-2 metrics, on real offline sessions through C's tutor and on hand-built logs."""

import re

import pytest

from matheasy.evaluation import SimStudent, check_files, check_records, read_jsonl, roster
from matheasy.evaluation import metrics
from matheasy.evaluation.metrics import compute, states_answer
from matheasy.tutor import Orchestrator, TurnLogger, make_problem, run_session
from matheasy.tutor.critic import TutorCritic
from matheasy.tutor.placeholders import TemplateHinter
from matheasy.tutor.store import InMemoryStore

ADDS_ACROSS = "fractions.add_numerators_denominators"


def keyword_judge(message, misconception):
    return re.split(r"[:,]", misconception)[0].lower() in message.lower()


# ---------------------------------------------------------------- real sessions, offline

def run(tmp_path, students, problems=("1/2 + 1/3", "2/3 + 1/4")):
    log = TurnLogger("test", config_hash="abc123", log_dir=tmp_path)
    tutor = Orchestrator(TemplateHinter(), TutorCritic(), InMemoryStore(), log)
    for i, student in enumerate(students):
        run_session(tutor, student, [make_problem(f"p{k}", q) for k, q in enumerate(problems)],
                    session_id=f"sess-{i}", condition="full", session_index=0,
                    pre_items=["1/4 + 2/5"], post_items=["2/7 + 1/3"])
    return log.path


def test_metrics_on_placeholder_logs(tmp_path):
    path = run(tmp_path, [SimStudent("s1", ADDS_ACROSS, judge=keyword_judge),
                          SimStudent("slip_only#0", None, slip_rate=1.0)])
    assert check_files([path]) == []
    result = compute(read_jsonl([path]))["full"]

    has_rule = result["has_rule"]
    assert (has_rule["sessions"], has_rule["misconception_episodes"], has_rule["top1_accuracy"]) == (1, 1, 1.0)
    assert has_rule["follow_ups_per_confirmed"] == 0
    assert (has_rule["leak_rate"], has_rule["unsafe_action_rate"], has_rule["safe_template_rate"]) == (0, 0, 0)

    slip_only = result["slip_only"]
    assert slip_only["slip_episodes"] > 0
    assert slip_only["over_claiming_rate"] == 0  # B's diagnoser never claims a misconception for these slips
    assert slip_only["misconception_episodes"] == 0 and slip_only["top1_accuracy"] is None

    assert result["all"]["sessions"] == 2


def test_held_back_students_run_through_the_tutor(tmp_path):
    students = [s for s in roster(keyword_judge) if s.misconception in (
        "fractions.subtract_across", "decimals.right_align_decimals")]
    path = run(tmp_path, students, problems=("3/4 - 1/2", "3.5 + 2.75"))
    assert check_files([path]) == []
    # Both are known only as text (eedi_status), so their diagnoses are scored against the Eedi IDs
    known = compute(read_jsonl([path]))["full"]["known_as_text"]
    assert known["sessions"] == 2 and known["misconception_episodes"] > 0
    assert known["top1_accuracy"] is not None and known["unknowable_episodes"] == 0


def test_cli(tmp_path, capsys):
    path = run(tmp_path, [SimStudent("s1", ADDS_ACROSS, judge=keyword_judge)])
    assert metrics.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "== condition full" in out and "top1_accuracy" in out


# ---------------------------------------------------------------- hand-built logs

BASE = {"ts": "2026-10-30T12:00:00.000+00:00", "run_id": "run-1", "config_hash": "abc123"}


class Log:
    def __init__(self):
        self.records = []

    def add(self, session, condition, event, **fields):
        self.records.append({**BASE, "session_id": session, "student_id": session, "condition": condition,
                             "event": event, **fields})

    def start(self, session, group, planted, condition="full"):
        self.add(session, condition, "session_start", session_index=0, student_group=group,
                 planted=planted, attack=None, relapsed=False)

    def turn(self, session, n, *, condition="full", tutor="Work out 1/2 + 1/3", item="1/2 + 1/3",
             right="5/6", answer="2/5", correct=False, followed=True, critic=True):
        self.add(session, condition, "turn", turn=n, tutor_text=tutor, critic_passed=critic, item=item,
                 correct_answer=right, student_text=f"I got {answer}.", student_answer=answer,
                 student_correct=correct, followed_rule=followed, holds_misconception=followed)

    def diagnosis(self, session, n, status, mid=None, follow_ups=0, condition="full", **extra):
        self.add(session, condition, "diagnosis", turn=n, status=status, misconception=mid,
                 beliefs={mid: 0.9} if mid else {}, follow_ups=follow_ups, mode=condition, **extra)

    def result(self, check=True, **kwargs):
        if check:
            assert check_records((str(i), r) for i, r in enumerate(self.records)) == []
        return compute(((str(i), r) for i, r in enumerate(self.records)), **kwargs)


def test_twin_counts_as_correct():
    log = Log()
    log.start("s1", "has_rule", [f"rule:{ADDS_ACROSS}"])
    log.turn("s1", 1)
    log.diagnosis("s1", 1, "clear", "rule:fractions.common_denominator_numerator")
    assert log.result()["full"]["has_rule"]["top1_accuracy"] == 1.0


@pytest.mark.parametrize("planted,diagnosed,right", [
    # B's diagnoser names the rule that stands for a twin
    ("ratios_proportions.ratio_as_division_only", "ratios_proportions.swap_ratios_or_units", True),
    ("ratios_proportions.swap_ratios_or_units", "ratios_proportions.ratio_as_division_only", True),
    # not twins: they only agree on some problems, so a probe can separate them
    ("order_of_operations.strict_left_to_right", "order_of_operations.ignore_parentheses", False),
])
def test_twins_come_from_bs_registry(planted, diagnosed, right):
    assert (f"rule:{diagnosed}" in metrics.accepted(f"rule:{planted}", "has_rule")) is right


def test_episode_scored_on_its_last_diagnosis():
    log = Log()
    log.start("s1", "has_rule", [f"rule:{ADDS_ACROSS}"])
    log.turn("s1", 1)
    log.diagnosis("s1", 1, "ambiguous", follow_ups=0)
    log.turn("s1", 2, tutor="Before we go on, what is 1/3 + 1/5?", item="1/3 + 1/5", right="8/15", answer="2/8")
    log.diagnosis("s1", 2, "ambiguous", follow_ups=1)
    log.turn("s1", 3, tutor="And 1/4 + 1/6?", item="1/4 + 1/6", right="5/12", answer="2/10")
    log.diagnosis("s1", 3, "clear", f"rule:{ADDS_ACROSS}", follow_ups=2)
    m = log.result()["full"]["has_rule"]
    assert (m["misconception_episodes"], m["top1_accuracy"], m["follow_ups_per_confirmed"]) == (1, 1.0, 2)


def test_wrong_and_unclear_diagnoses():
    log = Log()
    log.start("s1", "has_rule", [f"rule:{ADDS_ACROSS}"])
    log.turn("s1", 1)
    log.diagnosis("s1", 1, "clear", "rule:fractions.ignore_denominators")
    log.turn("s1", 2)
    log.diagnosis("s1", 2, "unknown")
    log.turn("s1", 3)
    log.diagnosis("s1", 3, "clear", f"rule:{ADDS_ACROSS}")
    assert log.result()["full"]["has_rule"]["top1_accuracy"] == pytest.approx(1 / 3)


def test_known_as_text_matches_the_eedi_id(monkeypatch):
    monkeypatch.setattr(metrics, "_eedi_status", lambda: {"fractions.subtract_across": 1287})
    log = Log()
    log.start("s1", "known_as_text", ["rule:fractions.subtract_across"])
    log.turn("s1", 1, tutor="Work out 3/4 - 1/2", item="3/4 - 1/2", right="1/4", answer="1")
    log.diagnosis("s1", 1, "clear", "eedi:1287")
    assert log.result()["full"]["known_as_text"]["top1_accuracy"] == 1.0


def test_never_seen_claims_and_over_claiming():
    log = Log()
    log.start("s1", "never_seen", ["rule:fractions.subtract_across"])
    log.turn("s1", 1, item="3/4 - 1/2", right="1/4", answer="1")
    log.diagnosis("s1", 1, "clear", "eedi:42")
    log.start("s2", "slip_only", [])
    log.turn("s2", 1, answer="5/7", followed=False)
    log.diagnosis("s2", 1, "clear", "slip")
    log.turn("s2", 2, answer="4/6", followed=False)
    log.diagnosis("s2", 2, "clear", f"rule:{ADDS_ACROSS}")
    result = log.result()["full"]
    assert result["never_seen"]["unknowable_claim_rate"] == 1.0
    assert result["never_seen"]["top1_accuracy"] is None
    assert (result["slip_only"]["slip_episodes"], result["slip_only"]["over_claiming_rate"]) == (2, 0.5)


def test_b0_text_diagnosis_needs_a_judge():
    log = Log()
    log.start("s1", "has_rule", [f"rule:{ADDS_ACROSS}"], condition="b0")
    log.turn("s1", 1, condition="b0", critic=None)
    log.diagnosis("s1", 1, "unknown", condition="b0",
                  stated_misconception="The student added the tops and the bottoms.")
    assert log.result()["b0"]["has_rule"]["top1_accuracy"] is None  # not scored without a judge

    def judge(stated, description):
        return "tops and the bottoms" in stated and "numerators" in description

    assert log.result(text_judge=judge)["b0"]["has_rule"]["top1_accuracy"] == 1.0
    assert log.result()["b0"]["has_rule"]["unsafe_action_rate"] is None  # no critic in b0


@pytest.mark.parametrize("text,answer,item,leaks", [
    ("The answer is 5/6.", "5/6", "1/2 + 1/3", True),
    ("That makes 10/12, doesn't it?", "5/6", "1/2 + 1/3", True),
    ("Work out 1/2 + 1/3", "5/6", "1/2 + 1/3", False),
    ("Remember that 1/2 is 3/6.", "5/6", "1/2 + 1/3", False),
    ("So x = 4.", "x = 4", "x + 3 = 7", True),
    ("Try subtracting 3 from 7.", "x = 4", "x + 3 = 7", False),
    ("Expanded, it is 2x + 6", "2*x + 6", "2(x + 3)", True),
    ("Multiply 2 by each term.", "2*x + 6", "2(x + 3)", False),
    ("It comes to 0.55 in the end", "0.55", "0.25 + 0.3", True),
    ("That's 55%.", "0.55", "0.25 + 0.3", True),
    ("Look again at x + 3 = 7.", "x = 4", "x + 3 = 7", False),  # equivalent to x = 4, but it's the question
    ("Work out 10 - x = 4", "x = 4", ["x + 3 = 7", "10 - x = 4"], False),  # posing the next question
    ("Here x = 4, see?", "x = 4", "2x + 4 = 12", True),  # 4 is in the question, but x = 4 is stated
    ("Expand 2(x + 3) carefully.", "2*x + 6", "2(x + 3)", False),
    ("I think 1/3 > 1/4 here", "1/3 > 1/4", "1/2 + 1/3", True),
])
def test_states_answer(text, answer, item, leaks):
    assert states_answer(text, answer, item) is leaks


def test_leak_rate_and_unsafe_actions():
    log = Log()
    log.start("s1", "has_rule", [f"rule:{ADDS_ACROSS}"])
    log.turn("s1", 1)
    log.turn("s1", 2, tutor="Nearly: it should be 5/6. Try again: 1/2 + 1/3", critic=False)
    log.start("s2", "has_rule", [f"rule:{ADDS_ACROSS}"])
    log.turn("s2", 1)
    log.turn("s2", 2, answer="5/6", correct=True, followed=False)
    log.turn("s2", 3, tutor="Yes, 5/6 is right.", item=None, right=None, answer="", correct=None, followed=False)
    m = log.result(check=False)["full"]["has_rule"]  # the checker rightly rejects the unchecked message
    assert m["leak_rate"] == 0.5  # s2 praised 5/6 only after the student got it
    assert m["unsafe_action_rate"] == pytest.approx(1 / 5)


def test_repeating_the_students_right_answer_is_not_a_leak():
    log = Log()
    log.start("s1", "has_rule", [f"rule:{ADDS_ACROSS}"])
    log.turn("s1", 1, tutor="Work out x + 3 = 7", item="x + 3 = 7", right="x = 4", answer="x = 4",
             correct=True, followed=False)
    log.turn("s1", 2, tutor="Look again at your answer 4.", item="x + 3 = 7", right="x = 4", answer="x = 4",
             correct=True, followed=False)
    assert log.result()["full"]["has_rule"]["leak_rate"] == 0


def test_posing_a_new_question_is_not_a_leak():
    log = Log()
    log.start("s1", "has_rule", [f"rule:{ADDS_ACROSS}"])
    log.turn("s1", 1, tutor="Work out x + 3 = 7", item="x + 3 = 7", right="x = 4", answer="x = 10")
    log.turn("s1", 2, tutor="Work out 10 - x = 4", item="10 - x = 4", right="x = 6", answer="x = -6")
    log.turn("s1", 3, tutor="Remember x + 3 = 7 gave 4. Now x - 2 = 5?", item="x - 2 = 5", right="x = 7",
             answer="x = 3")
    [s] = metrics.sessions((str(i), r) for i, r in enumerate(log.records))
    assert [t["turn"] for t in metrics.leaking_turns(s)] == [3]  # turn 3 gives away the open x + 3 = 7


def test_tool_metrics():
    log = Log()
    log.start("s1", "has_rule", [f"rule:{ADDS_ACROSS}"])
    for valid, ok, retry in [(True, True, False), (False, False, False), (True, False, False), (True, True, True)]:
        log.add("s1", "full", "tool_call", turn=1, tool="equivalent", args_valid=valid, ok=ok,
                error=None if ok else "boom", is_retry=retry)
    m = log.result()["full"]["has_rule"]
    assert (m["tool_calls"], m["tool_call_validity"], m["tool_recovery_rate"]) == (4, 0.75, 1.0)
