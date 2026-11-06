import dataclasses
import json
import re

import pytest

from matheasy.diagnosis import Diagnosis, Probe, locate
from matheasy.evaluation import SimStudent, check_files
from matheasy.llm import BudgetExceeded
from matheasy.tutor import (
    MAX_FOLLOWUPS, MAX_REWRITES, SAFE_TEMPLATE, Hint, HintRequest, Orchestrator, TurnLogger, TutorState,
    Verdict, extract_answer, make_problem, run_session,
)
from matheasy.tutor.critic import TutorCritic
from matheasy.tutor.placeholders import TemplateHinter
from matheasy.tutor.store import InMemoryStore

RULE = "fractions.add_numerators_denominators"
PROBLEM = make_problem("p1", "1/2 + 1/3")


def keyword_judge(message, misconception):
    return re.split(r"[:,]", misconception)[0].lower() in message.lower()


def make_tutor(tmp_path, hinter=None, critic=None, store=None, **kwargs):
    log = TurnLogger("test", config_hash="abc123", log_dir=tmp_path)
    return Orchestrator(hinter or TemplateHinter(), critic or TutorCritic(), store or InMemoryStore(), log, **kwargs)


def records(tutor, event=None):
    rows = [json.loads(line) for line in tutor.log.path.read_text().splitlines()]
    return [r for r in rows if event is None or r["event"] == event]


def session(tutor, student, problems=(PROBLEM,), index=0, **kwargs):
    return run_session(tutor, student, list(problems), session_id=f"sess-{index}", condition="full",
                       session_index=index, **kwargs)


def new_state(tutor):
    return TutorState("sess", "full", "s1", tutor.store.load("s1"))


class RecordingHinter:
    def __init__(self, text="Check how you combined the denominators.", error=None):
        self.text, self.error, self.calls = text, error, []

    def hint(self, request, feedback=None, **_):
        self.calls.append((request, feedback))
        if self.error:
            raise self.error
        return Hint(self.text, ("Prealgebra 2e, 4.5",))


class RejectingCritic:
    """Rejects the first `n` hints."""

    def __init__(self, n):
        self.n = n

    def review(self, text, kind, problem, **_):
        if kind == "hint" and self.n > 0:
            self.n -= 1
            return Verdict(False, ("too vague",))
        return Verdict(True)


def always_ambiguous(before, after, **kwargs):
    probe = Probe("1/3 + 1/5", {"rule:a": "2/8", "rule:b": "2/15"})
    return Diagnosis("ambiguous", None, {"rule:a": 0.5, "rule:b": 0.5}, probe, locate(before, after), "full")


# --- the whole session, on B's diagnoser and D's student -------------------------

def test_session_fixes_the_misconception_and_logs_pass_ds_checker(tmp_path):
    tutor = make_tutor(tmp_path)
    student = SimStudent("s1", RULE, judge=keyword_judge)
    result = session(tutor, student, [PROBLEM, make_problem("p2", "2/3 + 1/4")],
                     pre_items=["1/4 + 2/5"], post_items=["2/7 + 1/3"])

    kinds = [u.kind for u in result.state.transcript if u.speaker == "tutor"]
    assert kinds == ["question", "hint", "feedback", "question", "feedback"]
    assert (result.pre_correct, result.post_correct, result.ended_by) == (0, 1, "solved")
    [diagnosis] = records(tutor, "diagnosis")
    assert (diagnosis["status"], diagnosis["misconception"]) == ("clear", f"rule:{RULE}")
    assert records(tutor, "session_end")[0]["holds"] == []
    assert check_files([tutor.log.path]) == []


def test_every_turn_passed_the_critic_and_is_paired_with_ground_truth(tmp_path):
    tutor = make_tutor(tmp_path)
    session(tutor, SimStudent("s1", RULE, judge=keyword_judge))
    turns = records(tutor, "turn")
    assert [t["turn"] for t in turns] == list(range(1, len(turns) + 1))
    assert all(t["critic_passed"] for t in turns)
    assert turns[0]["followed_rule"] and turns[0]["student_answer"] == "2/5"


def test_student_without_misconception_gets_no_diagnosis(tmp_path):
    tutor = make_tutor(tmp_path)
    result = session(tutor, SimStudent("s1", None), post_items=["2/7 + 1/3"])
    assert records(tutor, "diagnosis") == []
    assert result.post_correct == 1
    assert check_files([tutor.log.path]) == []


def test_session_stops_at_max_turns_when_hints_never_land(tmp_path):
    tutor = make_tutor(tmp_path)
    student = SimStudent("s1", RULE, judge=lambda message, misconception: False)
    result = session(tutor, student, max_turns_per_problem=3)
    assert result.ended_by == "max_turns"
    assert records(tutor, "session_end")[0]["holds"] == [f"rule:{RULE}"]


def test_memory_carries_over_between_sessions(tmp_path):
    tutor = make_tutor(tmp_path)
    for i in range(2):
        session(tutor, SimStudent("s1", RULE, judge=lambda m, d: False), index=i, max_turns_per_problem=1)
    assert tutor.store.load("s1").times_diagnosed[f"rule:{RULE}"] >= 2
    assert tutor.store.misconception_prior("s1", f"rule:{RULE}") > 0.9  # still giving its answer


# --- orchestrator rules ------------------------------------------------------------

def test_hinter_gets_the_misconception_in_words_but_never_an_answer(tmp_path):
    hinter = RecordingHinter()
    tutor = make_tutor(tmp_path, hinter=hinter)
    state = new_state(tutor)
    tutor.start(state, PROBLEM)
    tutor.respond(state, "I got 2/5.")

    request = hinter.calls[0][0]
    assert request.misconception.startswith("Adds the numerators and adds the denominators")
    assert request.wrong_part == "1/2 + 1/3 became 2/5"
    assert not any(f.name in ("correct_answer", "expected") for f in dataclasses.fields(HintRequest))
    assert "5/6" not in json.dumps(dataclasses.asdict(request))


def test_rewrite_gets_critic_feedback(tmp_path):
    hinter = RecordingHinter()
    tutor = make_tutor(tmp_path, hinter=hinter, critic=RejectingCritic(1))
    state = new_state(tutor)
    tutor.start(state, PROBLEM)

    assert tutor.respond(state, "2/5").kind == "hint"
    assert hinter.calls[0][1] is None and hinter.calls[1][1] == ["too vague"]
    [hint] = records(tutor, "hint")
    assert (hint["rewrites"], hint["safe_template"], hint["citations"]) == (1, False, ["Prealgebra 2e, 4.5"])


def test_safe_template_after_rewrite_limit(tmp_path):
    tutor = make_tutor(tmp_path, hinter=RecordingHinter(), critic=RejectingCritic(99))
    state = new_state(tutor)
    tutor.start(state, PROBLEM)

    reply = tutor.respond(state, "2/5")
    assert (reply.kind, reply.text) == ("safe_template", SAFE_TEMPLATE)
    [hint] = records(tutor, "hint")
    assert hint["safe_template"] and len(hint["critic_reasons"]) == MAX_REWRITES + 1


def test_followup_limit_then_general_hint(tmp_path):
    hinter = RecordingHinter()
    tutor = make_tutor(tmp_path, hinter=hinter, diagnose_fn=always_ambiguous)
    state = new_state(tutor)
    tutor.start(state, PROBLEM)

    replies = [tutor.respond(state, "2/5")] + [tutor.respond(state, "2/8") for _ in range(MAX_FOLLOWUPS)]
    assert [r.kind for r in replies] == ["probe"] * MAX_FOLLOWUPS + ["hint"]
    assert replies[0].item == "1/3 + 1/5"  # D's student is told which maths to answer
    assert replies[-1].item == PROBLEM.question
    assert hinter.calls[0][0].misconception is None
    assert [d["follow_ups"] for d in records(tutor, "diagnosis")] == [0, 1, 2]


def test_budget_exhausted_falls_back_and_ends_the_session(tmp_path):
    tutor = make_tutor(tmp_path, hinter=RecordingHinter(error=BudgetExceeded("no tokens")))
    result = session(tutor, SimStudent("s1", RULE, judge=keyword_judge), [PROBLEM, make_problem("p2", "2/3 + 1/4")])

    assert result.ended_by == "budget"
    assert result.state.transcript[-1].text == SAFE_TEMPLATE
    assert "BudgetExceeded" in records(tutor, "hint")[0]["error"]
    assert check_files([tutor.log.path]) == []


def test_unreadable_reply_asks_to_retype(tmp_path):
    tutor = make_tutor(tmp_path)
    state = new_state(tutor)
    tutor.start(state, PROBLEM)
    reply = tutor.respond(state, "no idea")
    assert reply.kind == "question" and "type just your answer" in reply.text
    assert reply.item == PROBLEM.question


def test_b2_mode_reaches_the_diagnoser(tmp_path):
    tutor = make_tutor(tmp_path, mode="b2")
    state = new_state(tutor)
    tutor.start(state, PROBLEM)
    tutor.respond(state, "2/5")
    assert records(tutor, "diagnosis")[0]["mode"] == "b2"


def test_basic_critic_blocks_a_leaked_answer():
    critic = TutorCritic()
    assert not critic.review("The answer is 5/6.", "hint", PROBLEM).passed
    assert critic.review("Work out 1/2 + 1/3", "question", PROBLEM).passed


@pytest.mark.parametrize("text, answer", [
    ("I got 2/5.", "2/5"), ("2/5", "2/5"), ("0.25", "0.25"), ("2x + 6", "2x + 6"),
    ("maybe 3 or 4", "4"), ("no idea", None),
    ("I got x = 4.", "x = 4"),  # Rahul's report: this used to read only "4"
    ("I got 2x + 6.", "2x + 6"), ("I got 1/3 > 1/4.", "1/3 > 1/4"), ("I got 3 1/2.", "3 1/2"),
    ("I got 25%.", "25%"), ("It's 5/6 I think", "5/6"), ("a 3/4", "3/4"), ("x = 4 I think", "x = 4"),
])
def test_extract_answer(text, answer):
    assert extract_answer(text) == answer


def test_equation_answer_is_marked_correct(tmp_path):
    tutor = make_tutor(tmp_path)
    result = session(tutor, SimStudent("s1", None), [make_problem("e1", "2x + 4 = 10")])
    assert result.state.transcript[1].text == "I got x = 3."
    assert [u.kind for u in result.state.transcript if u.speaker == "tutor"] == ["question", "feedback"]
    assert records(tutor, "diagnosis") == []


def test_runner_starts_each_session_for_the_student(tmp_path):
    tutor = make_tutor(tmp_path)
    student = SimStudent("s1", RULE, judge=keyword_judge, relapse_rate=1.0)
    for i in range(2):
        session(tutor, student, index=i)
    starts = records(tutor, "session_start")
    assert [(r["session_index"], r["relapsed"]) for r in starts] == [(0, False), (1, True)]
    assert student.session_index == 1
    assert check_files([tutor.log.path]) == []


# --- B's probe bank ------------------------------------------------------------------------

def test_a_tie_is_settled_by_one_of_bs_probes_and_stays_settled(tmp_path):
    from matheasy.evaluation.metrics import accepted
    rule = "fractions.multiply_across_for_division"
    tutor = make_tutor(tmp_path)
    student = SimStudent("s1", rule, judge=lambda message, misconception: False)  # never fixed
    session(tutor, student, [make_problem("p1", "2/3 ÷ 4/5")], max_turns_per_problem=4)

    turns = records(tutor, "turn")
    probes = [t for t in turns if t["tutor_kind"] == "probe"]
    diagnoses = records(tutor, "diagnosis")
    assert len(probes) == 1 and probes[0]["item"] == diagnoses[0]["probe"]["question"]
    assert diagnoses[0]["status"] == "ambiguous"
    # Settled by the probe answer, and still settled when the same wrong answer comes again.
    assert all(d["status"] == "clear" and d["misconception"] in accepted(f"rule:{rule}", "has_rule")
               for d in diagnoses[1:])
    assert len(diagnoses) > 2 and check_files([tutor.log.path]) == []


def test_runner_tests_never_reuse_a_probe_or_a_tutored_item():
    from matheasy.diagnosis.probes import probe_items
    from matheasy.evaluation import overlapping, roster
    from matheasy.tutor.runner import SLIP_ONLY, materials
    for student in roster(keyword_judge, slip_only=SLIP_ONLY):
        pool, forms = materials(student, sessions=2)
        tests = [item for form in forms for item in form.pre + form.post]
        assert overlapping(tests, probe_items()) == [] and overlapping(tests, pool) == []
