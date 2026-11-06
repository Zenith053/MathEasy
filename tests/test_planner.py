import json

import pytest

from matheasy.evaluation import SimStudent, check_files
from matheasy.tutor import JsonStudentStore, Orchestrator, StudentMemory, TurnLogger, make_problem, run_session
from matheasy.tutor import planner
from matheasy.tutor.critic import TutorCritic
from matheasy.tutor.placeholders import TemplateHinter
from matheasy.tutor.store import InMemoryStore

RULE = "fractions.add_numerators_denominators"
MID = f"rule:{RULE}"
EXPOSING = make_problem("p1", "1/2 + 1/3")  # the rule says 2/5
NOT_EXPOSING = make_problem("p2", "2(x + 3)")  # the fraction rule doesn't apply


def memory_holding(p=None):
    memory = StudentMemory("s1")
    planner.begin_session(memory, 0)
    planner.diagnosed(memory, MID)
    if p is not None:
        memory.p_holds[MID] = p
    return memory


def test_knowledge_tracing_moves_with_answers():
    right, wrong = StudentMemory("s1"), StudentMemory("s2")
    planner.answered(right, EXPOSING, "5/6", True)
    planner.answered(wrong, EXPOSING, "2/5", False)
    assert right.p_knows["fractions"] > planner.KNOWS_AT_START > wrong.p_knows["fractions"]


def test_diagnosis_then_evidence_updates_p_holds():
    memory = memory_holding()
    assert memory.p_holds[MID] == planner.HOLDS_WHEN_DIAGNOSED
    planner.answered(memory, EXPOSING, "2/5", False)  # the rule's answer
    assert memory.p_holds[MID] > planner.HOLDS_WHEN_DIAGNOSED
    for _ in range(3):  # after giving its answer, three right answers to count as fixed
        planner.answered(memory, EXPOSING, "5/6", True)
    assert memory.p_holds[MID] < planner.FIXED_BELOW


def test_items_that_dont_expose_the_misconception_say_nothing():
    memory = memory_holding()
    planner.answered(memory, NOT_EXPOSING, "2x + 6", True)
    planner.answered(memory, EXPOSING, "1/5", False)  # wrong, but not the rule's answer
    assert memory.p_holds[MID] == planner.HOLDS_WHEN_DIAGNOSED
    assert planner.exposes(MID, "2(x + 3)") is None
    assert planner.exposes("eedi:1287", "1/2 + 1/3") is None  # no code for an Eedi misconception
    assert planner.exposes(MID, "1/2 + 1/3") == "2/5"


def test_a_slip_is_counted_but_not_held():
    memory = StudentMemory("s1")
    planner.diagnosed(memory, "slip")
    assert memory.times_diagnosed == {"slip": 1} and memory.p_holds == {}


def test_misconceptions_drift_back_between_sessions_only():
    memory = memory_holding(0.2)
    planner.begin_session(memory, 0)  # the same session again
    assert memory.p_holds[MID] == 0.2
    planner.begin_session(memory, 1)
    assert memory.p_holds[MID] == pytest.approx(0.2 + 0.8 * planner.RELAPSE)


def test_rechecks_space_out_while_they_pass_and_reset_when_they_fail():
    memory = memory_holding()
    planner.answered(memory, EXPOSING, "5/6", True)
    planner.answered(memory, EXPOSING, "5/6", True)
    planner.end_session(memory)
    assert memory.next_check[MID] == 1  # fixed in session 0: check again in session 1

    schedule = []
    for session, answer in [(1, "5/6"), (3, "5/6"), (7, "2/5")]:
        planner.begin_session(memory, session)
        assert planner.due(memory, session) == [MID]
        planner.answered(memory, EXPOSING, answer, answer == "5/6")
        planner.end_session(memory)
        schedule.append(memory.next_check[MID])
    assert schedule == [3, 7, 8]  # gaps of 2 and 4 after passes; next session after a failure


def test_still_held_is_due_every_session():
    memory = memory_holding()
    planner.end_session(memory)
    assert planner.due(memory, 1) == [MID] and memory.check_gap[MID] == 1
    assert planner.due(memory, 0) == []


def test_plan_puts_a_recheck_first_then_the_weakest_topic():
    memory = memory_holding()
    planner.end_session(memory)
    memory.p_knows = {"fractions": 0.9, "expressions": 0.2}
    pool = ["2(x + 3)", "3/4 > 5/8", "1/2 + 1/3", "x + 3 = 7"]
    problems, rechecks = planner.plan(memory, 1, pool, 3)
    assert problems == ["1/2 + 1/3", "2(x + 3)", "x + 3 = 7"] and rechecks == [MID]
    assert planner.plan(StudentMemory("new"), 0, pool, 2) == (pool[:2], [])


def test_json_store_keeps_the_whole_model(tmp_path):
    path = tmp_path / "students.json"
    store = JsonStudentStore(path, autosave=False)
    memory = store.load("s1")
    store.begin_session(memory, 0)
    store.record(memory, MID)
    store.observe(memory, EXPOSING, "5/6", True)
    assert not path.exists()  # nothing written until save()
    store.end_session(memory)
    store.save()

    again = JsonStudentStore(path).load("s1")
    assert again == memory
    assert JsonStudentStore(path).misconception_prior("s1", MID) == memory.p_holds[MID]


def test_second_session_rechecks_a_fixed_misconception(tmp_path):
    log = TurnLogger("t", "h", tmp_path)
    tutor = Orchestrator(TemplateHinter(), TutorCritic(), InMemoryStore(), log)
    student = SimStudent("s1", RULE, judge=lambda message, misconception: "numerators" in message)
    pool = ["2/5 × 3/5", "1/2 + 1/3", "1/4 + 1/6"]
    for index in range(2):
        problems, rechecks = planner.plan(tutor.store.load("s1"), index, pool, 2)
        run_session(tutor, student, [make_problem(f"p{i}", q) for i, q in enumerate(problems)],
                    session_id=f"sess-{index}", condition="full", session_index=index, rechecks=rechecks)
        if index == 0:
            assert rechecks == [] and not student.holds_misconception

    starts = [json.loads(line) for line in log.path.read_text().splitlines() if '"session_start"' in line]
    assert [s["rechecks"] for s in starts] == [[], [MID]]
    memory = tutor.store.load("s1")
    assert memory.p_holds[MID] < planner.FIXED_BELOW and memory.next_check[MID] == 3
    assert check_files([log.path]) == []
