import dataclasses
import json

from matheasy.diagnosis import diagnose
from matheasy.evaluation import SimStudent, check_files
from matheasy.tutor import Orchestrator, TurnLogger, make_problem, run_session
from matheasy.tutor.analysis import analyse
from matheasy.tutor.critic import TutorCritic, leaks_answer
from matheasy.tutor.placeholders import TemplateHinter
from matheasy.tutor.planner import exposes
from matheasy.tutor.store import InMemoryStore

RULE = "fractions.add_numerators_denominators"


def test_a_clear_diagnosis_gives_all_four_parts():
    problem = make_problem("p", "1/2 + 1/3")
    a = analyse(diagnose("1/2 + 1/3", "2/5"), problem)
    assert a.where == "1/2 + 1/3 became 2/5"
    assert a.mistake.startswith("Adds the numerators and adds the denominators")
    assert a.topics == ("fractions",)
    assert len(a.practice) == 2 and problem.question not in a.practice
    assert all(exposes(f"rule:{RULE}", q) for q in a.practice)  # the same mistake gets them wrong
    assert a.text().splitlines()[0] == "Where it went wrong: 1/2 + 1/3 became 2/5."
    assert a.practice_text().startswith("Practise next: ")


def test_parts_that_give_the_answer_away_are_left_out():
    # This rule's description quotes "3 × -2 = 6", and 6 is this problem's answer.
    problem = make_problem("p", "(-2)(-3)")
    d = dataclasses.replace(diagnose("(-2)(-3)", "-6"), status="clear",
                            misconception="rule:negative_numbers.larger_absolute_value_always_wins")
    a = analyse(d, problem)
    assert a.mistake is None and a.where == "-2 × -3 became -6"
    assert not any(leaks_answer(q, "6", "(-2)(-3)") for q in a.practice)


def test_practice_skips_problems_already_worked():
    problem = make_problem("p", "1/2 + 1/3")
    first = analyse(diagnose("1/2 + 1/3", "2/5"), problem)
    again = analyse(diagnose("1/2 + 1/3", "2/5"), problem, seen=first.practice)
    assert not set(first.practice) & set(again.practice)


def test_no_diagnosis_no_analysis():
    a = analyse(None, make_problem("p", "1/2 + 1/3"))
    assert a.text() == "" and a.practice_text() == ""


def test_the_tutor_shows_the_analysis_then_the_hint_then_practice(tmp_path):
    log = TurnLogger("t", "h", tmp_path)
    tutor = Orchestrator(TemplateHinter(), TutorCritic(), InMemoryStore(), log)
    student = SimStudent("s1", RULE, judge=lambda message, misconception: "numerators" in message)
    run_session(tutor, student, [make_problem("p1", "1/2 + 1/3")], session_id="s", condition="full",
                session_index=0)
    records = [json.loads(line) for line in log.path.read_text().splitlines()]
    [hint] = [r for r in records if r["event"] == "hint"]
    lines = hint["text"].splitlines()
    assert lines[0].startswith("Where it went wrong:") and lines[1].startswith("The mistake:")
    assert lines[2] == "Topics: fractions." and lines[-1].startswith("Practise next:")
    assert hint["analysis"]["topics"] == ["fractions"]
    memory = tutor.store.load("s1")
    assert memory.mistakes == {"fractions": 1} and memory.worked == ["1/2 + 1/3"]
    assert check_files([log.path]) == []


def test_nothing_gives_away_another_open_item_in_the_session():
    problem = make_problem("p", "12 ÷ 2 × 3")
    d = diagnose("12 ÷ 2 × 3", "2")
    assert "19 - 11 × 3 + 6" in analyse(d, problem).practice  # 11 answers an earlier, unsolved problem
    shown = analyse(d, problem, unsolved={"10 - 2 + 3": "11"}).practice
    assert shown and not any(leaks_answer(q, "11", "10 - 2 + 3") for q in shown)
