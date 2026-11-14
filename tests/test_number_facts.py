"""Wrong basic number facts (diagnosis/facts.py): found by code with no LLM call, recorded as a
slip with the reason, and raised to a weak number fact when the same fact goes wrong again."""

import pytest

from matheasy.diagnosis import SLIP, diagnose, fact_id, fact_of, number_fact, split_id
from matheasy.tutor import make_problem, planner
from matheasy.tutor.describe import misconception_text
from matheasy.tutor.state import StudentMemory


class NoLLM:
    """A session that fails the test if diagnose() tries to call the model."""
    def __getattr__(self, name):
        raise AssertionError("the LLM was called for a number fact")


@pytest.mark.parametrize("before, after, reason, where", [
    ("6 + 3", "8", "6 + 3 is 9, not 8", ("6 + 3", "8")),
    ("7 × 8", "54", "7 × 8 is 56, not 54", ("7 × 8", "54")),
    ("15 - 7", "9", "15 - 7 is 8, not 9", ("15 - 7", "9")),
    ("56 ÷ 7", "9", "56 ÷ 7 is 8, not 9", None),
    ("2x + 6 + 3", "2x + 8", "6 + 3 is 9, not 8", ("6 + 3", "8")),      # inside an algebra line
    ("7 × 8 + 6", "60", "7 × 8 is 56, not 54", ("7 × 8", "54")),        # solve back finds the operation
    ("2x = 11 - 3", "2x = 9", "11 - 3 is 8, not 9", ("11 - 3", "9")),   # one side of an equation
    ("7 × 8", "48", "7 × 8 is 56, not 48", ("7 × 8", "48")),            # a neighbouring times-table entry
])
def test_a_wrong_fact_is_a_slip_with_its_reason_and_no_llm_call(before, after, reason, where):
    d = diagnose(before, after, session=NoLLM())
    assert (d.status, d.misconception, d.topics) == ("clear", SLIP, ("whole_number_arithmetic",))
    slip = next(t for t in d.trace if t["stage"] == "slip")
    assert slip["reason"] == reason and "is" not in slip["said"]   # the student's words leave out the right result
    if where:
        assert (d.located.source, d.located.wrong) == where


@pytest.mark.parametrize("before, after", [
    ("6 + 3", "20"),          # far from 9: a method, not a fact
    ("47 + 38", "84"),        # not a basic fact
    ("23 + 14", "27"),
    ("2x + 3 = 11", "2x = 14"),
    ("1/2 + 1/3", "2/5"),     # a rule explains it
])
def test_not_a_number_fact(before, after):
    assert number_fact(before, after) is None or diagnose(before, after).misconception != SLIP


def test_a_rule_still_wins_and_a_right_step_is_no_fact():
    assert diagnose("1/2 + 1/3", "2/5").misconception == "rule:fractions.add_numerators_denominators"
    assert number_fact("6 + 3", "9") is None


def test_forgetting_the_carry_is_a_rule_now():
    d = diagnose("47 + 38", "75")
    assert (d.status, d.misconception) == ("clear", "rule:buggy_addition.forget_carry")


def test_fact_ids():
    assert fact_id(6, "+", 3) == fact_id(3, "+", 6) == "fact:3+6"
    assert fact_id(8, "*", 7) == "fact:7*8" and fact_id(15, "-", 7) == "fact:15-7"
    assert split_id("fact:3+6") == ("fact", "3+6")
    with pytest.raises(ValueError):
        split_id("fact:6+")
    assert misconception_text("fact:7*8") == "Gets the number fact 7 × 8 wrong, more than once."


def test_the_same_fact_wrong_twice_is_a_weak_fact():
    memory = StudentMemory("s")
    first = planner.raise_fact(memory, diagnose("6 + 3", "8"))
    assert first.misconception == SLIP                                    # once: a slip
    other = planner.raise_fact(memory, diagnose("7 × 8", "54"))
    assert other.misconception == SLIP                                    # another fact: still a slip
    again = planner.raise_fact(memory, diagnose("2x + 3 + 6", "2x + 10"))  # 3 + 6 again, in another answer
    assert again.misconception == "fact:3+6" and fact_of(again) == "fact:3+6"
    assert not planner.is_slip(memory, again.topics, again.misconception)   # even with the topic mastered


def test_a_weak_fact_used_right_counts_against_it():
    memory = StudentMemory("s")
    memory.session_index = 0
    planner.diagnosed(memory, "fact:7*8", ["whole_number_arithmetic"])
    held = memory.p_holds["fact:7*8"]
    planner.answered(memory, make_problem("p", "7 × 8 + 6"), "62", True)
    assert memory.p_holds["fact:7*8"] < held
    assert planner.question_facts("7 × 8 + 6") == frozenset({"fact:7*8"})
