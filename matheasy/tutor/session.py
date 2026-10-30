"""Run one session (Role C): pre-test, tutoring on each problem, post-test.

A session is one sitting of one student: it has one token budget and a
session_index, so retention can be measured across a student's sessions. This
is the core of the experiment runner; the week-3 runner loops it over
students, conditions and shards.

The student is D's SimStudent: it gets each tutor message plus the maths it is
asked (`item`), and its reply carries the ground truth logged in `turn`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..diagnosis import ParseError, equivalent, parse, topic_of
from ..evaluation import SimStudent, correct_answer
from .parts import Tutor
from .state import Problem, TutorState

MAX_TURNS_PER_PROBLEM = 8


@dataclass(frozen=True)
class SessionResult:
    state: TutorState
    ended_by: str  # "solved", "max_turns" or "budget"
    pre_correct: int
    post_correct: int


def make_problem(pid: str, question: str) -> Problem:
    return Problem(id=pid, topic=topic_of(parse(question)), question=question,
                   correct_answer=correct_answer(question))


def run_session(
    tutor: Tutor,
    student: SimStudent,
    problems: Sequence[Problem],
    *,
    session_id: str,
    condition: str,
    session_index: int,
    pre_items: Sequence[str] = (),
    post_items: Sequence[str] = (),
    llm_session=None,  # the tutor's matheasy.llm Session, for tokens_used
    max_turns_per_problem: int = MAX_TURNS_PER_PROBLEM,
) -> SessionResult:
    log = tutor.log
    state = TutorState(session_id=session_id, condition=condition, student_id=student.student_id,
                       memory=tutor.store.load(student.student_id), llm=llm_session)
    # D's student needs this before every session: from the second one on, a fixed
    # misconception may come back (relapse_rate).
    relapsed = student.begin_session(session_index)
    log.log(state, "session_start", session_index=session_index, student_group=student.group,
            planted=[student.misconception_id] if student.misconception_id else [], attack=None,
            relapsed=relapsed)

    pre_correct = _test(log, state, student, "pre", pre_items)

    ended_by = "solved"
    for problem in problems:
        reply = tutor.start(state, problem)
        tutor_turns = 0
        while True:
            answer = student.respond(reply.text, item=reply.item)
            log.log(state, "turn", turn=state.turn, tutor_text=reply.text, critic_passed=reply.critic_passed,
                    item=reply.item, correct_answer=answer.correct_answer, student_text=answer.text,
                    student_answer=answer.answer, student_correct=answer.correct,
                    followed_rule=answer.followed_rule, holds_misconception=answer.holds_misconception,
                    tutor_kind=reply.kind)
            if reply.done or state.budget_exhausted:
                break
            if tutor_turns >= max_turns_per_problem:
                ended_by = "max_turns"
                break
            reply = tutor.respond(state, answer.text)
            tutor_turns += 1
        if state.budget_exhausted:
            ended_by = "budget"
            break

    post_correct = _test(log, state, student, "post", post_items)
    log.log(state, "session_end", turns=state.turn, ended_by=ended_by,
            tokens_used=llm_session.used_tokens if llm_session is not None else 0,
            holds=[student.misconception_id] if student.holds_misconception else [])
    return SessionResult(state, ended_by, pre_correct, post_correct)


def _test(log, state: TutorState, student: SimStudent, phase: str, items: Sequence[str]) -> int:
    """Ask each test item outside the tutor and log whether the student got it right."""
    score = 0
    for item in items:
        reply = student.respond(f"Test question: work out {item}", item=item)
        right = correct_answer(item)
        try:
            correct = reply.answer is not None and equivalent(reply.answer, right)
        except ParseError:
            correct = False
        score += correct
        log.log(state, "test", phase=phase, item=item, correct_answer=right,
                student_answer=reply.answer or "", correct=correct)
    return score
