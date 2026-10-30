"""Baseline B0 (Role C): one prompted Socratic tutor, with no tools, retrieval, memory or critic.

It is the plain-LLM starting point the full tutor is compared against, on the
same backbone model and the same per-session token budget. It plugs into
run_session() exactly like the Orchestrator.

Each turn the model sees the question and the conversation so far, and returns
its next message plus two things for the run log: whether it thinks the
student's last answer is correct (which also ends the problem), and what it
thinks the student's misconception is. It never gets the correct answer: like
any plain LLM tutor, it must work the maths out itself.

B0's diagnosis is free text, so its `diagnosis` records have status "unknown"
and the text in `stated_misconception`; scoring them for H1 needs a mapping
from text to misconception IDs (to agree with D).
"""

from __future__ import annotations

from typing import Optional

from ..llm import BudgetExceeded, LLMCallError, LLMOutputError
from .orchestrator import SAFE_TEMPLATE
from .state import MessageKind, Problem, TutorReply, TutorState, Utterance
from .store import NoMemoryStore
from .turnlog import TurnLogger

SYSTEM = """\
You are a Socratic maths tutor for a student aged 11 to 14 who is working on one question.

Help the student reach the answer themselves: point them at what to check and ask guiding \
questions. Never tell them the final answer. Keep each message to at most three short sentences.

Each turn you also report, for the teacher's records only (the student never sees these):
- student_is_correct: whether the student's latest answer to the question is correct.
- student_misconception: in one sentence, the misconception you think explains their mistake, \
or an empty string if their answer is correct or you cannot tell.

When the student's answer is correct, your message should simply confirm it."""

SCHEMA = {
    "type": "object",
    "properties": {
        "message": {"type": "string"},
        "student_is_correct": {"type": "boolean"},
        "student_misconception": {"type": "string"},
    },
    "required": ["message", "student_is_correct", "student_misconception"],
}


class B0Tutor:
    role = "b0_tutor"

    def __init__(self, log: TurnLogger):
        self.log = log
        self.store = NoMemoryStore()

    def start(self, state: TutorState, problem: Problem) -> TutorReply:
        if state.llm is None:
            raise ValueError("B0 needs the tutor's LLM session: pass llm_session to run_session()")
        state.problem = problem
        state.solved = False
        state.hints_given = 0
        state.problem_start = len(state.transcript)
        return self._deliver(state, "question", f"Work out {problem.question}")

    def respond(self, state: TutorState, student_text: str) -> TutorReply:
        if state.problem is None or state.solved:
            raise RuntimeError("no open problem: call start() first")
        answered_turn = state.turn
        state.transcript.append(Utterance("student", student_text))
        try:
            result = state.llm.call(self.role, [{"role": "user", "content": self._prompt(state)}],
                                    system=SYSTEM, json_schema=SCHEMA)
            message = str(result.parsed["message"]).strip()
            correct = bool(result.parsed["student_is_correct"])
            stated = str(result.parsed.get("student_misconception") or "").strip() or None
        except (BudgetExceeded, LLMCallError, LLMOutputError, KeyError) as e:
            state.budget_exhausted |= isinstance(e, BudgetExceeded)
            reply = self._deliver(state, "safe_template", SAFE_TEMPLATE)
            self._log_hint(state, SAFE_TEMPLATE, safe_template=True, error=f"{type(e).__name__}: {e}")
            return reply

        if not correct:
            self.log.log(state, "diagnosis", turn=answered_turn, status="unknown", misconception=None,
                         beliefs={}, follow_ups=0, mode="b0", stated_misconception=stated)
            state.hints_given += 1
        state.solved = correct
        reply = self._deliver(state, "feedback" if correct else "hint", message)
        if not correct:
            self._log_hint(state, message, safe_template=False)
        return reply

    def _prompt(self, state: TutorState) -> str:
        lines = [f"Question: {state.problem.question}", "", "Conversation so far:"]
        for u in state.transcript[state.problem_start:]:
            lines.append(f"{'Tutor' if u.speaker == 'tutor' else 'Student'}: {u.text}")
        lines.append("\nWrite the tutor's next message.")
        return "\n".join(lines)

    def _log_hint(self, state: TutorState, text: str, *, safe_template: bool, error: Optional[str] = None) -> None:
        self.log.log(state, "hint", turn=state.turn, text=text, citations=[], rewrites=0,
                     safe_template=safe_template, critic_reasons=[], error=error)

    def _deliver(self, state: TutorState, kind: MessageKind, text: str) -> TutorReply:
        state.turn += 1
        state.transcript.append(Utterance("tutor", text, kind))
        item = None if state.solved else state.problem.question
        # No critic in B0: critic_passed is None, as D's log fields allow for b0.
        return TutorReply(kind=kind, text=text, item=item, critic_passed=None, done=state.solved)
