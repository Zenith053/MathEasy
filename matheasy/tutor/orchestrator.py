"""The orchestrator (Role C): a state machine in plain Python.

For each student message it runs, in order:

    read -> check the step -> diagnose -> remember -> probe or hint (or praise) -> critic -> send

Every routing decision and limit is code, so runs are reproducible and every
step is logged. The limits come from the proposal: at most 2 follow-up
questions per problem and at most 2 critic rewrites per hint, after which a
safe template is sent. Nothing reaches the student without passing the critic.

The diagnosis is B's: check_step() first, then diagnose() on a wrong step,
called again with each probe answer as a FollowUp.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from ..diagnosis import FollowUp, ParseError, check_step, diagnose, parse
from ..diagnosis import Diagnosis
from ..llm import BudgetExceeded, LLMCallError, LLMOutputError
from .describe import misconception_text
from .parts import Critic, Hinter, StudentStore
from .state import Hint, HintRequest, MessageKind, Problem, TutorReply, TutorState, Utterance
from .turnlog import TurnLogger

MAX_FOLLOWUPS = 2
MAX_REWRITES = 2

# Pre-approved fallbacks. They never contain an answer, so they skip the critic.
SAFE_TEMPLATE = (
    "Let's slow down and look at this step again. "
    "Can you explain how you got your answer, one step at a time?"
)
CORRECT_FEEDBACK = "That's right, well done."
RETYPE = "I couldn't find an answer in that. Please type just your answer, for example 3/4."

# Errors from LLM-backed parts that end in the safe template, not a crash.
LLM_ERRORS = (BudgetExceeded, LLMCallError, LLMOutputError)

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?")


def extract_answer(text: str) -> Optional[str]:
    """The maths in a student's message: the whole message if it parses, else its last number."""
    stripped = text.strip().rstrip(".!")
    try:
        parse(stripped)
        return stripped
    except ParseError:
        pass
    numbers = _NUMBER.findall(text)
    return numbers[-1].strip() if numbers else None


def with_citation(hint: Hint) -> str:
    """The hint as the student sees it. The citation is added after the critic's leak
    check, so a section number such as 4.5 can't be mistaken for an answer."""
    return f"{hint.text} (See {hint.citations[0]}.)" if hint.citations else hint.text


class Orchestrator:
    def __init__(
        self,
        hinter: Hinter,
        critic: Critic,
        store: StudentStore,
        log: TurnLogger,
        *,
        mode: str = "full",  # "b2" runs B's one-shot diagnosis, for baseline B2
        diagnose_fn: Callable[..., Diagnosis] = diagnose,
    ):
        self.hinter = hinter
        self.critic = critic
        self.store = store
        self.log = log
        self.mode = mode
        self.diagnose_fn = diagnose_fn

    def start(self, state: TutorState, problem: Problem) -> TutorReply:
        """Pose a new problem. Returns the message for the student."""
        state.problem = problem
        state.probe = None
        state.wrong_answer = None
        state.follow_ups = []
        state.followups_asked = 0
        state.hints_given = 0
        state.diagnosis = None
        state.solved = False
        return self._send(state, "question", f"Work out {problem.question}")

    def respond(self, state: TutorState, student_text: str) -> TutorReply:
        """Handle the student's reply to the last message and return the next one."""
        if state.problem is None or state.solved:
            raise RuntimeError("no open problem: call start() first")
        answered_turn = state.turn

        # read
        state.transcript.append(Utterance("student", student_text))
        answer = extract_answer(student_text)
        if answer is None:
            return self._send(state, "question", RETYPE)

        # check the step, or record the probe answer
        if state.probe is None:
            try:
                correct = check_step(state.problem.question, answer)
            except ParseError:
                return self._send(state, "question", RETYPE)
            if correct:
                state.solved = True
                return self._send(state, "feedback", CORRECT_FEEDBACK)
            state.wrong_answer = answer
            state.follow_ups = []
        else:
            state.follow_ups.append(FollowUp(state.probe, answer))
            state.probe = None

        # diagnose
        diagnosis = self._diagnose(state)
        state.diagnosis = diagnosis
        if diagnosis is not None:
            self.log.log(state, "diagnosis", turn=answered_turn, follow_ups=len(state.follow_ups),
                         **diagnosis.to_log())

        # remember (the planner will schedule re-checks from this)
        if diagnosis is not None and diagnosis.misconception:
            self.store.record(state.memory, diagnosis.misconception)

        # probe or hint
        if (diagnosis is not None and diagnosis.status == "ambiguous" and diagnosis.probe is not None
                and state.followups_asked < MAX_FOLLOWUPS):
            state.followups_asked += 1
            state.probe = diagnosis.probe
            reply = self._send(state, "probe", f"Before we go on, what is {diagnosis.probe.question}?")
            if reply.kind == "safe_template":
                state.probe = None  # the probe was never asked
            return reply
        return self._hint(state, diagnosis)

    # --- stages ----------------------------------------------------------------

    def _diagnose(self, state: TutorState) -> Optional[Diagnosis]:
        try:
            return self.diagnose_fn(
                state.problem.question, state.wrong_answer, student_id=state.student_id,
                follow_ups=state.follow_ups, store=self.store, mode=self.mode,
            )
        except LLM_ERRORS as e:
            state.budget_exhausted |= isinstance(e, BudgetExceeded)
            return None
        except ParseError:
            return None

    def _hint(self, state: TutorState, diagnosis: Optional[Diagnosis]) -> TutorReply:
        # Only a clear diagnosis gets a targeted hint. Ambiguous after the
        # follow-up limit, unknown, or no diagnosis at all gets a general one.
        clear = diagnosis is not None and diagnosis.status == "clear"
        located = diagnosis.located if diagnosis is not None else None
        request = HintRequest(
            topic=state.problem.topic,
            question=state.problem.question,
            student_answer=state.wrong_answer,
            misconception=misconception_text(diagnosis.misconception) if clear else None,
            wrong_part=f"{located.source} became {located.wrong}" if located else None,
        )
        state.probe = None
        state.hints_given += 1

        feedback: list[str] = []
        reasons: list[list[str]] = []
        for attempt in range(MAX_REWRITES + 1):
            try:
                hint = self.hinter.hint(request, feedback or None, llm=state.llm)
                verdict = self.critic.review(hint.text, "hint", state.problem, request=request, llm=state.llm)
            except LLM_ERRORS as e:
                return self._fallback(state, e, rewrites=attempt, critic_reasons=reasons)
            if verdict.passed:
                text = with_citation(hint)
                reply = self._deliver(state, "hint", text)
                self.log.log(state, "hint", turn=state.turn, text=text, citations=list(hint.citations),
                             rewrites=attempt, safe_template=False, critic_reasons=reasons)
                return reply
            feedback = list(verdict.reasons)
            reasons.append(feedback)
        return self._fallback(state, None, rewrites=MAX_REWRITES, critic_reasons=reasons)

    def _send(self, state: TutorState, kind: MessageKind, text: str) -> TutorReply:
        """Send a message that is not rewritten: question, probe or feedback."""
        try:
            verdict = self.critic.review(text, kind, state.problem, llm=state.llm)
        except LLM_ERRORS as e:
            return self._fallback(state, e)
        if not verdict.passed:
            return self._fallback(state, None)
        return self._deliver(state, kind, text)

    def _fallback(self, state: TutorState, error: Optional[Exception], *, rewrites: int = 0,
                  critic_reasons: Optional[list] = None) -> TutorReply:
        if isinstance(error, BudgetExceeded):
            state.budget_exhausted = True
        reply = self._deliver(state, "safe_template", SAFE_TEMPLATE)
        self.log.log(state, "hint", turn=state.turn, text=SAFE_TEMPLATE, citations=[], rewrites=rewrites,
                     safe_template=True, critic_reasons=critic_reasons or [],
                     error=f"{type(error).__name__}: {error}" if error else None)
        return reply

    def _deliver(self, state: TutorState, kind: MessageKind, text: str) -> TutorReply:
        """The only place a message reaches the student. Every caller has passed the critic."""
        state.turn += 1
        state.transcript.append(Utterance("tutor", text, kind))
        item = None if state.solved else state.item  # nothing to answer after praise
        return TutorReply(kind=kind, text=text, item=item, critic_passed=True, done=state.solved)
