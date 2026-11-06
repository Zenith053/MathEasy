"""The orchestrator (Role C): a state machine in plain Python.

For each student message it runs, in order:

    read -> check the step -> diagnose -> remember -> probe, or analysis and hint (or praise) -> critic -> send

A wrong answer is answered LeetCode style (docs/design_decisions.md, "What we are
trying to build"): where it went wrong, what the mistake is, its topics, the
hinter's hint, then similar problems to practise (analysis.py).

Read is the same in every condition (read_answer(), also used by B0 and B1):
code finds the maths in the message, then B's transcribe() reads what the
student meant. A line in standard notation needs no model call; an ambiguous
one is asked about once ("Did you mean 1) ... or 2) ...?").

Every routing decision and limit is code, so runs are reproducible and every
step is logged. The limits come from the proposal: at most 2 follow-up
questions per problem and at most 2 critic rewrites per hint, after which a
safe template is sent. Nothing reaches the student without passing the critic.

The diagnosis is B's: check_step() first, then diagnose() on a wrong step,
called again with each probe answer as a FollowUp. With an LLM session,
diagnose() reads a step no rule explains with its LLM fallback; a verified
reading of a misconception is the mistake shown and what the hinter aims at.
"""

from __future__ import annotations

from typing import Callable, Optional

from ..diagnosis import (
    FollowUp, ParseError, Transcription, check_step, diagnose, equivalent, parse, resolve, transcribe,
)
from ..diagnosis import Diagnosis
from ..evaluation import correct_answer
from ..llm import BudgetExceeded, LLMCallError, LLMOutputError
from .analysis import analyse, mistake
from .mathtext import EDGE_WORDS, maths_runs
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

def extract_answer(text: str) -> Optional[str]:
    """The maths in a student's message: the whole message if it parses, else the last
    piece of maths in it that parses ("I got x = 4." -> "x = 4")."""
    stripped = EDGE_WORDS.sub("", text.strip().rstrip(".!?"))
    try:
        parse(stripped)
        return stripped
    except ParseError:
        pass
    for candidate in reversed(maths_runs(text)):
        try:
            parse(candidate)
            return candidate
        except ParseError:
            continue
    return None


def read_answer(text: str, question: str, session=None) -> Transcription:
    """The Read stage, the same for every condition. Code finds the maths in the message
    and B's transcribe() reads what the student meant (an LLM only for lines not in
    standard notation). Without an LLM session: the parser's literal reading."""
    answer = extract_answer(text)
    if session is None:
        if answer is None:
            return Transcription("unreadable", None, text, used_llm=False, reason="no maths found")
        return Transcription("ok", answer, text, used_llm=False)
    return transcribe(answer if answer is not None else text, question=question, session=session)


def clarify(choices: tuple[str, ...]) -> str:
    if len(choices) == 1:
        return f"Did you mean {choices[0]}? Reply yes, or type your answer again."
    listed = " or ".join(f"{i}) {c}" for i, c in enumerate(choices, 1))
    return f"Did you mean {listed}? Reply with the number."


def _open(state: TutorState) -> dict[str, str]:
    """The session's other open items and their answers: no message may give these away either."""
    return {q: a for q, a in state.unsolved.items() if a and q != state.problem.question}


def _answer(item: str) -> Optional[str]:
    try:
        return correct_answer(item)
    except Exception:  # D's answer key can't work it out
        return None


def _same(a: str, b: Optional[str]) -> bool:
    try:
        return b is not None and equivalent(a, b)
    except (ParseError, ValueError, TypeError):
        return False


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
        state.clarifying = None
        state.unsolved[problem.question] = problem.correct_answer
        return self._send(state, "question", f"Work out {problem.question}")

    def respond(self, state: TutorState, student_text: str) -> TutorReply:
        """Handle the student's reply to the last message and return the next one."""
        if state.problem is None or state.solved:
            raise RuntimeError("no open problem: call start() first")
        answered_turn = state.turn

        # read
        try:
            if state.clarifying is not None:  # asked once; resolve() reads the reply in code
                reading, state.clarifying = resolve(state.clarifying, student_text), None
            else:
                reading = read_answer(student_text, state.item, state.llm)
        except LLM_ERRORS as e:
            state.transcript.append(Utterance("student", student_text))
            return self._fallback(state, e)
        state.transcript.append(Utterance("student", student_text, reading=reading))
        if reading.status == "unsure" and reading.choices:
            state.clarifying = reading
            return self._send(state, "question", clarify(reading.choices))
        if reading.status != "ok":
            return self._send(state, "question", RETYPE)
        answer = reading.line

        # check the step, or record the probe answer
        fresh = state.probe is None  # an answer to the problem itself
        if fresh:
            try:
                correct = check_step(state.problem.question, answer)
            except ParseError:
                return self._send(state, "question", RETYPE)
            if correct:
                state.unsolved.pop(state.problem.question, None)
                self.store.observe(state.memory, state.problem, answer, True)
                state.solved = True
                return self._send(state, "feedback", CORRECT_FEEDBACK)
            if not _same(answer, state.wrong_answer):
                state.follow_ups = []  # a new wrong step; the same one again keeps its probe answers
            state.wrong_answer = answer
        else:
            state.follow_ups.append(FollowUp(state.probe, answer))
            if _same(answer, state.unsolved.get(state.probe.question)):
                state.unsolved.pop(state.probe.question)
            state.probe = None

        # diagnose
        diagnosis = self._diagnose(state)
        state.diagnosis = diagnosis
        if diagnosis is not None:
            self.log.log(state, "diagnosis", turn=answered_turn, follow_ups=len(state.follow_ups),
                         read=reading.to_log() if reading.used_llm else None, **diagnosis.to_log())

        # remember: the planner updates the student model and schedules re-checks from it
        if fresh:
            self.store.observe(state.memory, state.problem, answer, False,
                               topics=diagnosis.topics if diagnosis is not None else ())
        if diagnosis is not None and diagnosis.misconception:
            self.store.record(state.memory, diagnosis.misconception)

        # probe or hint
        if (diagnosis is not None and diagnosis.status == "ambiguous" and diagnosis.probe is not None
                and state.followups_asked < MAX_FOLLOWUPS):
            state.followups_asked += 1
            state.probe = diagnosis.probe
            state.unsolved.setdefault(diagnosis.probe.question, _answer(diagnosis.probe.question))
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
                question=state.problem.question, session=state.llm,
            )
        except LLM_ERRORS as e:
            state.budget_exhausted |= isinstance(e, BudgetExceeded)
            return None
        except ParseError:
            return None

    def _hint(self, state: TutorState, diagnosis: Optional[Diagnosis]) -> TutorReply:
        # A clear diagnosis, or a verified LLM reading of an unknown step, gets a targeted
        # hint. Ambiguous after the follow-up limit, unexplained, or no diagnosis gets a general one.
        located = diagnosis.located if diagnosis is not None else None
        analysis = analyse(diagnosis, state.problem, state.memory.worked, _open(state))
        request = HintRequest(
            topic=state.problem.topic,
            question=state.problem.question,
            student_answer=state.wrong_answer,
            misconception=mistake(diagnosis, state.problem) if diagnosis is not None else None,
            wrong_part=f"{located.source} became {located.wrong}" if located else None,
        )
        state.probe = None
        state.hints_given += 1

        feedback: list[str] = []
        reasons: list[list[str]] = []
        for attempt in range(MAX_REWRITES + 1):
            try:
                hint = self.hinter.hint(request, feedback or None, llm=state.llm)
                verdict = self.critic.review(hint.text, "hint", state.problem, request=request, llm=state.llm,
                                             unsolved=_open(state))
            except LLM_ERRORS as e:
                return self._fallback(state, e, rewrites=attempt, critic_reasons=reasons)
            if verdict.passed:
                # Every part of the analysis passed the leak check when it was built.
                text = "\n".join(part for part in (analysis.text(), with_citation(hint), analysis.practice_text())
                                  if part)
                reply = self._deliver(state, "hint", text)
                self.log.log(state, "hint", turn=state.turn, text=text, citations=list(hint.citations),
                             rewrites=attempt, safe_template=False, critic_reasons=reasons,
                             analysis=analysis.to_log())
                return reply
            feedback = list(verdict.reasons)
            reasons.append(feedback)
        return self._fallback(state, None, rewrites=MAX_REWRITES, critic_reasons=reasons)

    def _send(self, state: TutorState, kind: MessageKind, text: str) -> TutorReply:
        """Send a message that is not rewritten: question, probe or feedback."""
        try:
            verdict = self.critic.review(text, kind, state.problem, llm=state.llm, unsolved=_open(state))
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
