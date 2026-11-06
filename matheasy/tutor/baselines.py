"""Baselines B0 and B1 (Role C).

    B0Tutor   one prompted Socratic tutor, with no tools, retrieval, memory or critic
    B1Agent   one agent with the full tutor's tools and retrieval, but no hypothesis
              testing, no persistent memory and no critic

B0:

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

Both read the student's input like the full tutor (read_answer(), B's
transcriber): where it reads a line differently from how it was typed, the
prompt shows the reading next to the student's words.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from ..diagnosis import rule_id
from ..diagnosis.rules import get_rule
from ..llm import BudgetExceeded, LLMCallError, LLMOutputError
from .orchestrator import SAFE_TEMPLATE, read_answer, with_citation
from .state import Hint, MessageKind, Problem, TutorReply, TutorState, Utterance
from .store import NoMemoryStore
from .tools import ToolBox
from .turnlog import TurnLogger

LLM_ERRORS = (BudgetExceeded, LLMCallError, LLMOutputError)

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
        failed = self._read(state, student_text)
        if failed:
            return failed
        try:
            result = state.llm.call(self.role, [{"role": "user", "content": self._prompt(state)}],
                                    system=SYSTEM, json_schema=SCHEMA)
            message = str(result.parsed["message"]).strip()
            correct = bool(result.parsed["student_is_correct"])
            stated = str(result.parsed.get("student_misconception") or "").strip() or None
        except (*LLM_ERRORS, KeyError) as e:
            return self._fallback(state, e)

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
            lines.append(f"{'Tutor' if u.speaker == 'tutor' else 'Student'}: {u.text}{_reading(u)}")
        lines.append("\nWrite the tutor's next message.")
        return "\n".join(lines)

    def _read(self, state: TutorState, student_text: str) -> Optional[TutorReply]:
        """Add the student's message to the transcript with its reading; on an LLM error,
        the safe template."""
        try:
            reading = read_answer(student_text, state.problem.question, state.llm)
        except LLM_ERRORS as e:
            state.transcript.append(Utterance("student", student_text))
            return self._fallback(state, e)
        state.transcript.append(Utterance("student", student_text, reading=reading))
        return None

    def _fallback(self, state: TutorState, error: Exception) -> TutorReply:
        state.budget_exhausted |= isinstance(error, BudgetExceeded)
        reply = self._deliver(state, "safe_template", SAFE_TEMPLATE)
        self._log_hint(state, SAFE_TEMPLATE, safe_template=True, error=f"{type(error).__name__}: {error}")
        return reply

    def _log_hint(self, state: TutorState, text: str, *, safe_template: bool, error: Optional[str] = None,
                  citations: tuple[str, ...] = ()) -> None:
        self.log.log(state, "hint", turn=state.turn, text=text, citations=list(citations), rewrites=0,
                     safe_template=safe_template, critic_reasons=[], error=error)

    def _deliver(self, state: TutorState, kind: MessageKind, text: str) -> TutorReply:
        state.turn += 1
        state.transcript.append(Utterance("tutor", text, kind))
        item = None if state.solved else state.problem.question
        # No critic in B0: critic_passed is None, as D's log fields allow for b0.
        return TutorReply(kind=kind, text=text, item=item, critic_passed=None, done=state.solved)


# --- B1 ------------------------------------------------------------------------------

B1_SYSTEM = """\
You are a maths tutor for a student aged 11 to 14 who is working on one question. You work \
alone, with tools for exact maths, for misconception rules and, if listed, for searching a textbook.

Each reply does exactly one thing, as JSON:
- Call a tool: action "tool", tool its name, and args its arguments as a JSON object written as \
a string, e.g. "{{\\"before\\": \\"1/2 + 1/3\\", \\"after\\": \\"2/5\\"}}". You get its result back.
- Message the student: action "message", with message and the record fields below.

Help the student reach the answer themselves: point them at what to check and ask guiding \
questions. Never tell them the final answer. Keep each message to at most three short sentences. \
Check the maths with the tools rather than working it out in your head.

With each message you also report, for the teacher's records only (the student never sees these):
- student_is_correct: whether the student's latest answer to the question is correct.
- misconception_rule: the name of the rule (from list_rules) that explains their mistake, or an \
empty string if none fits, their answer is correct, or you cannot tell.
- student_misconception: in one sentence, the misconception you think explains their mistake, or \
an empty string.
- citation: the reference of a textbook section you used for this message, exactly as the search \
returned it, or an empty string.

When the student's answer is correct, your message should simply confirm it.

Tools (arguments as JSON schema):
{tools}"""

B1_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["tool", "message"]},
        "tool": {"type": "string"},
        "args": {"type": "string"},
        "message": {"type": "string"},
        "student_is_correct": {"type": "boolean"},
        "misconception_rule": {"type": "string"},
        "student_misconception": {"type": "string"},
        "citation": {"type": "string"},
    },
    "required": ["action", "tool", "args", "message", "student_is_correct", "misconception_rule",
                 "student_misconception", "citation"],
}

MAX_TOOL_CALLS = 4  # per student message; then the agent must write its message
LAST_CALL = "That was your last tool call for this message: write your message to the student now."


class B1Agent(B0Tutor):
    """One LLM agent that calls the tools itself, through the same ToolBox the
    tool-call metric reads. A call that repeats a tool which just failed is logged
    as a retry. It names a misconception by rule name, so its diagnoses can be
    scored like the full tutor's; a name that is not one of B's rules counts as unknown."""

    role = "b1_agent"

    def __init__(self, log: TurnLogger, tools: ToolBox):
        super().__init__(log)
        self.tools = tools
        self.system = B1_SYSTEM.format(tools=json.dumps(tools.declarations(), indent=1))

    def respond(self, state: TutorState, student_text: str) -> TutorReply:
        if state.problem is None or state.solved:
            raise RuntimeError("no open problem: call start() first")
        answered_turn = state.turn
        failed = self._read(state, student_text)
        if failed:
            return failed
        messages = [{"role": "user", "content": self._prompt(state)}]
        refs: set[str] = set()  # sections the agent retrieved for this message
        failed: Optional[str] = None  # the tool whose last call failed
        for calls in range(MAX_TOOL_CALLS + 1):
            try:
                result = state.llm.call(self.role, messages, system=self.system, json_schema=B1_SCHEMA)
                out = result.parsed
                action = out["action"]
            except (*LLM_ERRORS, KeyError, TypeError) as e:
                return self._fallback(state, e)
            if action != "tool":
                break
            if calls == MAX_TOOL_CALLS:
                return self._fallback(state, LLMOutputError("B1 called a tool after its last tool call"))
            name = str(out.get("tool", ""))
            done = self.tools.call(state, name, _arguments(out.get("args")), is_retry=name == failed)
            failed = None if done.ok else name
            if done.ok and name == "retrieve":
                refs |= {p["ref"] for p in done.value}
            feedback = json.dumps({"result": done.value} if done.ok else {"error": done.error}, default=str)
            messages += [{"role": "assistant", "content": result.text},
                         {"role": "user", "content": f"Tool result: {feedback}"
                                                     + (f"\n{LAST_CALL}" if calls == MAX_TOOL_CALLS - 1 else "")}]

        message = str(out.get("message", "")).strip()
        if not message:
            return self._fallback(state, LLMOutputError("B1 sent an empty message"))
        correct = bool(out.get("student_is_correct"))
        if not correct:
            misconception = _rule(out.get("misconception_rule"))
            stated = str(out.get("student_misconception") or "").strip() or None
            self.log.log(state, "diagnosis", turn=answered_turn, status="clear" if misconception else "unknown",
                         misconception=misconception, beliefs={misconception: 1.0} if misconception else {},
                         follow_ups=0, mode="b1", stated_misconception=stated,
                         stated_rule=str(out.get("misconception_rule") or "") or None)
            state.hints_given += 1
        citation = str(out.get("citation") or "")
        hint = Hint(message, (citation,) if citation in refs else ())
        state.solved = correct
        text = message if correct else with_citation(hint)
        reply = self._deliver(state, "feedback" if correct else "hint", text)
        if not correct:
            self._log_hint(state, text, safe_template=False, citations=hint.citations)
        return reply


def _reading(u: Utterance) -> str:
    """The transcriber's reading, when the model was needed to read the line."""
    r = u.reading
    if r is None or not r.used_llm:
        return ""
    if r.status == "ok":
        return f" (read as: {r.line})"
    if r.status == "unsure" and r.choices:
        return f" (could mean: {' or '.join(r.choices)})"
    return ""


def _arguments(args: Any) -> Any:
    """The agent's arguments, parsed from their JSON string; left as they are if they
    don't parse, so the ToolBox logs the call as invalid."""
    try:
        return json.loads(args) if isinstance(args, str) else args
    except json.JSONDecodeError:
        return args


def _rule(name: Any) -> Optional[str]:
    """The misconception ID for a rule name the agent gave, or None if it isn't one of B's rules."""
    try:
        return rule_id(get_rule(str(name).strip()).name) if name else None
    except (KeyError, ValueError):
        return None
