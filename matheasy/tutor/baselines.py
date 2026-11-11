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

Both are LangGraph state graphs (docs/design_decisions.md, T4 and T5), one turn per
invoke():

    B0   read -> tutor -> deliver
    B1   read -> agent -> tool -> agent ... -> deliver   (the edge after agent follows its action)

Only the control flow is LangGraph's. Every model call still goes through the
tutor's LLMClient session, so the token budget, the cache and the run logs are the
same as in every other condition, and every tool call goes through the ToolBox,
which the tool-call metric reads. LangChain's own model wrappers are not used.
The full tutor stays plain Python (orchestrator.py).
"""

from __future__ import annotations

import json
from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

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


class Turn(TypedDict, total=False):
    """One student message going through a baseline's graph."""

    state: TutorState
    student_text: str
    answered_turn: int  # the turn the student answered, for the diagnosis record
    parsed: dict  # the model's last reply
    error: Exception  # set when the turn must end in the safe template
    reply: TutorReply  # set by the last node
    # B1 only
    messages: list[dict]  # the agent's conversation for this turn
    text: str  # the model's last reply, as text, to send back to it
    calls: int  # tool calls made this turn
    refs: frozenset[str]  # textbook sections the agent retrieved this turn
    failed: Optional[str]  # the tool whose last call failed


class B0Tutor:
    role = "b0_tutor"

    def __init__(self, log: TurnLogger):
        self.log = log
        self.store = NoMemoryStore()
        self.graph = self._graph()

    def _graph(self):
        g = StateGraph(Turn)
        g.add_node("read", self._read_node)
        g.add_node("tutor", self._tutor_node)
        g.add_node("deliver", self._deliver_node)
        g.add_edge(START, "read")
        g.add_conditional_edges("read", _replied, ["tutor", END])
        g.add_edge("tutor", "deliver")
        g.add_edge("deliver", END)
        return g.compile()

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
        return self.graph.invoke({"state": state, "student_text": student_text, "answered_turn": state.turn})["reply"]

    # --- nodes -----------------------------------------------------------------------

    def _read_node(self, turn: Turn) -> dict:
        failed = self._read(turn["state"], turn["student_text"])
        return {"reply": failed} if failed else {}

    def _tutor_node(self, turn: Turn) -> dict:
        state = turn["state"]
        try:
            result = state.llm.call(self.role, [{"role": "user", "content": self._prompt(state)}],
                                    system=SYSTEM, json_schema=SCHEMA)
        except LLM_ERRORS as e:
            return {"error": e}
        missing = [k for k in ("message", "student_is_correct") if k not in (result.parsed or {})]
        if missing:
            return {"error": KeyError(f"B0's reply has no {', '.join(missing)}")}
        return {"parsed": result.parsed}

    def _deliver_node(self, turn: Turn) -> dict:
        state, out = turn["state"], turn.get("parsed") or {}
        if turn.get("error") is not None:
            return {"reply": self._fallback(state, turn["error"])}
        message = str(out["message"]).strip()
        correct = bool(out["student_is_correct"])
        stated = str(out.get("student_misconception") or "").strip() or None
        if not correct:
            self.log.log(state, "diagnosis", turn=turn["answered_turn"], status="unknown", misconception=None,
                         beliefs={}, follow_ups=0, mode="b0", stated_misconception=stated)
            state.hints_given += 1
        state.solved = correct
        reply = self._deliver(state, "feedback" if correct else "hint", message)
        if not correct:
            self._log_hint(state, message, safe_template=False)
        return {"reply": reply}

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
        self.tools = tools
        self.system = B1_SYSTEM.format(tools=json.dumps(tools.declarations(), indent=1))
        super().__init__(log)

    def _graph(self):
        g = StateGraph(Turn)
        g.add_node("read", self._read_node)
        g.add_node("agent", self._agent_node)
        g.add_node("tool", self._tool_node)
        g.add_node("deliver", self._deliver_node)
        g.add_edge(START, "read")
        g.add_conditional_edges("read", _replied, ["agent", END])
        g.add_conditional_edges("agent", _next_after_agent, ["tool", "deliver"])
        g.add_edge("tool", "agent")
        g.add_edge("deliver", END)
        return g.compile()

    # --- nodes -----------------------------------------------------------------------

    def _read_node(self, turn: Turn) -> dict:
        failed = self._read(turn["state"], turn["student_text"])
        if failed:
            return {"reply": failed}
        return {"messages": [{"role": "user", "content": self._prompt(turn["state"])}], "calls": 0,
                "refs": frozenset(), "failed": None}

    def _agent_node(self, turn: Turn) -> dict:
        try:
            result = turn["state"].llm.call(self.role, turn["messages"], system=self.system, json_schema=B1_SCHEMA)
            action = result.parsed["action"]
        except (*LLM_ERRORS, KeyError, TypeError) as e:
            return {"error": e}
        if action == "tool" and turn["calls"] == MAX_TOOL_CALLS:
            return {"error": LLMOutputError("B1 called a tool after its last tool call")}
        return {"parsed": result.parsed, "text": result.text}

    def _tool_node(self, turn: Turn) -> dict:
        out, calls = turn["parsed"], turn["calls"]
        name = str(out.get("tool", ""))
        done = self.tools.call(turn["state"], name, _arguments(out.get("args")), is_retry=name == turn["failed"])
        refs = turn["refs"] | ({p["ref"] for p in done.value} if done.ok and name == "retrieve" else set())
        feedback = json.dumps({"result": done.value} if done.ok else {"error": done.error}, default=str)
        last = f"\n{LAST_CALL}" if calls == MAX_TOOL_CALLS - 1 else ""
        messages = [*turn["messages"], {"role": "assistant", "content": turn["text"]},
                    {"role": "user", "content": f"Tool result: {feedback}{last}"}]
        return {"messages": messages, "calls": calls + 1, "refs": frozenset(refs),
                "failed": None if done.ok else name}

    def _deliver_node(self, turn: Turn) -> dict:
        state, out = turn["state"], turn.get("parsed") or {}
        answered_turn, refs = turn["answered_turn"], turn.get("refs", frozenset())
        if turn.get("error") is not None:
            return {"reply": self._fallback(state, turn["error"])}
        message = str(out.get("message", "")).strip()
        if not message:
            return {"reply": self._fallback(state, LLMOutputError("B1 sent an empty message"))}
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
        return {"reply": reply}


def _replied(turn: Turn) -> str:
    """After reading: END if reading already ended the turn (the safe template)."""
    return END if turn.get("reply") is not None else ("agent" if "messages" in turn else "tutor")


def _next_after_agent(turn: Turn) -> str:
    """The agent asked for a tool: run it. Anything else (a message, an error): deliver."""
    if turn.get("error") is None and turn["parsed"].get("action") == "tool":
        return "tool"
    return "deliver"


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
