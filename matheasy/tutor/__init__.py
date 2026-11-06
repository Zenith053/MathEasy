"""The tutor (Role C): shared state, orchestrator, run log and session runner."""

from .baselines import B0Tutor, B1Agent
from .critic import TutorCritic
from .hinter import LLMHinter
from .orchestrator import MAX_FOLLOWUPS, MAX_REWRITES, SAFE_TEMPLATE, Orchestrator, extract_answer
from .store import InMemoryStore, JsonStudentStore, NoMemoryStore
from .tools import Tool, ToolBox, ToolResult, standard_tools
from .session import SessionResult, make_problem, run_session
from .state import Hint, HintRequest, Problem, StudentMemory, TutorReply, TutorState, Utterance, Verdict
from .turnlog import TurnLogger

__all__ = [
    "B0Tutor", "B1Agent", "InMemoryStore", "JsonStudentStore", "LLMHinter", "NoMemoryStore", "Tool", "ToolBox",
    "ToolResult", "TutorCritic", "standard_tools",
    "Hint", "HintRequest", "MAX_FOLLOWUPS", "MAX_REWRITES", "Orchestrator", "Problem", "SAFE_TEMPLATE",
    "SessionResult", "StudentMemory", "TurnLogger", "TutorReply", "TutorState", "Utterance", "Verdict",
    "extract_answer", "make_problem", "run_session",
]
