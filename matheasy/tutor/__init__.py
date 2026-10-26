"""The tutor (Role C): shared state, orchestrator, run log and session runner."""

from .orchestrator import MAX_FOLLOWUPS, MAX_REWRITES, SAFE_TEMPLATE, Orchestrator, extract_answer
from .session import SessionResult, make_problem, run_session
from .state import Hint, HintRequest, Problem, StudentMemory, TutorReply, TutorState, Utterance, Verdict
from .turnlog import TurnLogger

__all__ = [
    "Hint", "HintRequest", "MAX_FOLLOWUPS", "MAX_REWRITES", "Orchestrator", "Problem", "SAFE_TEMPLATE",
    "SessionResult", "StudentMemory", "TurnLogger", "TutorReply", "TutorState", "Utterance", "Verdict",
    "extract_answer", "make_problem", "run_session",
]
