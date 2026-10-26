"""Evaluation (owner: D): simulated students, log fields and their checker, test items.

Shared functions other roles call:
    SimStudent, LLMFeedbackJudge     simulated students          (C's runner)
    EVENTS, check_files              required log fields         (C's log format)
"""

from .items import item_key, overlapping
from .log_fields import EVENTS, check_files, check_record, check_records
from .students import GROUPS, LLMFeedbackJudge, SimStudent, StudentReply, correct_answer, student_group

__all__ = [
    "EVENTS", "GROUPS", "LLMFeedbackJudge", "SimStudent", "StudentReply", "check_files", "check_record",
    "check_records", "correct_answer", "item_key", "overlapping", "student_group",
]
