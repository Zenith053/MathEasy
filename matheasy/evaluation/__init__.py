"""Evaluation (owner: D): simulated students, log fields and their checker, test items, metrics.

Shared functions other roles call:
    SimStudent, roster, LLMFeedbackJudge     simulated students          (C's runner)
    EVENTS, check_files                      required log fields         (C's log format)
    metrics.compute                          metrics from run logs       (everyone)
"""

from .items import item_key, overlapping
from .log_fields import EVENTS, check_files, check_record, check_records, read_jsonl
from .students import (
    GROUPS, SLIP_ONLY_RATE, LLMFeedbackJudge, SimStudent, StudentReply, correct_answer, roster, rule_answers, slips,
    student_group,
)

__all__ = [
    "EVENTS", "GROUPS", "LLMFeedbackJudge", "SLIP_ONLY_RATE", "SimStudent", "StudentReply", "check_files",
    "check_record", "check_records", "correct_answer", "item_key", "overlapping", "read_jsonl", "roster",
    "rule_answers", "slips", "student_group",
]
