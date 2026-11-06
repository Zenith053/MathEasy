"""Evaluation (owner: D): simulated and adversarial students, pre/post tests, log fields and
their checker, metrics, statistics, judges and labelling.

Shared functions other roles call:
    SimStudent, roster, LLMFeedbackJudge        simulated students                 (C's runner)
    AdversarialStudent, adversarial_roster      students who push for the answer    (C's runner)
    prepost_forms                               pre/post-test items per session     (C's runner)
    EVENTS, check_files                         required log fields and checker     (C's log format)
    metrics.compute, stats                      metrics and statistics from run logs (everyone)
    labels                                      blind labelling sheets, agreement    (everyone, week 4)
"""

from .adversarial import ATTACKS, AdversarialStudent, LLMAttackLine, adversarial_roster
from .items import SAMPLE_ITEMS, SLIP_ITEMS, item_key, overlapping
from .judges import LLMTextJudge
from .log_fields import EVENTS, check_files, check_record, check_records, read_jsonl
from .prepost import TestForm, prepost_forms
from .students import (
    GROUPS, SLIP_ONLY_RATE, LLMFeedbackJudge, SimStudent, StudentReply, correct_answer, roster, rule_answers, slips,
    student_group,
)

__all__ = [
    "ATTACKS", "AdversarialStudent", "EVENTS", "GROUPS", "LLMAttackLine", "LLMFeedbackJudge", "LLMTextJudge",
    "SAMPLE_ITEMS", "SLIP_ITEMS", "SLIP_ONLY_RATE", "SimStudent", "StudentReply", "TestForm", "adversarial_roster",
    "check_files", "check_record", "check_records", "correct_answer", "item_key", "overlapping", "prepost_forms",
    "read_jsonl", "roster", "rule_answers", "slips", "student_group",
]
