"""The diagnosis engine (owner: B): maths tools, misconception rules and bank, and diagnose().

Shared functions other roles call:
    parse, equivalent, check_step, locate    the maths layer          (A, C, D)
    run_rule, rule_names                     the rule runner           (A, D)
    load_bank                                the misconception bank    (A, D)
    diagnose                                 the diagnoser             (C's orchestrator)
    explain                                  LLM reading of a step no rule reproduces (called by diagnose)
    transcribe, notation_issue               student message -> standard notation, before the diagnoser (C, D)
"""

from .bank import Bank, Misconception, load_bank
from .diagnose import MAX_FOLLOW_UPS, Diagnosis, FollowUp, Probe, StepIsCorrect, diagnose
from .explain import Explanation, WrongStep, explain
from .ids import SLIP, eedi_id, rule_id, split_id
from .interfaces import Candidate, EediShortlister, NoMemory, RuleShortlister, Shortlister, StudentStore
from .maths import (
    LocatedError, Node, ParseError, as_fraction, check_step, equivalent, locate, notation_issue, parse, topic_of,
    unknown_runs,
)
from .rules import HeldBackRuleError, RuleError, held_back_rules, rule_names, run_rule
from .transcribe import Transcription, resolve, transcribe

__all__ = [
    "Bank", "Candidate", "Diagnosis", "EediShortlister", "FollowUp", "HeldBackRuleError", "LocatedError", "MAX_FOLLOW_UPS",
    "Misconception", "Node", "NoMemory", "ParseError", "Probe", "RuleShortlister", "SLIP", "Shortlister",
    "StepIsCorrect", "StudentStore", "WrongStep", "as_fraction", "check_step", "diagnose", "eedi_id", "equivalent",
    "Explanation", "explain",
    "held_back_rules", "load_bank", "locate", "notation_issue", "parse", "rule_id", "RuleError", "rule_names",
    "run_rule", "resolve", "split_id", "topic_of", "Transcription", "transcribe", "unknown_runs",
]
