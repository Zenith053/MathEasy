"""The diagnosis engine (owner: B): maths tools, misconception rules and bank, and diagnose().

Shared functions other roles call:
    parse, equivalent, check_step, locate    the maths layer          (A, C, D)
    run_rule, rule_names                     the rule runner           (A, D)
    load_bank                                the misconception bank    (A, D)
    diagnose                                 the diagnoser             (C's orchestrator)
"""

from .bank import Bank, Misconception, load_bank
from .diagnose import MAX_FOLLOW_UPS, Diagnosis, FollowUp, Probe, StepIsCorrect, diagnose
from .ids import SLIP, eedi_id, rule_id, split_id
from .interfaces import Candidate, EediShortlister, NoMemory, RuleShortlister, Shortlister, StudentStore
from .maths import LocatedError, Node, ParseError, as_fraction, check_step, equivalent, locate, parse, topic_of
from .rules import HeldBackRuleError, held_back_rules, rule_names, run_rule

__all__ = [
    "Bank", "Candidate", "Diagnosis", "EediShortlister", "FollowUp", "HeldBackRuleError", "LocatedError", "MAX_FOLLOW_UPS",
    "Misconception", "Node", "NoMemory", "ParseError", "Probe", "RuleShortlister", "SLIP", "Shortlister",
    "StepIsCorrect", "StudentStore", "as_fraction", "check_step", "diagnose", "eedi_id", "equivalent",
    "held_back_rules", "load_bank", "locate", "parse", "rule_id", "rule_names", "run_rule", "split_id",
    "topic_of",
]
