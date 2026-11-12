"""diagnose(): which misconception caused a wrong step. Called by C's orchestrator.

    from matheasy.diagnosis import check_step, diagnose

    if not check_step("1/2 + 1/3", "2/5"):
        d = diagnose("1/2 + 1/3", "2/5", student_id="student-017")
        d.status         # 'clear' | 'ambiguous' | 'unknown'
        d.misconception  # 'rule:fractions.add_numerators_denominators' (or 'eedi:1287') when clear, else None
        d.probe          # a follow-up question to ask, or None
        d.to_log()       # JSON-safe record for the run logs

The diagnoser keeps no state between calls. When it returns a probe, the
orchestrator asks it, then calls diagnose() again with the same step and the
answers so far in `follow_ups`. At most MAX_FOLLOW_UPS probes are ever returned.

Rules run first, on the located change and then on solve back's pieces (solve_back.py).
Rules that tie are 'ambiguous', and a probe from the bank (probes.py) tells them apart.
A step no rule reproduces goes to the LLM fallback (explain.py) if a session is given;
see docs/diagnosis_fallback.md. With an `eedi_ranker` too (eedi.py), the fallback picks
from the closest Eedi misconceptions, and a pick becomes the diagnosis: 'eedi:<n>'. d.topics names the topics whichever found the mistake.
A step done right on every term of a bracket but one, where a term like the wrong one came
out right, is a slip (slips.py): d.misconception == 'slip', with no model call.
The LLM's reconstructed working is checked step by step: a wrong step a rule reproduces
exactly makes that rule the diagnosis (ahead of an Eedi pick), and wrong steps that are all
bracket slips make it a slip. Several rules, one per wrong step, are all kept in d.beliefs.
mode="b2" is the same without probes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, Union

from .eedi import EediRanker
from .eedi import options as eedi_options
from .explain import Explanation, explain, rule_topic
from .bank import load_bank
from .ids import eedi_id, split_id
from .interfaces import NoMemory, RuleShortlister, Shortlister, StudentStore
from .maths import BinOp, LocatedError, Node, Num, equivalent, locate
from .probes import Probe, choose_probe
from .ids import SLIP
from .slips import bracket_slip
from .solve_back import rules_for, solve_back
from .topics import question_topics

MAX_FOLLOW_UPS = 2

Status = Literal["clear", "ambiguous", "unknown"]
Mode = Literal["full", "b2"]


@dataclass(frozen=True)
class FollowUp:
    probe: Probe
    answer: str  # what the student wrote


@dataclass(frozen=True)
class Diagnosis:
    status: Status
    misconception: Optional[str]   # set only when status is 'clear'
    beliefs: dict[str, float]      # misconception ID -> probability, most likely first
    probe: Optional[Probe]         # ask this next, or None
    located: LocatedError
    mode: Mode
    trace: tuple[dict, ...] = field(default=())  # every reasoning step, for the logs
    explanation: Optional[Explanation] = None    # the LLM's reading of an 'unknown' step, if a session was given
    topics: tuple[str, ...] = ()                 # every topic involved, whether rules or the LLM found the mistake

    def to_log(self) -> dict:
        return {
            "topics": list(self.topics),
            "explanation": None if self.explanation is None else self.explanation.to_log(),
            "status": self.status,
            "misconception": self.misconception,
            "beliefs": self.beliefs,
            "probe": None if self.probe is None else {"question": self.probe.question,
                                                      "predictions": self.probe.predictions},
            "located": self.located.to_dict(),
            "mode": self.mode,
            "trace": list(self.trace),
        }


class StepIsCorrect(ValueError):
    """diagnose() was called on a correct step. Call check_step() first."""


def diagnose(
    before: Union[str, Node],
    after: Union[str, Node],
    *,
    student_id: Optional[str] = None,
    follow_ups: Sequence[FollowUp] = (),
    shortlister: Optional[Shortlister] = None,
    store: Optional[StudentStore] = None,
    mode: Mode = "full",
    k: int = 25,
    question: str = "",
    session=None,
    eedi_ranker: Optional[EediRanker] = None,
) -> Diagnosis:
    """Diagnose the wrong step before -> after. Raises ParseError or StepIsCorrect.

    With an LLM `session`, a step no rule reproduces is passed to explain(), so the
    result carries the model's reading of it; without one, no model is called. With an
    `eedi_ranker` as well, the model may name an Eedi misconception (eedi.py).
    """
    if mode not in ("full", "b2"):
        raise ValueError(f"unknown mode {mode!r}")
    if len(follow_ups) > MAX_FOLLOW_UPS:
        raise ValueError(f"at most {MAX_FOLLOW_UPS} follow-ups")
    located = locate(before, after)
    if located is None:
        raise StepIsCorrect(f"{before!r} -> {after!r} is correct")
    shortlister = shortlister or RuleShortlister()
    store = store or NoMemory()  # not read yet

    candidates = sorted(shortlister(located, k), key=lambda c: -c.score)
    trace = [
        {"stage": "locate", **located.to_dict()},
        {"stage": "shortlist", "candidates": [[c.id, c.score] for c in candidates],
         "failed_rules": list(getattr(shortlister, "failed", []))},  # rules that crashed: bugs to fix
    ]
    total = sum(max(c.score, 0.0) for c in candidates)
    beliefs = {c.id: (max(c.score, 0.0) / total if total else 1 / len(candidates)) for c in candidates}

    if not candidates:
        status, top = "unknown", None
    elif len(candidates) > 1 and candidates[1].score == candidates[0].score:
        status, top = "ambiguous", None
    else:
        status, top = "clear", candidates[0].id
    trace.append({"stage": "decide", "status": status, "misconception": top})

    pieces = []
    if status == "unknown":  # a whole line turned into one number: find which operation went wrong
        pieces = [(p, rules_for(p)) for p in solve_back(before, after)]
        trace.append({"stage": "solve_back", "pieces": [[p.source, p.wrong, rules] for p, rules in pieces]})
        hits = {rule: p for p, rules in pieces for rule in rules}
        if hits:
            beliefs = {rule: 1 / len(hits) for rule in hits}
            status, top = ("clear", next(iter(hits))) if len(hits) == 1 else ("ambiguous", None)
            if len({id(p) for p in hits.values()}) == 1:
                located = next(iter(hits.values()))  # the wrong piece, not the whole line
            trace.append({"stage": "decide", "status": status, "misconception": top})

    probe = None
    suspects = list(beliefs)
    if status == "ambiguous" and mode == "full" and all(m.startswith("rule:") for m in suspects):
        # the rules tie: ask a probe, or read the answers so far
        for follow_up in follow_ups:
            suspects = _still_suspected(suspects, follow_up)
        beliefs = {m: 1 / len(suspects) for m in suspects}
        if not suspects:
            status = "unknown"  # the student's answers fit none of the rules
        elif len(suspects) == 1:
            status, top = "clear", suspects[0]
        elif len(follow_ups) < MAX_FOLLOW_UPS:
            probe = choose_probe(suspects, [f.probe.question for f in follow_ups])
        trace.append({"stage": "probe", "suspects": suspects, "status": status,
                      "probe": None if probe is None else probe.question})

    slipped = None
    if status == "unknown":  # a step done right on every term of a bracket but one: a slip, not a belief
        slipped = bracket_slip(before, after)
        if slipped:
            status, top, beliefs = "clear", SLIP, {SLIP: 1.0}
            trace.append({"stage": "slip", "reason": slipped})

    explanation = None
    if status == "unknown" and session is not None:
        choices = eedi_options(located, eedi_ranker) if eedi_ranker is not None else []
        explanation = explain(located, question=question, session=session, pieces=[p for p, _ in pieces],
                              options=choices)
        trace.append({"stage": "explain", "eedi_options": [mid for mid, _ in choices], **explanation.to_log()})
        found = _from_working(explanation)
        if found:
            top, beliefs = found[0], {m: 1 / len(found) for m in found}
            status = "clear"
            if top == SLIP:
                trace.append({"stage": "slip", "reason": "; and ".join(w.slip for w in explanation.wrong_steps)})
            trace.append({"stage": "decide", "status": status, "misconception": top, "from": "working"})
        elif explanation.eedi is not None and explanation.kind == "misconception" and explanation.status == "verified":
            # Only from a verified reading (T6): when code found the model's working wrong (it called a right
            # step the mistake, say), its Eedi pick was made for a mistake the student did not make.
            # One ID per mistake: an Eedi name that belongs to exactly one of our rules is that rule,
            # so memory counts it with the rule's own diagnoses and the rule can re-check it later.
            # A near-duplicate Eedi name is recorded as its group (T7), so memory counts one mistake once.
            from matheasy.shortlister.name_groups import canonical
            picked = canonical(int(split_id(explanation.eedi)[1]))
            owners = load_bank().for_eedi(picked)
            top = owners[0] if len(owners) == 1 else eedi_id(picked)
            status, beliefs = "clear", {top: 1.0}
            trace.append({"stage": "decide", "status": status, "misconception": top, "eedi": explanation.eedi})
    topics = (explanation.topics if explanation is not None
              else ("expanding_brackets",) if slipped
              else tuple(sorted({rule_topic(m) for m in beliefs if m.startswith("rule:")})))
    if not topics and status == "unknown":
        # Nothing named the mistake, so name what it was made on (T3): otherwise the wrong answer
        # leaves no trace in the student's memory.
        topics = step_topics(located, [p for p, _ in pieces])
        trace.append({"stage": "topics", "from": "solve_back" if pieces else "located", "topics": list(topics)})
    return Diagnosis(status, top, beliefs, probe, located, mode, tuple(trace), explanation, topics)


def step_topics(located: LocatedError, pieces: Sequence[LocatedError] = ()) -> tuple[str, ...]:
    """The topics of the wrong part of a step, read from its form, for a step nothing explains.
    2(x + 3) -> 2x + 3 gives expanding_brackets and algebraic_notation. For a line turned into one
    number, solve back's pieces are used: 5 × 6 -> 26 gives whole_number_arithmetic."""
    if not pieces:
        return question_topics(located.source_node)
    found: set[str] = set()
    for piece in pieces:
        topics = question_topics(piece.source_node)
        if not topics and _whole_numbers_only(piece.source_node):
            topics = ("whole_number_arithmetic",)  # 5 × 6: a number fact, below question_topics' 10 + 10
        found.update(topics)
    return tuple(sorted(found))


def _whole_numbers_only(node: Node) -> bool:
    if isinstance(node, Num):
        return node.text.isdigit()
    if isinstance(node, BinOp) and not node.typed_fraction:
        return _whole_numbers_only(node.left) and _whole_numbers_only(node.right)
    return False


def _from_working(explanation: Explanation) -> list[str]:
    """What code finds in the LLM's verified working: the rules reproducing its wrong steps, in
    order, or [SLIP] when every wrong step is a bracket slip; else []."""
    if explanation.status != "verified" or not explanation.wrong_steps:
        return []
    rules = list(dict.fromkeys(w.rule for w in explanation.wrong_steps if w.rule))
    if rules:
        return rules
    return [SLIP] if all(w.slip for w in explanation.wrong_steps) else []


def _still_suspected(suspects: list[str], follow_up: FollowUp) -> list[str]:
    """The suspects that fit the student's answer.

    Those predicting it, if any. A right answer rules out those predicting a wrong one. An
    answer no suspect predicts says nothing, so all stay: no rule wins by elimination alone.
    """
    predicted = follow_up.probe.predictions
    matching = [s for s in suspects if s in predicted and equivalent(predicted[s], follow_up.answer)]
    if matching:
        return matching
    if equivalent(follow_up.probe.question, follow_up.answer):
        return [s for s in suspects if s not in predicted]
    return suspects
