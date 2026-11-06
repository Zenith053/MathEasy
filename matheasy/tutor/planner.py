"""The planner (Role C): keeps the student model up to date and schedules re-checks.

Hand-written rules, not an LLM (proposal, section 7). The store calls the
update functions; the experiment runner calls plan() before each session.

    P(knows skill)          per topic (B's topic labels, explain.TOPICS): Bayesian
                            knowledge tracing on every answer the student gives to a
                            problem, for the topics its rules cover and the topics
                            the diagnosis names. A topic is mastered at MASTERED.
    Mistakes per topic      a count of the wrong answers diagnosed in each topic.
    P(holds misconception)  per misconception ID: set high by a clear diagnosis,
                            then updated by Bayes' rule whenever the student answers
                            an item that the misconception gets wrong: down for the
                            right answer, up for the misconception's answer. Between
                            sessions it rises a little, since a fixed misconception
                            can come back.
    Re-checks               a misconception that still looks held is worked on again
                            next session. One that looks fixed is re-checked next
                            session, then after 2, 4, ... sessions: each passed
                            re-check doubles the gap, a failed one resets it.

plan() puts one re-check item first for each misconception that is due, then one
practice problem from B's probe bank on the weakest topic the student has made
mistakes in ("what to practise"), then the rest of the pool, weakest topic first.
Only rule misconceptions can be re-checked: an Eedi ID has no code to say which
items it gets wrong.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Sequence

from ..diagnosis import SLIP, ParseError, RuleError, equivalent, rule_id, rule_names, run_rule, split_id
from ..diagnosis.explain import rule_topic
from ..diagnosis.probes import similar_items
from ..evaluation import correct_answer
from .state import Problem, StudentMemory

# Knowledge tracing (Corbett and Anderson, 1995), standard textbook values.
KNOWS_AT_START = 0.5
LEARN = 0.2  # chance of learning the skill at each answer
GUESS = 0.2  # right answer without the skill
SLIP_RATE = 0.1  # wrong answer despite the skill
MASTERED = 0.95  # the usual knowledge-tracing threshold

HOLDS_WHEN_DIAGNOSED = 0.9
FOLLOWS = 0.9  # a student holding a misconception gives its answer on an item it gets wrong
STRAY = 0.05  # a student without it gives that same wrong answer anyway
RELAPSE = 0.1  # share of the gap to 1 that P(holds) regains between sessions
FIXED_BELOW = 0.5  # P(holds) under this counts as fixed


def begin_session(memory: StudentMemory, session_index: int) -> None:
    """Start a session: misconceptions may have come back since the last one."""
    if session_index <= memory.session_index:
        return  # the same session again: nothing has changed
    for mid, p in memory.p_holds.items():
        memory.p_holds[mid] = p + (1 - p) * RELAPSE
    memory.session_index = session_index


def diagnosed(memory: StudentMemory, misconception_id: str) -> None:
    """A clear diagnosis of `misconception_id` on the student's last answer."""
    memory.times_diagnosed[misconception_id] = memory.times_diagnosed.get(misconception_id, 0) + 1
    if misconception_id == SLIP:
        return  # a slip is not something the student holds
    memory.p_holds[misconception_id] = max(memory.p_holds.get(misconception_id, 0.0), HOLDS_WHEN_DIAGNOSED)
    memory.next_check.setdefault(misconception_id, memory.session_index + 1)
    _note(memory, "diagnosed", misconception=misconception_id)


def answered(memory: StudentMemory, problem: Problem, answer: str, correct: bool,
             topics: Sequence[str] = ()) -> None:
    """The student answered `problem` (not a probe or a test item). `topics` are the
    diagnosis's, for a wrong answer."""
    if problem.question not in memory.worked:
        memory.worked.append(problem.question)
    for topic in sorted(set(problem_topics(problem.question)) | set(topics)):
        p = memory.p_knows.get(topic, KNOWS_AT_START)
        if correct:
            p = p * (1 - SLIP_RATE) / (p * (1 - SLIP_RATE) + (1 - p) * GUESS)
        else:
            p = p * SLIP_RATE / (p * SLIP_RATE + (1 - p) * (1 - GUESS))
        memory.p_knows[topic] = p + (1 - p) * LEARN
    for topic in set(topics):
        memory.mistakes[topic] = memory.mistakes.get(topic, 0) + 1

    for mid, held in list(memory.p_holds.items()):
        predicted = exposes(mid, problem.question, problem.correct_answer)
        if predicted is None:
            continue
        if correct:
            likely_if_held, likely_if_not = 1 - FOLLOWS, 1 - STRAY
        elif _equal(answer, predicted):
            likely_if_held, likely_if_not = FOLLOWS, STRAY
        else:
            continue  # wrong in some other way: says nothing about this misconception
        memory.p_holds[mid] = held * likely_if_held / (held * likely_if_held + (1 - held) * likely_if_not)
        _note(memory, "rechecked", misconception=mid, item=problem.question, passed=correct)


def end_session(memory: StudentMemory) -> None:
    """Schedule each misconception's next re-check from what this session showed."""
    now = memory.session_index
    passed = {h["misconception"]: h["passed"] for h in memory.history
              if h["session"] == now and h["event"] == "rechecked"}  # the last re-check of each counts
    for mid, p in memory.p_holds.items():
        if p >= FIXED_BELOW:
            memory.check_gap[mid] = 1
        elif passed.get(mid) and memory.next_check.get(mid, now + 1) <= now:
            memory.check_gap[mid] = 2 * memory.check_gap.get(mid, 1)  # a re-check that was due, passed
        elif passed.get(mid) is False:
            memory.check_gap[mid] = 1
        else:
            memory.check_gap.setdefault(mid, 1)
        memory.next_check[mid] = now + memory.check_gap[mid]


def due(memory: StudentMemory, session_index: int) -> list[str]:
    """Misconceptions to re-check in `session_index`, most likely still held first."""
    found = [mid for mid, s in memory.next_check.items() if s <= session_index and mid in memory.p_holds]
    return sorted(found, key=lambda mid: -memory.p_holds[mid])


def mastered(memory: StudentMemory, topic: str) -> bool:
    return memory.p_knows.get(topic, KNOWS_AT_START) >= MASTERED


def weak_topics(memory: StudentMemory) -> list[str]:
    """Topics the student has made mistakes in and not mastered, weakest first."""
    found = [t for t in memory.mistakes if not mastered(memory, t)]
    return sorted(found, key=lambda t: (memory.p_knows.get(t, KNOWS_AT_START), -memory.mistakes[t], t))


def plan(memory: StudentMemory, session_index: int, pool: Sequence[str], n: int) -> tuple[list[str], list[str]]:
    """The session's `n` problems, and the misconceptions they re-check."""
    chosen, rechecks = [], []
    for mid in due(memory, session_index):
        item = next((q for q in pool if q not in chosen and exposes(mid, q) is not None), None)
        if item is not None and len(chosen) < n:
            chosen.append(item)
            rechecks.append(mid)
    for topic in weak_topics(memory)[:1]:
        practice = similar_items([topic], [*memory.worked, *pool, *chosen], n=1)
        if practice and len(chosen) < n:
            chosen.append(practice[0])
    rest = [q for q in pool if q not in chosen]
    rest.sort(key=lambda q: _weakness(memory, q))  # stable: pool order within a topic
    return chosen + rest[:n - len(chosen)], rechecks


@lru_cache(maxsize=None)
def problem_topics(question: str) -> tuple[str, ...]:
    """The topics of the rules that get `question` wrong: what it tests."""
    try:
        right = correct_answer(question)
    except Exception:  # D's answer key can't work it out
        return ()
    return tuple(sorted({rule_topic(rule_id(name)) for name in rule_names()
                         if exposes(rule_id(name), question, right) is not None}))


def exposes(misconception_id: str, item: str, right: Optional[str] = None) -> Optional[str]:
    """The misconception's answer to `item` if it is wrong, else None (or not a rule)."""
    try:
        kind, name = split_id(misconception_id)
        if kind != "rule":
            return None
        predicted = run_rule(name, item)
        right = right or correct_answer(item)
    except (KeyError, ValueError, ParseError, RuleError):
        return None
    if predicted is None or _equal(predicted, right):
        return None
    return predicted


def _equal(a: str, b: str) -> bool:
    try:
        return equivalent(a, b)
    except (ParseError, ValueError, TypeError):
        return False


def _weakness(memory: StudentMemory, question: str) -> float:
    return min((memory.p_knows.get(t, KNOWS_AT_START) for t in problem_topics(question)), default=KNOWS_AT_START)


def _note(memory: StudentMemory, event: str, **fields) -> None:
    memory.history.append({"session": memory.session_index, "event": event, **fields})
