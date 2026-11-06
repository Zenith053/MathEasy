"""Pre- and post-test items (owner: D): the same idea with different numbers, kept apart from probes.

    from matheasy.evaluation.prepost import prepost_forms

    forms = prepost_forms("fractions.add_numerators_denominators", sessions=2, student_id="s1#0")
    forms[0].pre     # ['3/7 + 2/5', '1/8 + 5/9', ...]  session 0 pre-test
    forms[0].post    # ['4/9 + 1/7', ...]               session 0 post-test: parallel to pre
    forms[1].pre     # session 1 pre-test: also the retention check for what session 0 fixed

How an item is made (isomorphs()). Start from a seed item that shows the
misconception (SAMPLE_ITEMS), then give it new numbers:

- a single digit becomes 2-9; in longer numbers each digit 1-9 becomes a random
  digit 1-9 and zeros stay zeros (302 - 7 keeps its zero to borrow across, 45000
  and 0.0032 keep their zeros);
- 1 and 0 as whole numbers stay as they are (unit fractions, x^0), and so do the
  10 in 10^4 and the power on a bracketed sum ((x + 3)^2); other powers become 2-6;
- equal numbers stay equal and different numbers stay different (3/4 - 1/4 keeps
  its common denominator);
- each single-letter variable is renamed ((a + b)^2 -> (p + q)^2);
- if a seed runs out of fresh numbers, single digits may widen to 2-12, then 2-25.

A new item is kept only if it tests the same thing as its seed:

- the misconception still gives a wrong answer on it (for a student who has one);
- its answer has the same shape (a whole number stays a whole number, a fraction
  stays a fraction, x = n stays x = n);
- fractions keep their kind (proper or not, in lowest terms or not) and stay in
  B's probe ranges (denominators 2-12) when the seed's were;
- it matches no probe, no item the tutor teaches on and no other test item
  (items.item_key: same operations on the same numbers, in any order).

Pre and post forms are parallel: pre[i] and post[i] come from the same seed.
Every draw is seeded by (seed, student_id, misconception), so a re-run gives the
same tests.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from math import gcd
from typing import Callable, Iterable, Optional, Sequence

from ..diagnosis import ParseError, RuleError, equivalent, held_back_rules, parse, run_rule
from ..diagnosis.maths import Rel
from .held_back import held_back_rule
from .items import SAMPLE_ITEMS, SLIP_ITEMS, item_key
from .students import correct_answer

TEST_SIZE = 4         # items per pre-test and per post-test
TRIES_PER_ITEM = 400  # random variants tried before giving up on a seed
PROBE_DENOMINATORS = range(2, 13)


@dataclass(frozen=True)
class TestForm:
    session_index: int
    pre: tuple[str, ...]
    post: tuple[str, ...]
    seeds: tuple[str, ...]  # seeds[i] is the item pre[i] and post[i] were made from


# ---------------------------------------------------------------- the rule as code

def predictor(misconception: Optional[str]) -> Optional[Callable[[str], Optional[str]]]:
    """item -> the answer the misconception gives, for B's rules and the held-back ones.
    Accepts a name or a rule: ID. None for a student with no misconception."""
    if misconception is None:
        return None
    name = misconception.split(":", 1)[1] if misconception.startswith("rule:") else misconception
    if name in held_back_rules():
        return held_back_rule(name).predict
    return lambda item: run_rule(name, item)


def shows(misconception: Optional[str], item: str) -> bool:
    """Does the misconception give a wrong answer on `item`? Always True with no misconception."""
    predict = predictor(misconception)
    if predict is None:
        return True
    try:
        predicted = predict(item)
        return predicted is not None and not equivalent(predicted, correct_answer(item))
    except (ParseError, ValueError, TypeError, ZeroDivisionError, RuleError):
        return False


def seed_items(misconception: Optional[str]) -> list[str]:
    """The sample items that show the misconception (the fixed slip items for none)."""
    if misconception is None:
        return list(SLIP_ITEMS)
    return [item for item in SAMPLE_ITEMS if shows(misconception, item)]


# ---------------------------------------------------------------- new numbers

# A fraction of whole numbers, or a single number (decimal, whole, percent written separately).
_FRACTION = re.compile(r"(?<![\w.^])(\d+)\s*/\s*(\d+)(?![\w.])")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def _fractions(text: str) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in _FRACTION.findall(text)]


def _is_exponent(text: str, start: int) -> bool:
    return re.search(r"\^\s*-?\s*$", text[:start]) is not None


def _powered_sum(text: str, start: int) -> bool:
    """Is the number at `start` the power on a bracketed sum, as in (x + 3)^2?"""
    before = text[:start].rstrip(" -^")
    if not before.endswith(")"):
        return False
    depth = 0
    for i in range(len(before) - 1, -1, -1):
        depth += {")": 1, "(": -1}.get(before[i], 0)
        if depth == 0:
            return re.search(r"[+\-]", before[i:]) is not None
    return False


def _keep(text: str, start: int, end: int, number: str) -> bool:
    """Numbers that carry the idea and must not change: 0, 1 and 10 as whole numbers
    (x^0, unit fractions, 10^n), and the power on a bracketed sum ((x + 3)^2)."""
    if number in ("0", "1"):
        return True
    if number == "10" and text[end:].lstrip().startswith("^"):
        return True
    return _is_exponent(text, start) and _powered_sum(text, start)


def _redraw(number: str, rng: random.Random, wide: int) -> str:
    """A single digit becomes 2-9 (a 1 would change the idea: 1^0, 1x); other digits 1-9."""
    if re.fullmatch(r"[1-9]", number):
        return str(rng.randint(2, (9, 12, 25)[wide]))  # 12: B's probe range
    return "".join(d if d in "0." else str(rng.randint(1, 9)) for d in number)


# Single-letter variables (not the letters of a word such as "of"), and the letters they may become.
_VARIABLE = re.compile(r"(?<![A-Za-z])[a-z](?![A-Za-z])")
LETTERS = "abcmnpqtxy"


def _reletter(item: str, rng: random.Random) -> str:
    found = list(dict.fromkeys(_VARIABLE.findall(item)))
    if not found:
        return item
    new = dict(zip(found, rng.sample(LETTERS, len(found))))
    return _VARIABLE.sub(lambda m: new[m.group()], item)


def renumber(item: str, rng: random.Random, *, wide: int = 0) -> str:
    """`item` with new numbers and letters: same number of digits, zeros kept, equal numbers
    kept equal (exponents are drawn on their own, from 2 to 6), each letter renamed.
    `wide` 1 lets a single digit become 2-12, and 2 lets it become 2-25, for items with
    few numbers to change (5^0)."""
    item = _reletter(item, rng)
    new: dict[str, str] = {}
    out, last = [], 0
    for m in _NUMBER.finditer(item):
        start, end, number = m.start(), m.end(), m.group()
        if _keep(item, start, end, number):
            continue
        if _is_exponent(item, start):
            value = str(rng.randint(2, 6))
        else:
            if number not in new:
                for _ in range(20):
                    candidate = _redraw(number, rng, wide)
                    if candidate not in new.values():
                        break
                new[number] = candidate
            value = new[number]
        out.append(item[last:start] + value)
        last = end
    out.append(item[last:])
    return "".join(out)


def answer_shape(answer: str) -> str:
    """What kind of answer: 'relation', 'solved', 'expression', 'integer', 'fraction', 'decimal',
    each with a sign. Isomorphic items have answers of the same shape."""
    node = parse(answer)
    if isinstance(node, Rel):
        return "solved" if re.fullmatch(r"\s*[A-Za-z]\s*=.*", answer) else "relation"
    if re.search(r"[A-Za-z]", answer):
        return "expression"
    sign = "-" if answer.strip().startswith("-") else "+"
    if "/" in answer:
        return sign + "fraction"
    return sign + ("decimal" if "." in answer else "integer")


def _trivial(item: str, answer: str) -> bool:
    """The item is already its own answer (12/34 when it is in lowest terms)."""
    return re.sub(r"\s", "", item) == re.sub(r"\s", "", answer)


def _fraction_kinds(text: str) -> list[tuple[bool, bool, bool]]:
    """For each fraction: (proper, in lowest terms, denominator in B's probe range)."""
    return [(a < b, gcd(a, b) == 1, b in PROBE_DENOMINATORS) for a, b in _fractions(text)]


def isomorphic(seed: str, item: str, misconception: Optional[str] = None) -> bool:
    """Does `item` test the same thing as `seed`? See the module docstring."""
    try:
        if not shows(misconception, item):
            return False
        seed_answer, answer = correct_answer(seed), correct_answer(item)
        if answer_shape(seed_answer) != answer_shape(answer):
            return False
        if _trivial(seed, seed_answer) != _trivial(item, answer):
            return False
    except (ParseError, ValueError, TypeError, ZeroDivisionError, NotImplementedError):
        return False
    seed_kinds, kinds = _fraction_kinds(seed), _fraction_kinds(item)
    if len(seed_kinds) != len(kinds):
        return False
    for (proper, lowest, in_range), (p, low, r) in zip(seed_kinds, kinds):
        if p != proper or low != lowest or (in_range and not r):
            return False
    return not any(b == 0 or b == 1 for _, b in _fractions(item))


def isomorphs(seed: str, n: int, *, misconception: Optional[str] = None, rng: Optional[random.Random] = None,
              taken: Optional[set[str]] = None) -> list[str]:
    """Up to `n` new items isomorphic to `seed`, none matching a key in `taken` (which is updated)."""
    rng = rng or random.Random(0)
    taken = set() if taken is None else taken
    found = []
    for attempt in range(TRIES_PER_ITEM):
        if len(found) == n:
            break
        candidate = renumber(seed, rng, wide=3 * attempt // TRIES_PER_ITEM)  # widen once digits run short
        try:
            key = item_key(candidate)
        except (ParseError, TypeError, ValueError):
            continue
        if key in taken or not isomorphic(seed, candidate, misconception):
            continue
        taken.add(key)
        found.append(candidate)
    return found


# ---------------------------------------------------------------- probes and forms

def probe_items() -> list[str]:
    """Every probe B can ask (B's probe_items(), once the probe bank lands; empty until then)."""
    from .. import diagnosis
    published = getattr(diagnosis, "probe_items", None)
    return list(published()) if callable(published) else []


def _keys(items: Iterable[str]) -> set[str]:
    keys = set()
    for item in items:
        try:
            keys.add(item_key(item))
        except (ParseError, TypeError, ValueError):
            pass
    return keys


def prepost_forms(
    misconception: Optional[str],
    *,
    sessions: int = 1,
    size: int = TEST_SIZE,
    student_id: str = "",
    seed: int = 0,
    avoid: Sequence[str] = (),
    probes: Optional[Sequence[str]] = None,
) -> list[TestForm]:
    """One TestForm per session, each with `size` pre and `size` post items.

    misconception   a rule name or rule: ID, or None for a slip-only student
    avoid           items the tutor teaches on; tests never match them
    probes          B's probe items; by default probe_items()

    Items cycle through the seeds that show the misconception, so each seed idea
    appears in pre and post equally. Every item in every form is different. Raises
    ValueError if no sample item shows the misconception."""
    seeds = seed_items(misconception)
    if not seeds:
        raise ValueError(f"no sample item shows {misconception}; add one to items.SAMPLE_ITEMS")
    rng = random.Random(f"{seed}:{student_id}:{misconception}:tests")
    taken = _keys(probe_items() if probes is None else probes) | _keys(avoid) | _keys(seeds)
    pools: dict[str, list[str]] = {s: [] for s in seeds}
    forms = []
    order = [seeds[i % len(seeds)] for i in range(size)]
    for index in range(sessions):
        pre, post, used = [], [], []
        for s in order:
            while len(pools[s]) < 2:
                more = isomorphs(s, 2 * size, misconception=misconception, rng=rng, taken=taken)
                if not more:
                    break
                pools[s] += more
            if len(pools[s]) < 2:
                continue  # this seed has run out of fresh numbers; the others cover it
            pre.append(pools[s].pop(0))
            post.append(pools[s].pop(0))
            used.append(s)
        forms.append(TestForm(index, tuple(pre), tuple(post), tuple(used)))
    return forms
