"""Whole-number column arithmetic bugs, ported from MalruleLib's digit-by-digit algorithms.

Each runs on two whole numbers joined by + − or × (written with digits), wherever
they appear in the student's line once narrowed, and says nothing when the bug
happens to give the right answer (no carry or borrow needed).
"""

from __future__ import annotations

from fractions import Fraction
from typing import Callable, Optional

from ..maths import BinOp, Node
from . import rule
from ._common import fraction_node, whole

Bug = Callable[[int, int], Optional[int]]


def _column_rule(name: str, op: str, bug: Bug, applies: Callable[[int, int], bool] = lambda a, b: True):
    """Registers `bug` as a rule for `a op b` with whole numbers a and b."""
    correct = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}[op]

    def apply(node: Node) -> Optional[Node]:
        if not (isinstance(node, BinOp) and node.op == op):
            return None
        a, b = whole(node.left), whole(node.right)
        if a is None or b is None or not applies(a, b):
            return None
        wrong = bug(a, b)
        return None if wrong is None or wrong == correct(a, b) else fraction_node(Fraction(wrong))

    apply.__doc__ = bug.__doc__
    rule(name)(apply)


def _digits(n: int, width: int = 0) -> list[int]:
    """Digits, most significant first, padded with zeros on the left to `width`."""
    return [int(d) for d in str(n).zfill(width)]


def _join(digits: list[int]) -> int:
    """MalruleLib's way of writing a result: each column's value side by side ('1' '12' -> 112)."""
    return int("".join(str(d) for d in digits).lstrip("0") or "0")


# ---------------------------------------------------------------- addition

def _add_all_digits(a: int, b: int) -> int:
    """Adds every digit regardless of place: 23 + 45 = 2 + 3 + 4 + 5 = 14."""
    return sum(int(d) for d in f"{a}{b}")


def _carry_adds_to_same_column(a: int, b: int) -> int:
    """Adds the carry into the same column: 37 + 25 gives 7 + 5 = 12, written 1 + 2 = 3, so 53."""
    width = max(len(str(a)), len(str(b)))
    columns = [x + y for x, y in zip(_digits(a, width), _digits(b, width))]
    return _join([c % 10 + c // 10 if c > 9 else c for c in columns])


def _carry_not_reset(a: int, b: int) -> int:
    """Never resets the carry after using it: 328 + 917 = 1345, carrying into every later column."""
    width = max(len(str(a)), len(str(b)))
    out, carry = [], 0
    for x, y in zip(reversed(_digits(a, width)), reversed(_digits(b, width))):
        total = x + y + carry
        if total >= 10:
            out.append(total % 10)
            carry = total // 10
        else:
            out.append(total)  # the carry is kept for the next column too
    out += [int(d) for d in str(carry)[::-1]] if carry else []
    return _join(out[::-1])


def _left_justify(a: int, b: int) -> int:
    """Lines the numbers up on the left: 8 + 12 = 80 + 12 = 92."""
    la, lb = len(str(a)), len(str(b))
    return a * 10 ** (lb - la) + b if la < lb else a + b * 10 ** (la - lb)


def _single_digit_linked(a: int, b: int) -> int:
    """Writes the single digit next to the other number: 8 + 12 = 128."""
    return int(f"{b}{a}") if a < 10 <= b else int(f"{a}{b}")


def _write_carry_forget_units(a: int, b: int) -> int:
    """Writes the carry and forgets the units: 87 + 93 = 11 (7 + 3 = 10 and 8 + 9 = 17, each written as its carry)."""
    width = max(len(str(a)), len(str(b)))
    columns = [x + y for x, y in zip(_digits(a, width), _digits(b, width))]
    return _join([c // 10 if c >= 10 else c for c in columns])


def _no_column_limit(a: int, b: int) -> int:
    """Writes each column's whole sum without regrouping: 47 + 38 = 7|15 = 715."""
    width = max(len(str(a)), len(str(b)))
    return _join([x + y for x, y in zip(_digits(a, width), _digits(b, width))])


def _forget_carry(a: int, b: int) -> int:
    """Forgets to add the carried 1: 47 + 38 gives 7 + 8 = 15, writes 5 and drops the 1, then 4 + 3 = 7, so 75.
    The leftmost column is written in full, as students do: 95 + 18 = 103."""
    width = max(len(str(a)), len(str(b)))
    columns = [x + y for x, y in zip(_digits(a, width), _digits(b, width))]
    return _join([columns[0]] + [c % 10 for c in columns[1:]])


_column_rule("buggy_addition.add_all_digits", "+", _add_all_digits)
_column_rule("buggy_addition.forget_carry", "+", _forget_carry)
_column_rule("buggy_addition.carry_adds_to_same_column", "+", _carry_adds_to_same_column)
_column_rule("buggy_addition.carry_not_reset", "+", _carry_not_reset)
_column_rule("buggy_addition.left_justify", "+", _left_justify, lambda a, b: len(str(a)) != len(str(b)))
_column_rule("buggy_addition.single_digit_linked", "+", _single_digit_linked, lambda a, b: (a < 10) != (b < 10))
_column_rule("buggy_addition.write_carry_forget_units", "+", _write_carry_forget_units)
_column_rule("subtraction.no_column_limit", "+", _no_column_limit)  # MalruleLib files it under subtraction


# ---------------------------------------------------------------- subtraction (top number not smaller)

def _borrow(top: list[int], i: int, zero_becomes: int = 9) -> bool:
    """Borrows for column i from the nearest non-zero column to its left; zeros passed become `zero_becomes`."""
    j = i - 1
    while j >= 0 and top[j] == 0:
        j -= 1
    if j < 0:
        return False
    top[j] -= 1
    for k in range(j + 1, i):
        top[k] = zero_becomes
    return True


def _borrow_into_1_equals_10(a: int, b: int) -> int:
    """Borrowing into a 1 gives 10, not 11: 321 - 89 = 231."""
    original, bottom = _digits(a), _digits(b, len(str(a)))
    top, out = list(original), []
    for i in range(len(top) - 1, -1, -1):
        digit = top[i]
        if digit < bottom[i] and i > 0 and _borrow(top, i):
            digit = 10 if original[i] == 1 else digit + 10
            top[i] = digit
        out.insert(0, digit - bottom[i])
    return _join(out)


def _borrow_writes_10(a: int, b: int) -> int:
    """Borrowing across a 0 turns it into 10, not 9: 403 - 168 = 245."""
    bottom, top, out = _digits(b, len(str(a))), _digits(a), []
    for i in range(len(top) - 1, -1, -1):
        if top[i] < bottom[i] and _borrow(top, i, zero_becomes=10):
            top[i] += 10
        out.insert(0, top[i] - bottom[i])
    result, carry = [], 0  # a column of 10 or more carries into the next, as MalruleLib writes it
    for d in reversed(out):
        d += carry
        carry = d // 10
        result.insert(0, d % 10)
    return _join(([carry] if carry else []) + result)


def _diff_0_n(write_bottom: bool) -> Bug:
    def bug(a: int, b: int) -> int:
        original, bottom = _digits(a), _digits(b, len(str(a)))
        top, out = list(original), []
        for i in range(len(top) - 1, -1, -1):
            if original[i] == 0 and bottom[i] > 0:
                out.insert(0, bottom[i] if write_bottom else 0)
                continue
            if top[i] < bottom[i] and i > 0 and _borrow(top, i):
                top[i] += 10
            out.insert(0, top[i] - bottom[i])
        return _join(out)
    return bug


_diff_0_n_equals_0 = _diff_0_n(write_bottom=False)
_diff_0_n_equals_0.__doc__ = "Writes 0 whenever the top digit is 0: 140 - 21 = 120."
_diff_0_n_equals_n = _diff_0_n(write_bottom=True)
_diff_0_n_equals_n.__doc__ = "Writes the bottom digit whenever the top digit is 0: 140 - 21 = 121."


def _diff_n_0_equals_0(a: int, b: int) -> int:
    """Writes 0 whenever the bottom digit is 0: 403 - 208 = 105, the tens column written 0."""
    top, bottom = _digits(a)[::-1], _digits(b, len(str(a)))[::-1]
    out = []
    for i in range(len(top)):
        if bottom[i] == 0:
            out.append(0)
            continue
        digit = top[i]
        if digit < bottom[i]:
            j = i + 1
            while j < len(top) and top[j] == 0:
                top[j] = 9
                j += 1
            if j < len(top):
                top[j] -= 1
            digit += 10
        out.append(digit - bottom[i])
    return _join(out[::-1])


def _left_ten_ok(a: int, b: int) -> int:
    """Changes a 0 to 9 without borrowing further, except in a leading 10: 302 - 7 = 395, but 1003 - 7 = 996."""
    text, bottom = str(a), _digits(b, len(str(a)))
    top, out = _digits(a), []

    def leading_ten(zero: int) -> bool:
        if len(text) < 2 or text[:2] != "10":
            return False
        end = 1
        while end < len(text) and text[end] == "0":
            end += 1
        return 1 <= zero < end

    for i in range(len(top) - 1, -1, -1):
        digit = top[i]
        if digit < bottom[i] and i > 0:
            if top[i - 1] == 0:
                if leading_ten(i - 1):
                    if _borrow(top, i):
                        digit += 10
                else:
                    top[i - 1] = 9  # the bug: no borrow from further left
                    digit += 10
            else:
                top[i - 1] -= 1
                digit += 10
        out.insert(0, digit - bottom[i])
    return _join(out)


def _move_over_zero_borrow(a: int, b: int) -> int:
    """Skips over a 0 when borrowing, leaving it 0: 304 - 75 = 139."""
    bottom, top, out = _digits(b, len(str(a))), _digits(a), []
    for i in range(len(top) - 1, -1, -1):
        digit = top[i]
        if digit < bottom[i]:
            j = i - 1
            while j >= 0 and top[j] == 0:
                j -= 1  # the zeros are skipped, not changed
            if j >= 0:
                top[j] -= 1
                out.insert(0, digit + 10 - bottom[i])
            else:
                out.insert(0, abs(digit - bottom[i]))
        else:
            out.insert(0, digit - bottom[i])
    return _join(out)


def _always_borrow_left(a: int, b: int) -> int:
    """Borrows in every column, needed or not: 85 - 23 = 52 (15 - 3 = 12, write 2; then 7 - 2 = 5)."""
    if a < 10:
        return a - b
    top, bottom = _digits(a)[::-1], _digits(b, len(str(a)))[::-1]
    out = []
    for i in range(len(top)):
        if i < len(top) - 1:
            out.append((top[i] + 10 - bottom[i]) % 10)
            top[i + 1] -= 1
        else:
            out.append(max(top[i] - bottom[i], 0))
    return sum(d * 10 ** i for i, d in enumerate(out))


def _borrow_from_bottom(a: int, b: int) -> int:
    """Borrows from the bottom number instead of the top: 52 - 17 = 55 (12 - 7 = 5; the 1 below becomes 0)."""
    top, bottom = _digits(a)[::-1], _digits(b, len(str(a)))[::-1]
    out = []
    for i in range(len(top)):
        if top[i] < bottom[i]:
            j = i + 1
            while j < len(bottom) and bottom[j] == 0:
                j += 1
            if j < len(bottom):
                for k in range(i + 1, j):
                    bottom[k] = 9
                bottom[j] -= 1
                out.append(top[i] + 10 - bottom[i])
            else:
                out.append(bottom[i] - top[i])
        else:
            out.append(top[i] - bottom[i])
    return _join(out[::-1])


def _borrow_no_decrement(a: int, b: int) -> int:
    """Adds 10 when borrowing but never takes 1 from the next column: 52 - 17 = 45."""
    top, bottom = _digits(a)[::-1], _digits(b, len(str(a)))[::-1]
    return _join([t + 10 - u if t < u else t - u for t, u in zip(top, bottom)][::-1])


def _smaller_from_larger(a: int, b: int) -> int:
    """Takes the smaller digit from the larger in every column: 62 - 37 = 35."""
    top, bottom = _digits(a)[::-1], _digits(b, len(str(a)))[::-1]
    return _join([abs(t - u) for t, u in zip(top, bottom)][::-1])


def _stops_borrow_at_zero(a: int, b: int) -> int:
    """Stops when it must borrow from a 0, and takes smaller from larger instead: 402 - 6 = 404."""
    top, bottom, out = _digits(a), _digits(b, len(str(a))), []
    for i in range(len(top) - 1, -1, -1):
        digit = top[i]
        if digit < bottom[i]:
            if i > 0 and top[i - 1] != 0:
                top[i - 1] -= 1
                out.insert(0, digit + 10 - bottom[i])
            else:
                out.insert(0, abs(digit - bottom[i]))
        else:
            out.insert(0, digit - bottom[i])
    return _join(out)


def _top_not_smaller(a: int, b: int) -> bool:
    return a >= b


for _name, _bug in [
    ("buggy_subtraction.borrow_into_1_equals_10", _borrow_into_1_equals_10),
    ("buggy_subtraction.borrow_writes_10", _borrow_writes_10),
    ("buggy_subtraction.diff_0_n_equals_0", _diff_0_n_equals_0),
    ("buggy_subtraction.diff_n_0_equals_0", _diff_n_0_equals_0),
    ("buggy_subtraction.left_ten_ok", _left_ten_ok),
    ("buggy_subtraction.move_over_zero_borrow", _move_over_zero_borrow),
    ("subtraction.always_borrow_left", _always_borrow_left),
    ("subtraction.borrow_from_bottom", _borrow_from_bottom),
    ("subtraction.borrow_no_decrement", _borrow_no_decrement),
    ("subtraction.diff_0_n_equals_n", _diff_0_n_equals_n),
    ("subtraction.smaller_from_larger", _smaller_from_larger),
    ("subtraction.stops_borrow_at_zero", _stops_borrow_at_zero),
]:
    _column_rule(_name, "-", _bug, _top_not_smaller)


def _forget_borrow_over_blanks(a: int, b: int) -> int:
    """With fewer digits on top, does not borrow over the blank columns: 85 - 127 = 158."""
    width = len(str(b))
    top, bottom = _digits(a, width), _digits(b)
    blank = width - len(str(a))
    out, borrow = [], 0
    for i in range(width - 1, -1, -1):
        digit, borrow = top[i] - borrow, 0
        if i < blank:
            digit = max(digit, 0)
            out.append(abs(bottom[i] - digit))
        else:
            if digit < bottom[i]:
                digit += 10
                borrow = 1
            out.append(digit - bottom[i])
    return _join(out[::-1])


_column_rule("buggy_subtraction.forget_borrow_over_blanks", "-", _forget_borrow_over_blanks,
             lambda a, b: len(str(a)) < len(str(b)))


# ---------------------------------------------------------------- multiplication

def _column_mimics_addition(a: int, b: int) -> int:
    """Multiplies column by column like adding, writing the tens of each product: 68 × 46 = 24 (6×4=24 → 2, 8×6=48 → 4)."""
    width = max(len(str(a)), len(str(b)))
    pad = lambda n: [int(d) for d in "1" * (width - len(str(n))) + str(n)]  # noqa: E731  (1 fills, not 0)
    return _join([p // 10 if p >= 10 else p for p in (x * y for x, y in zip(pad(a), pad(b)))])


def _alignment_error_in_multi_digit(a: int, b: int) -> int:
    """Does not shift the partial products: 23 × 14 = 92 + 23 = 115."""
    return sum(a * int(d) for d in str(b))


def _forget_to_add_carried_number(a: int, b: int) -> int:
    """Forgets the carry from the ones column: 23 × 4 = 82 (3 × 4 = 12, the 1 is lost)."""
    ones = b % 10
    first = (a // 10) * ones * 10 + (a % 10) * ones % 10  # the ones' carry is dropped
    return first + a * (b // 10) * 10


_column_rule("buggy_multiplication.column_mimics_addition", "*", _column_mimics_addition,
             lambda a, b: a >= 10 and b >= 10)
_column_rule("multiplication_division.alignment_error_in_multi_digit", "*", _alignment_error_in_multi_digit,
             lambda a, b: b >= 10)
_column_rule("multiplication_division.forget_to_add_carried_number", "*", _forget_to_add_carried_number,
             lambda a, b: a >= 10)
