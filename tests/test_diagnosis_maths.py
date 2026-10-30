import random
import re

import pytest
import sympy

from matheasy.diagnosis import ParseError, as_fraction, check_step, equivalent, locate, notation_issue, parse
from matheasy.diagnosis.maths import BinOp, Mixed, Neg, Num, Sym, Word

a, b, c, r, x, y, E, M, C = sympy.symbols("a b c r x y E M C")
R = sympy.Rational


# ---- the meaning table: what each piece of notation MEANS, checked by value.
# Printing tests cannot catch a wrong reading that prints back the same way
# (a/bc read as (a/b)c did), so every notation decision is pinned here.

@pytest.mark.parametrize("text, meaning", [
    # letters written together are one term; a power belongs to the letter before it
    ("a/bc", a / (b * c)),
    ("ab/c", a * b / c),
    ("a/bc^2", a / (b * c**2)),
    ("xy^2", x * y**2),
    ("2xy^2", 2 * x * y**2),
    ("x^2y^3", x**2 * y**3),
    ("(xy)^2", x**2 * y**2),
    ("3ab", 3 * a * b),
    ("3ab^2c", 3 * a * b**2 * c),
    ("-xy", -x * y),
    ("E = MC^2", sympy.Eq(E, M * C**2)),
    # a number written before a letter multiplies like ×, left to right
    ("1/2x", x / 2),
    ("x/2y", x * y / 2),
    ("2x/3", 2 * x / 3),
    ("2 × 22/7 × r", R(44, 7) * r),
    # signs and powers
    ("-x^2", -x**2),
    ("-2^2", -4),
    ("2^3^2", 512),
    ("2^-1", R(1, 2)),
    # fractions, mixed numbers and percentages are single numbers
    ("6 ÷ 2/3", 9),
    ("2/3 ÷ 4/5", R(5, 6)),
    ("12/3/4", 1),
    ("1 - 2 - 3", -4),
    ("3 1/2", R(7, 2)),
    ("1 / 3 1/2", R(2, 7)),
    ("25%", R(1, 4)),
    ("50% of 20", 10),
    # brackets
    ("2(x + 3)", 2 * x + 6),
    ("(x + 1)(x + 2)", (x + 1) * (x + 2)),
    ("x(y + 1)", x * y + x),
    ("2(30 + b) = 100", sympy.Eq(2 * b + 60, 100)),
    # π, and quantity words: a known word, or any word in braces, is one quantity
    ("2πr", 2 * sympy.pi * r),
    ("πr^2", sympy.pi * r**2),
    ("Area = 2πR", sympy.Eq(sympy.Symbol("area"), 2 * sympy.pi * sympy.Symbol("R"))),
    ("AREA + area", 2 * sympy.Symbol("area")),
    ("2Area", 2 * sympy.Symbol("area")),
    ("{fuel} × 3", 3 * sympy.Symbol("fuel")),
    ("xyzw", x * y * sympy.Symbol("z") * sympy.Symbol("w")),  # an unknown run is letters multiplied
    ("a × g × e", a * sympy.Symbol("g") * sympy.Symbol("e")),
])
def test_notation_means(text, meaning):
    got = parse(text).to_sympy()
    if isinstance(meaning, sympy.Equality):
        assert isinstance(got, sympy.Equality)
        assert sympy.simplify(got.lhs - meaning.lhs) == 0 and sympy.simplify(got.rhs - meaning.rhs) == 0
    else:
        assert sympy.simplify(got - meaning) == 0, f"{text} read as {got}, should be {meaning}"


# ---- every standard example in docs/input_notation.md, read by value

A, n = sympy.symbols("A n")


@pytest.mark.parametrize("text, meaning", [
    ("x + 3 = 5", sympy.Eq(x + 3, 5)),
    ("3ab", 3 * a * b), ("2xy", 2 * x * y), ("A = 30 × 20", sympy.Eq(A, 600)),
    ("x^2", x**2), ("2^n", 2**n), ("a^(bc)", a**(b * c)), ("(x + 1)^(xy)", (x + 1)**(x * y)),
    ("yx^2", y * x**2), ("x^2 × y", x**2 * y),
    ("xy^2", x * y**2), ("(xy)^2", x**2 * y**2),
    ("a/(bc)", a / (b * c)), ("x/2", x / 2), ("2x/3", 2 * x / 3), ("1/2 × x", x / 2), ("xy/2", x * y / 2),
    ("x^(2/3)", x**R(2, 3)), ("x^2 ÷ 3", x**2 / 3),
    ("2/3 ÷ 4/5", R(5, 6)), ("6 ÷ 2/3", 9), ("3 1/2", R(7, 2)),
    ("25%", R(1, 4)), ("50% of 20", 10), ("1/2 of 12", 6), ("25% × x", x / 4),
    ("-x", -x), ("6 ÷ -2", -3), ("x - (-3)", x + 3),
    ("A = 30 × 20 − 22/7 × 7^2", sympy.Eq(A, 446)),
])
def test_standard_notation(text, meaning):
    got = parse(text).to_sympy()
    if isinstance(meaning, sympy.Equality):
        assert sympy.simplify(got.lhs - meaning.lhs) == 0 and sympy.simplify(got.rhs - meaning.rhs) == 0
    else:
        assert sympy.simplify(got - meaning) == 0, f"{text} read as {got}, should be {meaning}"
    assert parse(str(parse(text))).to_sympy() == got  # printed back in standard notation, same value


# ---- the meaning round trip: printing a tree and reading it back never changes its value.

def _random_tree(rng: random.Random, depth: int):
    if depth == 0 or rng.random() < 0.3:
        return rng.choice([
            Num(sympy.Integer(n), str(n)) for n in (1, 2, 3, 7, 12)
        ] + [
            BinOp("/", Num(sympy.Integer(1), "1"), Num(sympy.Integer(2), "2")),
            BinOp("/", Num(sympy.Integer(22), "22"), Num(sympy.Integer(7), "7")),
            Mixed(3, 1, 2), Num(R(1, 2), "0.5"), Num(R(1, 4), "25%"),
            Sym("x"), Sym("y"), Sym("a"), Sym("g"), Sym("e"),
            Word("area", "Area"), Word("fuel", "fuel"), Num(sympy.pi, "π"),
        ])
    kind = rng.choice(["+", "-", "*", "*", "/", "^", "neg"])
    if kind == "neg":
        return Neg(_random_tree(rng, depth - 1))
    if kind == "^":
        n = rng.choice([2, 3, "1/2"])
        power = (BinOp("/", Num(sympy.Integer(1), "1"), Num(sympy.Integer(2), "2")) if n == "1/2"
                 else Num(sympy.Integer(n), str(n)))
        return BinOp("^", _random_tree(rng, depth - 1), power)
    return BinOp(kind, _random_tree(rng, depth - 1), _random_tree(rng, depth - 1))


def test_printing_never_changes_meaning():
    rng = random.Random(20261029)
    for _ in range(1500):
        tree = _random_tree(rng, 3)
        if tree.to_sympy().has(sympy.zoo, sympy.nan):  # divides by zero: no value to keep
            continue
        text = str(tree)
        assert sympy.cancel(parse(text).to_sympy() - tree.to_sympy()) == 0, f"{text!r} lost the meaning of {tree!r}"
        assert notation_issue(text) is None, f"printed {text!r} is not standard: {notation_issue(text)}"


@pytest.mark.parametrize("text, shown", [
    ("1/2 + 1/3", "1/2 + 1/3"),          # order kept, not "1/3 + 1/2"
    ("2/3 ÷ 4/5", "2/3 ÷ 4/5"),          # a fraction divided by a fraction
    ("6 ÷ 2/3", "6 ÷ 2/3"),
    ("2/5 × 3/5", "2/5 × 3/5"),
    ("2(x + 3)", "2(x + 3)"),
    ("3ab - 2", "3ab - 2"),
    ("3x^2", "3x^2"),
    ("x(y + 1)", "x(y + 1)"),
    ("(x + 1)(x + 2)", "(x + 1)(x + 2)"),
    ("2 × 3", "2 × 3"),
    ("0.25 > 0.3", "0.25 > 0.3"),
    ("0.30 + 0.5", "0.30 + 0.5"),        # decimals keep how they were written
    ("x + 3 = 5", "x + 3 = 5"),
    ("1/2 - 1/3 + x", "1/2 - 1/3 + x"),
    ("-1/2", "-1/2"),
    ("(x+1)/2", "(x + 1)/2"),
    ("x - (y - 1)", "x - (y - 1)"),
    ("2^3", "2^3"),
    ("5 − 2", "5 - 2"),                  # unicode minus
    ("3 1/2", "3 1/2"),                  # mixed numbers
    ("1 1/2 + 2 3/4", "1 1/2 + 2 3/4"),
    ("-3 1/2", "-(3 1/2)"),
    ("(3 1/2)^2", "(3 1/2)^2"),
    ("2 × 3 1/2", "2 × 3 1/2"),
    ("3 1/2 ÷ 2", "3 1/2 ÷ 2"),
    ("25%", "25%"),                      # percentages
    ("12.5% + 0.5", "12.5% + 0.5"),
    ("2 × 25%", "2 × 25%"),
    ("50% of 20", "50% × 20"),           # "of" after a number means times
    ("1/2 of 12", "1/2 × 12"),
])
def test_parse_keeps_what_the_student_wrote(text, shown):
    node = parse(text)
    assert str(node) == shown
    assert parse(str(node)) == node


def test_parse_reads_fractions_as_units():
    div = parse("2/3 ÷ 4/5")
    assert as_fraction(div.left) == (2, 3) and as_fraction(div.right) == (4, 5)
    assert as_fraction(parse("2/4")) == (2, 4)  # not reduced
    assert as_fraction(parse("0.5/2")) is None
    assert as_fraction(parse("3 1/2")) is None      # a mixed number is not a typed fraction
    assert as_fraction(parse("100%/2")) is None


@pytest.mark.parametrize("text, value", [
    ("3 1/2", "7/2"), ("-3 1/2", "-7/2"), ("25%", "1/4"), ("12.5%", "1/8"), (".5%", "1/200"),
    ("50% of 20", "10"), ("2^50", str(2 ** 50)),
])
def test_parse_values(text, value):
    assert str(parse(text).to_sympy()) == value


@pytest.mark.parametrize("text", [
    "", "x +", "1 < 2 < 3", "print(1)", "x.y",
    "1/2 3/4", "1 12 3/4", "3 1/2x",        # numbers side by side
    "3 1/0",                                # zero denominator in a mixed number
    "25 % 4", "%",                          # % only after a number
    "a_b",                                  # underscores are reserved
    "2^60", "2^3^4", "10^(10^2)",           # exponent above MAX_EXPONENT
    "+".join(["1"] * 200), "-" * 400 + "1",  # longer than MAX_LENGTH
])
def test_parse_errors(text):
    with pytest.raises(ParseError):
        parse(text)


@pytest.mark.parametrize("text", [
    "25%1", "25%12",            # the old parser read these as 251% and 2512%
    "25%25%", "25%3 1/2",       # the old parser crashed with TypeError
    "(1/2) of 1/2",             # the old parser read "of" as the letters o × f
    "x 1/2", "(x + 1) 1/2",     # a fraction cannot follow a letter or bracket with nothing between
    "(" * 101 + "1" + ")" * 101,  # nested deeper than MAX_DEPTH
    "-" * 150 + "1",
])
def test_parse_rejects_what_the_old_parser_misread(text):
    with pytest.raises(ParseError):
        parse(text)


@pytest.mark.parametrize("text, message", [
    ("x ! 2", "unexpected '!' at character 3"),
    ("(x + 1", "missing ) at the end"),
    ("x +", "expected a number, letter or bracket at the end"),
    ("1 < 2 < 3", "only one = < > <= >= per line at character 7"),
    ("25 % 4", "two numbers with nothing between them at character 6"),
    ("2^60", "exponent too large at character 2"),
    ("sqrt(4)", "cannot read sqrt(...)"),
    ("3 1/0", "zero denominator at character 5"),
])
def test_parse_errors_say_what_and_where(text, message):
    with pytest.raises(ParseError, match=re.escape(message)):
        parse(text)


@pytest.mark.parametrize("text, shown", [
    ("1 / 3 1/2", "1 ÷ 3 1/2"),   # a mixed number wins over a fraction
    ("6 ÷ 3 1/2", "6 ÷ 3 1/2"),
    ("x = 5.", "x = 5."),          # a full stop after an answer
    ("2^3^2", "2^3^2"),            # powers group right to left
    ("-x^2", "-x^2"),
])
def test_parse_notation(text, shown):
    node = parse(text)
    assert str(node) == shown
    assert parse(str(node)) == node


def test_parse_groups_like_maths():
    assert parse("2^3^2").to_sympy() == 512        # 2^(3^2)
    assert parse("-2^2").to_sympy() == -4          # -(2^2)
    assert parse("12/3/4").to_sympy() == 1         # (12/3)/4
    assert parse("1 - 2 - 3").to_sympy() == -4     # (1 - 2) - 3
    assert parse("6/3 × 2").to_sympy() == 4


def test_parse_never_crashes_on_garbage():
    rng = random.Random(20261029)
    alphabet = "0123456789 .+-*/^()=<>%xyab×÷−≤_!,of"
    for _ in range(3000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 25)))
        try:
            parse(text)
        except ParseError:
            pass


@pytest.mark.parametrize("a, b, same", [
    ("1/2 + 1/3", "5/6", True),
    ("1/2 + 1/3", "2/5", False),
    ("2/4", "1/2", True),
    ("0.25", "1/4", True),
    ("0.1 + 0.2", "0.3", True),          # exact, no float error
    ("2(x + 3)", "2x + 6", True),
    ("2(x + 3)", "2x + 3", False),
    ("x + 3 = 5", "x = 2", True),
    ("x + 3 = 5", "x = 8", False),
    ("x = 2", "2 = x", True),
    ("x < 2", "2 > x", True),
    ("x < 2", "-x < -2", False),
    ("0.25 < 0.3", "0.3 > 1/4", True),
    ("1 < 2", "3 < 5", False),           # both true, but not the same statement
    ("x = 2", "2", False),
    ("3 1/2", "7/2", True),
    ("3 1/2", "3.5", True),
    ("1 1/2 + 1 1/2", "3", True),
    ("3 1/2", "3/2", False),
    ("25%", "0.25", True),
    ("25%", "1/4", True),
    ("25%", "25", False),                # percent_equals_decimal must stay detectable
    ("50% of 20", "10", True),
    ("0.3 > 25%", "1/4 < 0.3", True),
])
def test_equivalent(a, b, same):
    assert equivalent(a, b) is same
    assert check_step(a, b) is same


def test_locate_correct_step_is_none():
    assert locate("1/2 + 1/3", "5/6") is None
    assert locate("2(x + 3)", "2x + 6") is None


def test_locate_whole_answer():
    e = locate("1/2 + 1/3", "2/5")
    assert (e.source, e.wrong, e.expected, e.difference) == ("1/2 + 1/3", "2/5", "5/6", "-13/30")


def test_locate_pairs_off_copied_terms():
    e = locate("1/2 + 1/3 + x", "2/5 + x")
    assert (e.source, e.wrong, e.expected) == ("1/2 + 1/3", "2/5", "5/6")


def test_locate_pairs_off_correctly_changed_terms():
    e = locate("1/2 + 1/3 + 2 × 3", "2/5 + 6")
    assert (e.source, e.wrong) == ("1/2 + 1/3", "2/5")


def test_locate_equation_side():
    e = locate("x + 1/2 + 1/3 = 7", "x + 2/5 = 7")
    assert (e.source, e.wrong, e.expected) == ("1/2 + 1/3", "2/5", "5/6")
    e = locate("x + 3 = 5", "x = 8")
    assert (e.source, e.wrong, e.expected) == ("x + 3 = 5", "x = 8", "x = 2")


def test_locate_record_is_json_safe():
    import json
    d = locate("1/2 + 1/3", "2/5").to_dict()
    assert "source_node" not in d
    json.dumps(d)


def test_locate_decimals_report_decimals():
    e = locate("0.3 + 0.25", "0.28")
    assert (e.expected, e.correct, e.difference) == ("0.55", "0.55", "-0.27")
    e = locate("50% + 0.2", "0.52")
    assert e.expected == "0.7"
    e = locate("0.1 ÷ 3", "0.3")                      # not a terminating decimal: stays a fraction
    assert e.expected == "1/30"
    e = locate("0.5x + 0.25x", "0.7x")                # decimals with letters stay decimals
    assert (e.expected, e.difference) == ("0.75x", "-0.05x")


def test_locate_mixed_numbers():
    e = locate("1 1/2 + 2 1/4", "3 2/6")
    assert (e.source, e.wrong, e.expected) == ("1 1/2 + 2 1/4", "3 2/6", "15/4")


# ---- narrowing to the changed part

@pytest.mark.parametrize("before, after, source, wrong", [
    ("2 × (1/2 + 1/3)", "2 × 2/5", "1/2 + 1/3", "2/5"),          # inside a product
    ("x + 2(1/2 + 1/3)", "x + 2 × 2/5", "1/2 + 1/3", "2/5"),     # a copied term, then a product
    ("(1/2 + 1/3)^2", "(2/5)^2", "1/2 + 1/3", "2/5"),            # inside a power
    ("2(x+3) + 4x", "2x + 3 + 4x", "2(x + 3)", "2x + 3"),
    ("2(x + 3) = 10", "2x + 3 = 10", "2(x + 3)", "2x + 3"),      # one side of an equation
    ("6/8", "3/5", "6/8", "3/5"),                                # never inside a fraction
])
def test_locate_narrows_to_the_changed_part(before, after, source, wrong):
    e = locate(before, after)
    assert (e.source, e.wrong) == (source, wrong)


# ---- the exact wrong term

@pytest.mark.parametrize("before, after, wrong_term, expected_term", [
    # expressions
    ("1/2 + 1/3", "2/5", "2/5", "5/6"),
    ("2(x + 3)", "2x + 3", "3", "6"),                 # multiplied only the first term
    ("3(x - 1)", "3x - 1", "-1", "-3"),
    ("2(x + 3)", "2x + 8", "8", "6"),
    ("3x + 2x", "6x", "6x", "5x"),
    ("3x + 2x + 4", "5x + 6", "6", "4"),
    ("-(x - 2)", "-x - 2", "-2", "2"),
    ("x(x + 2)", "x^2 + 2", "2", "2x"),               # an extra and a missing term become one pair
    ("2(x + 3)", "2x", "(nothing)", "3"),             # a term left out
    ("0.3 + 0.25", "0.28", "0.28", "0.55"),
    # equations: terms moved, sides multiplied or divided
    ("x + 3 = 5", "x = 5 + 3", "3", "-3"),            # moved without changing sign
    ("x + 3 = 5", "x = 8", "8", "2"),
    ("x - 4 = 6", "x = 2", "2", "10"),
    ("5 = x + 3", "x = 8", "8", "2"),                 # sides swapped
    ("2x + 4 = 10", "x + 4 = 5", "4", "2"),           # divided one term only
    ("2x = 6", "x = 12", "12", "3"),                  # multiplied instead of divided
    ("3x = 12", "x = 9", "9", "4"),
    ("x/2 = 3", "x = 3/2", "3/2", "6"),
    ("2x + 3 = 11", "2x = 14", "14", "8"),
    ("4x - 2 = 2x + 6", "2x = 4", "4", "8"),
    ("x + 1 < 5", "x < 6", "6", "4"),
])
def test_locate_finds_the_wrong_term(before, after, wrong_term, expected_term):
    e = locate(before, after)
    assert (e.wrong_term, e.expected_term) == (wrong_term, expected_term)


def test_expected_values_read_like_student_maths():
    e = locate("2(x + 3)", "2x + 3")
    assert (e.expected, e.correct) == ("2x + 6", "2x + 6")
    for text in (e.expected, e.correct, e.expected_term):
        parse(text)  # every reported value parses back
