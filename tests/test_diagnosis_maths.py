import random
import re

import pytest

from matheasy.diagnosis import ParseError, as_fraction, check_step, equivalent, locate, parse


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
    assert (e.topic, e.source, e.wrong, e.expected, e.difference) == ("fractions", "1/2 + 1/3", "2/5", "5/6", "-13/30")


def test_locate_pairs_off_copied_terms():
    e = locate("1/2 + 1/3 + x", "2/5 + x")
    assert (e.source, e.wrong, e.expected) == ("1/2 + 1/3", "2/5", "5/6")
    assert e.topic == "fractions"  # topic of the wrong part, not the whole line


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
    assert (e.topic, e.expected, e.correct, e.difference) == ("decimals", "0.55", "0.55", "-0.27")
    e = locate("50% + 0.2", "0.52")
    assert (e.topic, e.expected) == ("decimals", "0.7")
    e = locate("0.1 ÷ 3", "0.3")                      # not a terminating decimal: stays a fraction
    assert e.expected == "1/30"


def test_locate_mixed_numbers():
    e = locate("1 1/2 + 2 1/4", "3 2/6")
    assert (e.topic, e.source, e.wrong, e.expected) == ("fractions", "1 1/2 + 2 1/4", "3 2/6", "15/4")


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
