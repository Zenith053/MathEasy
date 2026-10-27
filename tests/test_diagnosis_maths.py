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
