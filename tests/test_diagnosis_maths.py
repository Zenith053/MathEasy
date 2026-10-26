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


@pytest.mark.parametrize("text", ["", "3 1/2", "25%", "x +", "1 < 2 < 3", "print(1)", "x.y"])
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
