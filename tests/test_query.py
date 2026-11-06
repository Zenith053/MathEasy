"""build_query() and its fallback. Pure functions: no data files needed."""

import sympy

from matheasy.shortlister.query import UNPARSED, build_query, clean, fallback_diff, parse, text_diff


def test_query_has_all_six_fields_in_order():
    q = build_query("fractions - add fractions", r"\frac{1}{2}+\frac{1}{3}", r"\frac{2}{5}", r"\frac{5}{6}")
    names = [part.split(":")[0] for part in q.split(" | ")]
    assert names == ["Topic", "Before", "Student wrote", "Correct", "Changed part", "Difference"]


def test_locate_output_is_used_when_given():
    q = build_query("decimals", "0.25 vs 0.3", "0.25", "0.3", changed="0.3 -> 0.25", difference="chose the longer")
    assert q.endswith("Changed part: 0.3 -> 0.25 | Difference: chose the longer")


def test_clean_strips_latex_delimiters_colour_and_images():
    assert clean(r"\( 3 \times 2 \)") == r"3 \times 2"
    assert clean(r"\color{green}\bigcirc") == r"\bigcirc"
    assert clean("![A pie chart]()") == "[image: A pie chart]"


def test_parse_numbers_fractions_and_expressions():
    assert parse(r"\( -20 \)") == -20
    assert parse(r"\frac{2}{3}") == sympy.Rational(2, 3)
    x = sympy.Symbol("x")
    assert sympy.simplify(parse("2 x+6") - (2 * x + 6)) == 0


def test_parse_decimals_are_exact():
    assert parse("0.3") - parse("0.25") == sympy.Rational(1, 20)


def test_parse_mixed_number():
    # The raw parser reads "1 \frac{1}{2}" as 1/2.
    assert parse(r"1 \frac{1}{2}") == sympy.Rational(3, 2)


def test_parse_strips_units_and_thousands_commas():
    assert parse("£ 144") == 144
    assert parse(r"20 \%") == 20
    assert parse("300,000") == 300000
    assert parse(r"\bigcirc = -9") == -9


def test_parse_rejects_words():
    # Would otherwise parse as V*a*r*i*a*b*l*e.
    assert parse("Variable") is None
    assert parse("Both Tom and Katie") is None


def test_text_diff_reads_swaps_position_by_position():
    assert text_diff(r"\frac{2}{5}", r"\frac{5}{6}") == "5 -> 2; 6 -> 5"
    assert text_diff("6 p-2", "6 p-12") == "12 -> 2"
    assert text_diff("same", "same") == "(same text)"


def test_fallback_difference_when_both_parse():
    _, diff = fallback_diff("2x+3", "2x+6")
    assert diff == "student - correct = -3"
    _, diff = fallback_diff("0.0003", "0.003")
    assert "student / correct = 1/10" in diff


def test_fallback_equivalent_answer():
    _, diff = fallback_diff(r"1 \frac{1}{2}", r"\frac{3}{2}")
    assert diff == "equivalent to the correct answer"


def test_fallback_unparsed_answers_still_get_a_text_diff():
    changed, diff = fallback_diff("Equation", "Variable")
    assert changed == "Variable -> Equation"
    assert diff == UNPARSED


def test_list_answers_do_not_crash():
    # A parsed tuple cannot be subtracted; must fall back, not raise.
    _, diff = fallback_diff("3, 5", "5, 3")
    assert diff == UNPARSED


def test_eedi_query_uses_bare_topic_like_live_tutor():
    # B's LocatedError.topic is only fractions|decimals|expressions.
    from matheasy.shortlister.query import eedi_query
    row = {"Topic": "fractions", "ConstructName": "Add proper fractions", "QuestionText": r"\frac{1}{2}+\frac{1}{3}",
           "WrongAnswerText": r"\frac{2}{5}", "CorrectAnswerText": r"\frac{5}{6}"}
    assert eedi_query(row).startswith("Topic: fractions | Before:")


def test_llm_examples_become_training_queries_and_other_topics_are_skipped():
    from matheasy.llm import FakeBackend, LLMClient
    from matheasy.shortlister.llm_examples import write_examples
    reply = ('{"items": [{"id": 4, "topic": "fractions", "before": "1/2 + 1/3", "student": "2/5", "correct": "5/6"},'
             ' {"id": 9, "topic": "other", "before": "", "student": "", "correct": ""},'
             ' {"id": 77, "topic": "decimals", "before": "x", "student": "y", "correct": "z"}]}')
    session = LLMClient(backend=FakeBackend(lambda req: reply)).session("t", "data", budget_tokens=None)
    out = write_examples(session, [(4, "Adds the numerators and adds the denominators"),
                                   (9, "Confuses the radius and the diameter"), (12, "left out by the model")])
    assert out[4]["Query"].startswith("Topic: fractions | Before: 1/2 + 1/3 | Student wrote: 2/5 | Correct: 5/6")
    assert out[9] is None
    assert 12 not in out and 77 not in out  # missing: asked again on rerun; never asked: ignored


def test_rule_pairs_use_live_query_format_and_need_an_eedi_mapping():
    from dataclasses import replace
    from matheasy.diagnosis import load_bank
    from matheasy.diagnosis.bank import Bank
    from matheasy.shortlister.rule_pairs import build
    bank = Bank([replace(m, eedi=()) for m in load_bank()])
    rid = "rule:fractions.add_numerators_denominators"
    pairs, skipped = build(bank)
    assert pairs.empty and rid in skipped          # no eedi: mapping, no pairs
    mapped = Bank([replace(m, eedi=(1287, 99)) if m.id == rid else m for m in bank])
    pairs, _ = build(mapped)
    assert set(pairs.RuleId) == {rid} and set(pairs.MisconceptionId) == {1287}  # first ID only
    assert len(pairs) == 30 and pairs.Before.is_unique
    assert all(q.startswith("Topic: fractions | Before: ") and "Changed part: " in q for q in pairs.Query)


def test_to_notation_turns_latex_into_our_notation():
    from matheasy.shortlister.live_queries import to_notation

    assert to_notation(r"\frac{1}{2}+\frac{1}{3}") == "1/2+1/3"
    assert to_notation(r"3 \frac{1}{2} \times 4") == "3 1/2 × 4"
    assert to_notation(r"\frac{x+1}{2} \div 3") == "(x+1)/(2) ÷ 3"
    assert to_notation(r"x^{2} + 300,000") == "x^(2) + 300000"


def test_live_query_matches_the_tutors_query():
    from matheasy.diagnosis import locate
    from matheasy.diagnosis.interfaces import query
    from matheasy.shortlister.live_queries import live_query

    q = live_query(r"Work out \( \frac{1}{2}+\frac{1}{3} \)", r"\( \frac{2}{5} \)")
    assert q == query(locate("1/2+1/3", "2/5"))
    assert q.startswith("Topic: fractions | Before: 1/2 + 1/3 | Student wrote: 2/5")


def test_live_query_skips_rows_with_no_step_to_locate():
    from matheasy.shortlister.live_queries import live_query

    assert live_query("Which of these is a prime number?", r"\( 9 \)") is None  # no maths part
    assert live_query(r"Work out \( \frac{1}{2}+\frac{1}{3} \)", "Only Tom") is None  # answer is words
    assert live_query(r"Work out \( \sqrt{4} \)", r"\( 3 \)") is None  # maths part we cannot read
