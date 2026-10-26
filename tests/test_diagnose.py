import json

import pytest

from matheasy.diagnosis import (
    MAX_FOLLOW_UPS, Candidate, FollowUp, ParseError, Probe, StepIsCorrect, check_step, diagnose,
)


def test_end_to_end_fraction_example():
    # The week-1 check: a student who adds tops and bottoms is diagnosed.
    assert not check_step("1/2 + 1/3", "2/5")
    d = diagnose("1/2 + 1/3", "2/5", student_id="student-001")
    assert d.status == "clear"
    assert d.misconception == "rule:fractions.add_numerators_denominators"
    assert d.probe is None
    assert d.located.expected == "5/6"


def test_inside_a_longer_line():
    d = diagnose("x + 2/3 ÷ 4/5", "x + 8/15")
    assert d.misconception == "rule:fractions.multiply_across_for_division"


def test_no_rule_fits_is_unknown():
    d = diagnose("1/2 + 1/3", "7/9")
    assert (d.status, d.misconception, d.beliefs) == ("unknown", None, {})


def test_tied_candidates_are_ambiguous():
    tie = lambda error, k=25: [Candidate("rule:fractions.add_numerators_denominators", 0.4),
                               Candidate("eedi:1287", 0.4), Candidate("slip", 0.2)]
    d = diagnose("1/2 + 1/3", "2/5", shortlister=tie)
    assert d.status == "ambiguous" and d.misconception is None
    assert d.beliefs == pytest.approx({"rule:fractions.add_numerators_denominators": 0.4, "eedi:1287": 0.4,
                                       "slip": 0.2})


def test_uses_the_given_shortlister():
    seen = []

    def shortlist(error, k=25):
        seen.append((error.source, error.wrong, k))
        return [Candidate("eedi:5", 2.0), Candidate("eedi:6", 6.0)]

    d = diagnose("1/2 + 1/3", "2/5", shortlister=shortlist, k=10)
    assert seen == [("1/2 + 1/3", "2/5", 10)]
    assert d.misconception == "eedi:6"
    assert list(d.beliefs) == ["eedi:6", "eedi:5"]
    assert d.beliefs["eedi:6"] == pytest.approx(0.75)


def test_b2_mode():
    d = diagnose("1/2 + 1/3", "2/5", mode="b2")
    assert d.mode == "b2" and d.status == "clear"
    with pytest.raises(ValueError):
        diagnose("1/2 + 1/3", "2/5", mode="b9")


def test_errors():
    with pytest.raises(StepIsCorrect):
        diagnose("1/2 + 1/3", "5/6")
    with pytest.raises(ParseError):
        diagnose("1/2 + 1/3", "2/5 +")
    probe = Probe("1/3 + 1/3", {})
    with pytest.raises(ValueError):
        diagnose("1/2 + 1/3", "2/5", follow_ups=[FollowUp(probe, "2/6")] * (MAX_FOLLOW_UPS + 1))


def test_log_record():
    log = diagnose("1/2 + 1/3", "2/5").to_log()
    json.dumps(log)
    assert [step["stage"] for step in log["trace"]] == ["locate", "shortlist", "decide"]
    assert log["located"]["wrong"] == "2/5"
