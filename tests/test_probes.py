import re

import pytest
import yaml

from matheasy.diagnosis import (
    Candidate, FollowUp, MAX_FOLLOW_UPS, Probe, choose_probe, diagnose, equivalent, parse, probe_items, rule_names, rule_id, run_rule,
    similar_items,
)
from matheasy.diagnosis.explain import TOPICS, rule_topic
from matheasy.diagnosis.probes import BANK
from matheasy.evaluation.items import item_key, overlapping

ENTRIES = yaml.safe_load(BANK.read_text(encoding="utf-8"))
TIE = ("2/3 ÷ 4/5", "8/15")  # two rules predict 8/15 here
COMPARISON_RULES = {"rule:decimals.longer_is_larger", "rule:fractions.natural_number_bias_numerator_only"}


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e["question"])
def test_the_stored_predictions_are_what_the_rules_say(entry):
    question = entry["question"]
    for rule, answer in entry["predictions"].items():
        assert equivalent(run_rule(rule, question), answer)
        assert not equivalent(answer, question)  # a wrong answer
    assert set(entry["topics"]) <= set(TOPICS)
    assert set(entry["topics"]) == {rule_topic(r) for r in entry["predictions"]}


def test_fractions_stay_in_the_agreed_range():
    for item in probe_items():
        for top, bottom in re.findall(r"(?<![\d^.])(\d+)/(\d+)", item):
            assert 2 <= int(bottom) <= 12 and int(top) < int(bottom), item


def test_every_rule_has_items_except_comparisons_for_now():
    covered = {r for e in ENTRIES for r in e["predictions"]}
    assert {rule_id(r) for r in rule_names()} - covered == COMPARISON_RULES


def test_the_list_is_complete_and_works_with_ds_test_check():
    assert len(probe_items()) == len(set(probe_items())) == len(ENTRIES)
    assert all(parse(item) is not None and item_key(item) for item in probe_items())
    assert overlapping(probe_items()[:1], probe_items()) == [(probe_items()[0], probe_items()[0])]


def test_choose_probe_separates_the_suspects_and_skips_what_was_asked():
    suspects = list(diagnose(*TIE).beliefs)
    first = choose_probe(suspects)
    assert first.predictions and first.question in probe_items()
    assert choose_probe(suspects, [first.question]).question != first.question


def test_a_tie_returns_a_probe_and_the_answer_settles_it():
    d = diagnose(*TIE)
    assert d.status == "ambiguous" and d.probe is not None
    rule, answer = next(iter(d.probe.predictions.items()))
    settled = diagnose(*TIE, follow_ups=[FollowUp(d.probe, answer)])
    assert settled.status == "clear" and settled.misconception == rule and settled.probe is None


def test_an_answer_no_rule_predicts_says_nothing():
    answers = []
    for _ in range(MAX_FOLLOW_UPS):
        d = diagnose(*TIE, follow_ups=answers)
        answers.append(FollowUp(d.probe, "1000"))
    last = diagnose(*TIE, follow_ups=answers)
    assert last.status == "ambiguous" and last.probe is None  # no rule wins by elimination


def test_a_right_answer_rules_out_every_rule_predicting_a_wrong_one():
    probe = Probe("1/2 + 1/4", {"rule:fractions.multiply_across_for_division": "2/6",
                                "rule:multiplication_division.division_makes_smaller": "2/6"})
    assert diagnose(*TIE, follow_ups=[FollowUp(probe, "3/4")]).status == "unknown"


def test_a_tie_with_candidates_that_are_not_rules_gets_no_probe():
    tie = lambda error, k=25: [Candidate("rule:fractions.add_numerators_denominators", 0.5), Candidate("eedi:1287", 0.5)]
    d = diagnose("1/2 + 1/3", "2/5", shortlister=tie)
    assert d.status == "ambiguous" and d.probe is None


def test_b2_never_probes():
    assert diagnose(*TIE, mode="b2").probe is None


def test_similar_items_by_topic_skip_what_was_seen():
    items = similar_items(["fractions"], n=3)
    assert len(items) == 3
    assert items[0] not in similar_items(["fractions"], seen=items[:1])
    like_terms = similar_items(["collecting_like_terms"])  # our own rules since T2
    assert like_terms and all(not item.startswith("(") for item in like_terms)  # no (x + 3)^2: that is brackets
    assert similar_items(["other"]) == []  # no rule, so no items
