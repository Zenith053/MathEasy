import pytest
import yaml

from matheasy.diagnosis import (
    SLIP, HeldBackRuleError, diagnose, eedi_id, held_back_rules, load_bank, rule_id, rule_names, run_rule, split_id,
)
from matheasy.diagnosis.rules import rule


# ---- ID scheme

def test_ids():
    assert rule_id("fractions.add_numerators_denominators") == "rule:fractions.add_numerators_denominators"
    assert eedi_id(1287) == "eedi:1287"
    assert split_id("rule:fractions.x") == ("rule", "fractions.x")
    assert split_id("eedi:12") == ("eedi", "12")
    assert split_id(SLIP) == ("slip", "")
    for bad in ["fractions.x", "rule:Fractions", "eedi:abc", "Slip"]:
        with pytest.raises(ValueError):
            split_id(bad)


# ---- rules: MalruleLib's own examples, then the student's own expressions

@pytest.mark.parametrize("name, problem, answer", [
    ("fractions.add_numerators_denominators", "1/2 + 1/3", "2/5"),
    ("fractions.add_numerators_denominators", "1/4 + 1/6", "1/5"),        # simplified, as MalruleLib does
    ("fractions.add_numerators_denominators", "1/2 + 1/3 + 1/4", "1/3"),      # 3/9, simplified
    ("fractions.keep_common_denominator_for_multiplication", "2/5 × 3/5", "6/5"),
    ("fractions.keep_common_denominator_for_multiplication", "3/4 * 2/4", "3/2"),
    ("fractions.multiply_across_for_division", "2/3 ÷ 4/5", "8/15"),
    ("fractions.multiply_across_for_division", "1/2 ÷ 1/4", "1/8"),
])
def test_rules_predict_the_wrong_answer(name, problem, answer):
    assert run_rule(name, problem) == answer
    assert run_rule(rule_id(name), problem) == answer


@pytest.mark.parametrize("name, problem", [
    ("fractions.add_numerators_denominators", "2x + 3"),
    ("fractions.add_numerators_denominators", "1/2 - 1/3"),
    ("fractions.add_numerators_denominators", "1/2"),
    ("fractions.add_numerators_denominators", "1/0 + 1/0"),
    ("fractions.keep_common_denominator_for_multiplication", "2/5 × 3/7"),   # different denominators
    ("fractions.multiply_across_for_division", "6 ÷ 2/3"),                   # not fraction ÷ fraction
])
def test_rules_do_not_apply(name, problem):
    assert run_rule(name, problem) is None


def test_unknown_rule():
    with pytest.raises(KeyError):
        run_rule("fractions.no_such_rule", "1/2 + 1/3")


# ---- held-back rules never get in

def test_no_held_back_rule_is_registered():
    held = held_back_rules()
    assert len(held) == 10
    assert not held & set(rule_names())


def test_registering_a_held_back_rule_fails():
    with pytest.raises(HeldBackRuleError):
        rule("fractions.subtract_across", topic="fractions")(lambda node: None)


def test_bank_rejects_held_back_rule(tmp_path):
    path = tmp_path / "bank.yaml"
    path.write_text(yaml.safe_dump({"misconceptions": {
        "rule:fractions.subtract_across": {"topic": "fractions", "text": "x", "eedi": []}}}))
    with pytest.raises(HeldBackRuleError):
        load_bank(path)


# ---- bank

def test_bank_matches_rules():
    bank = load_bank()
    assert SLIP in bank
    assert sorted(bank.rules()) == sorted(rule_id(n) for n in rule_names())
    for m in bank:
        split_id(m.id)
        assert m.text


def test_bank_eedi_mapping(tmp_path):
    path = tmp_path / "bank.yaml"
    path.write_text(yaml.safe_dump({"misconceptions": {
        "rule:fractions.add_numerators_denominators": {"topic": "fractions", "text": "x", "eedi": [1287]},
        "eedi:99": {"topic": "fractions", "text": "y"}}}))
    bank = load_bank(path)
    assert bank.for_eedi(1287) == ["rule:fractions.add_numerators_denominators"]
    assert bank["eedi:99"].rule is None


def test_bank_entry_without_a_rule_fails(tmp_path):
    path = tmp_path / "bank.yaml"
    path.write_text(yaml.safe_dump({"misconceptions": {
        "rule:fractions.ignore_denominators": {"topic": "fractions", "text": "x"}}}))
    with pytest.raises(KeyError):
        load_bank(path)
