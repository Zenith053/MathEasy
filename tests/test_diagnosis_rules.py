import pytest
import yaml

from matheasy.diagnosis import (
    SLIP, HeldBackRuleError, RuleError, RuleShortlister, diagnose, eedi_id, held_back_rules, load_bank, rule_id,
    rule_names, run_rule, split_id,
)
from matheasy.diagnosis import rules
from matheasy.diagnosis.bank import Bank, Misconception
from matheasy.diagnosis.rules import rule


# ---- ID scheme

def test_ids():
    assert rule_id("fractions.add_numerators_denominators") == "rule:fractions.add_numerators_denominators"
    assert eedi_id(1287) == "eedi:1287"
    assert split_id("rule:fractions.x") == ("rule", "fractions.x")
    assert split_id("eedi:12") == ("eedi", "12")
    assert split_id(SLIP) == ("slip", "")
    for bad in ["fractions.x", "rule:Fractions", "eedi:abc", "Slip",
                "eedi:007", "eedi:-5", "eedi:", "eedi:٣"]:  # leading zeros, negatives, non-ASCII digits
        with pytest.raises(ValueError):
            split_id(bad)


def test_eedi_id_accepts_only_whole_numbers():
    import numpy as np
    assert eedi_id(0) == "eedi:0"
    assert eedi_id(np.int64(1287)) == "eedi:1287"  # pandas columns give numpy integers
    for bad in [-5, 3.7, True, "12", None]:
        with pytest.raises(ValueError):
            eedi_id(bad)


def test_every_built_id_splits_back():
    for mid in [rule_id("fractions.add_numerators_denominators"), eedi_id(0), eedi_id(1287), SLIP]:
        split_id(mid)


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


def test_bad_rule_name_is_refused():
    for bad in ["Fractions.add", "fractions", "fractions.add-across"]:
        with pytest.raises(ValueError):
            rule(bad)


# ---- a rule that crashes

CRASHING = "fractions.crashes_for_test"


@pytest.fixture
def crashing_rule(monkeypatch):
    def crash(node):
        raise ZeroDivisionError("division by zero")
    monkeypatch.setitem(rules._REGISTRY, CRASHING, rules.Rule(CRASHING, crash))
    return rule_id(CRASHING)


def test_crashing_rule_raises_rule_error(crashing_rule):
    with pytest.raises(RuleError, match=f"rule {CRASHING} failed on 1/2 \\+ 1/3: ZeroDivisionError"):
        run_rule(crashing_rule, "1/2 + 1/3")


def test_diagnosis_survives_a_crashing_rule(crashing_rule):
    bank = Bank([Misconception(crashing_rule, "crashes", ()),
                 Misconception(rule_id("fractions.add_numerators_denominators"), "adds across", ())])
    d = diagnose("1/2 + 1/3", "2/5", shortlister=RuleShortlister(bank))
    assert d.misconception == "rule:fractions.add_numerators_denominators"
    failed = d.to_log()["trace"][1]["failed_rules"]
    assert len(failed) == 1 and failed[0].startswith(f"rule {CRASHING} failed")


# ---- held-back rules never get in

def test_no_held_back_rule_is_registered():
    held = held_back_rules()
    assert len(held) == 10
    assert not held & set(rule_names())


def test_registering_a_held_back_rule_fails():
    with pytest.raises(HeldBackRuleError):
        rule("fractions.subtract_across")(lambda node: None)


def test_bank_rejects_held_back_rule(tmp_path):
    path = tmp_path / "bank.yaml"
    path.write_text(yaml.safe_dump({"misconceptions": {
        "rule:fractions.subtract_across": {"text": "x", "eedi": []}}}))
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
        "rule:fractions.add_numerators_denominators": {"text": "x", "eedi": [1287]},
        "eedi:99": {"text": "y"}}}))
    bank = load_bank(path)
    assert bank.for_eedi(1287) == ["rule:fractions.add_numerators_denominators"]
    assert bank["eedi:99"].rule is None


def test_bank_entry_without_a_rule_fails(tmp_path):
    path = tmp_path / "bank.yaml"
    path.write_text(yaml.safe_dump({"misconceptions": {
        "rule:fractions.ignore_denominators": {"text": "x"}}}))
    with pytest.raises(KeyError):
        load_bank(path)
