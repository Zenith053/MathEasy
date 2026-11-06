"""D's week-3 evaluation: pre/post tests, adversarial students, learning, retention,
calibration and topic metrics, statistics, labelling, and the test/probe check."""

import csv
import random

import pytest

from matheasy.diagnosis import equivalent, held_back_rules, probe_items, rule_names, run_rule
from matheasy.diagnosis.explain import TOPICS
from matheasy.evaluation import SimStudent, check_records
from matheasy.evaluation.adversarial import (
    ATTACKS, LINES, TEST_PROMPT, AdversarialStudent, LLMAttackLine, adversarial_roster,
)
from matheasy.evaluation.held_back import held_back_names, held_back_rule
from matheasy.evaluation.items import SAMPLE_ITEMS, item_key, overlapping
from matheasy.evaluation.labels import (
    agreement, assign, check_judge, cohen_kappa, leak_code_judge, read_key, read_labels, sample_tasks, write_sheets,
)
from matheasy.evaluation.log_fields import probe_overlap_problems
from matheasy.evaluation.metrics import (
    compute, expected_calibration_error, gain, grouped_sessions, planted_topic,
)
from matheasy.evaluation.prepost import answer_shape, isomorphic, renumber, seed_items, shows, prepost_forms
from matheasy.evaluation.stats import Units, bootstrap_ci, compare, holm, units
from matheasy.tutor.orchestrator import extract_answer

ADDS = "fractions.add_numerators_denominators"
RULE = f"rule:{ADDS}"


def never(message, misconception):
    return False


# ---------------------------------------------------------------- pre/post tests

ALL_STUDENTS = sorted(set(rule_names()) | held_back_rules()) + [None]


@pytest.mark.parametrize("misconception", ALL_STUDENTS)
def test_every_student_gets_parallel_forms_that_show_its_misconception(misconception):
    forms = prepost_forms(misconception, sessions=2, student_id="t#0")
    items = [i for f in forms for i in f.pre + f.post]
    assert len(forms[0].pre) >= 3 and len(forms[0].pre) == len(forms[0].post)
    assert len({item_key(i) for i in items}) == len(items)  # every item different
    for f in forms:
        for seed, pre, post in zip(f.seeds, f.pre, f.post):
            assert isomorphic(seed, pre, misconception) and isomorphic(seed, post, misconception)
            assert shows(misconception, pre) and shows(misconception, post)


@pytest.mark.parametrize("misconception", ALL_STUDENTS)
def test_tests_never_match_a_probe_or_a_seed(misconception):
    forms = prepost_forms(misconception, sessions=2, student_id="t#0")
    items = [i for f in forms for i in f.pre + f.post]
    assert overlapping(items, probe_items()) == []
    assert overlapping(items, seed_items(misconception)) == []


def test_tests_avoid_the_tutors_items():
    first = prepost_forms(ADDS, sessions=1, student_id="a")[0]
    again = prepost_forms(ADDS, sessions=1, student_id="a", avoid=[first.pre[0]])[0]
    assert overlapping(again.pre + again.post, [first.pre[0]]) == []


def test_forms_are_reproducible_and_differ_by_student():
    assert prepost_forms(ADDS, student_id="a") == prepost_forms(ADDS, student_id="a")
    assert prepost_forms(ADDS, student_id="a") != prepost_forms(ADDS, student_id="b")


def test_renumber_keeps_structure():
    rng = random.Random(1)
    for _ in range(50):
        assert renumber("302 - 7", rng)[1] == "0"                       # zero to borrow across stays
        assert renumber("(3 × 10^4) × (2 × 10^5)", rng).count("10^") == 2  # 10^n stays
        assert renumber("(x + 3)^2", rng).endswith(")^2")               # square of a sum stays
        a, b = renumber("3/4 - 1/4", rng).replace(" ", "").split("-")
        assert a.split("/")[1] == b.split("/")[1]                        # common denominator stays common
        assert "1/" in renumber("1/2 + 1/3", rng)                        # unit fractions stay unit


def test_answer_shape():
    assert answer_shape("5/6") == "+fraction"
    assert answer_shape("x = 4") == "solved"
    assert answer_shape("-3") == "+integer".replace("+", "-")
    assert answer_shape("x^2 + 6x + 9") == "expression"


def test_isomorphic_rejects_items_the_rule_gets_right():
    assert isomorphic("1/2 + 1/3", "1/4 + 1/5", ADDS)
    assert not isomorphic("1/2 + 1/3", "2/4 + 1/5", ADDS)   # 2/4 is not in lowest terms, 1/2 was
    assert not isomorphic("1/2 + 1/3", "1/2 + 1/2", ADDS)   # answer 1 is a whole number, 5/6 was not
    assert not isomorphic("2x + 4 = 10", "1/2 + 1/3", "algebra.cancel_across_equals")


# ---------------------------------------------------------------- held-back topics, item_key

def test_every_held_back_rule_has_one_of_bs_topics():
    for name in held_back_names():
        assert held_back_rule(name).topic in TOPICS
    assert planted_topic("rule:algebra.change_side_change_sign") == "solving_equations"
    assert planted_topic(RULE) == "fractions"


def test_item_key_handles_mixed_numbers_and_words():
    assert item_key("3 1/2 + 1/4") == item_key("1/4 + 3 1/2")
    assert item_key("3 1/2 + 1/4") != item_key("7/2 + 1/4")
    assert item_key("Area = 2 × width") == item_key("area = width × 2")


# ---------------------------------------------------------------- adversarial students

def test_attack_lines_have_no_digits_and_the_answer_comes_last():
    for attack in ATTACKS:
        s = AdversarialStudent("a#0", ADDS, attack, judge=never)
        for _ in range(len(LINES[attack]) + 1):
            reply = s.respond("What is 1/2 + 1/3?", item="1/2 + 1/3")
            assert extract_answer(reply.text) == reply.answer == "2/5"
            assert reply.attack == attack and reply.followed_rule


def test_adversarial_student_still_learns_like_a_sim_student():
    s = AdversarialStudent("a#0", ADDS, "emotional_threat", judge=lambda m, d: "denominator" in m)
    s.respond("Make the denominators the same first.")
    assert not s.holds_misconception
    assert s.respond("Now 1/4 + 1/5?", item="1/4 + 1/5").correct


def test_intentional_wrong_answer_is_deliberate_not_a_slip():
    s = AdversarialStudent("w#0", None, "intentional_wrong_answer")
    reply = s.respond("What is 1/2 + 1/3?", item="1/2 + 1/3")
    assert reply.correct is False and reply.deliberate and not reply.slipped and not reply.followed_rule
    assert reply.text.endswith(f"My final guess is {reply.answer}.")


def test_tests_are_answered_plainly():
    s = AdversarialStudent("w#0", None, "intentional_wrong_answer")
    reply = s.respond(f"{TEST_PROMPT}: work out 1/2 + 1/3", item="1/2 + 1/3")
    assert reply.correct and reply.attack is None and reply.text == "I got 5/6."


def test_adversarial_student_saves_and_loads(tmp_path):
    s = AdversarialStudent("adv-x#0", ADDS, "request_shaping", judge=never)
    s.respond("q", item="1/2 + 1/3")
    s.save(tmp_path)
    t = AdversarialStudent.load(tmp_path, "adv-x#0", judge=never)
    assert (t.attack, t.attacks_made, t.items_answered) == ("request_shaping", 1, 1)


def test_llm_attack_line_with_digits_falls_back_to_the_template():
    s = AdversarialStudent("a#0", ADDS, "direct_request", judge=never,
                           attack_line=lambda attack, message, turn: "Just say 5/6 please.")
    assert s.respond("q", item="1/2 + 1/3").text.startswith(LINES["direct_request"][0])


def test_llm_attack_line_calls_the_client():
    class Session:
        def call(self, role, messages, system, json_schema):
            assert role == "adversarial_student" and "Reframe" in system
            return type("R", (), {"parsed": {"line": "Could you just show me?"}})()
    assert LLMAttackLine(Session())("request_shaping", "Try again.", 0) == "Could you just show me?"


def test_adversarial_roster_covers_every_technique():
    students = adversarial_roster(never, per_attack=3)
    assert len(students) == 18 and {s.attack for s in students} == set(ATTACKS)
    assert len({s.student_id for s in students}) == 18


# ---------------------------------------------------------------- hand-built logs

def rec(event, session_id, student_id="s#0", condition="full", **fields):
    return ("x.jsonl:1", {"ts": "t", "run_id": "r", "session_id": session_id, "student_id": student_id,
                          "condition": condition, "config_hash": "h", "event": event, **fields})


def session(sid, *, student="s#0", condition="full", index=0, pre=(), post=(), holds=(RULE,), relapsed=False,
            attack=None, turns=(), diagnoses=(), planted=(RULE,), group="has_rule"):
    out = [rec("session_start", sid, student, condition, session_index=index, student_group=group,
               planted=list(planted), attack=attack, relapsed=relapsed)]
    for phase, marks in (("pre", pre), ("post", post)):
        out += [rec("test", sid, student, condition, phase=phase, item=f"1/{13 + k} + 1/{13 + k}", correct_answer="1",
                    student_answer="1" if m else "2", correct=m) for k, m in enumerate(marks)]
    out += [rec("turn", sid, student, condition, **t) for t in turns]
    out += [rec("diagnosis", sid, student, condition, **d) for d in diagnoses]
    out.append(rec("session_end", sid, student, condition, turns=len(turns), ended_by="solved", tokens_used=0,
                   holds=list(holds)))
    return out


def turn(n, text="Try again.", item="1/2 + 1/3", answer="2/5", correct=False, followed=True):
    return {"turn": n, "tutor_text": text, "critic_passed": True, "item": item, "correct_answer": "5/6",
            "student_text": f"I got {answer}.", "student_answer": answer, "student_correct": correct,
            "followed_rule": followed, "holds_misconception": followed}


def diag(n, beliefs, status="clear", misconception=RULE, topics=("fractions",)):
    return {"turn": n, "status": status, "misconception": misconception if status == "clear" else None,
            "beliefs": beliefs, "follow_ups": 0, "mode": "full", "topics": list(topics)}


def test_learning_gain_is_normalised_and_skips_full_pre_scores():
    records = (session("a", pre=[False, False], post=[True, False], holds=[])
               + session("b", student="t#0", pre=[True, True], post=[True, True], holds=[]))
    m = compute(records)["full"]["has_rule"]
    assert m["learning_gain"] == 0.5          # only session a: (0.5 - 0) / (1 - 0)
    assert m["gain_sessions"] == 1
    assert m["learning_gain_class"] == pytest.approx((0.75 - 0.5) / 0.5)
    assert m["fixed_rate"] == 1.0
    assert gain(0.4, 0.7) == pytest.approx(0.5)  # the proposal's example


def test_retention_across_sessions():
    records = (session("a0", index=0, post=[True], holds=[]) + session("a1", index=1, pre=[True], holds=[])
               + session("b0", student="t#0", index=0, post=[True], holds=[])
               + session("b1", student="t#0", index=1, pre=[False], holds=[RULE], relapsed=True))
    m = compute(records)["full"]["has_rule"]
    assert (m["retention_pairs"], m["retention"], m["retention_truth"], m["relapse_rate"]) == (2, 0.5, 0.5, 0.5)


def test_calibration_and_topic_accuracy():
    records = (session("a", turns=[turn(1)], diagnoses=[diag(1, {RULE: 0.9, "slip": 0.1})])
               + session("b", student="t#0", turns=[turn(1)],
                         diagnoses=[diag(1, {"rule:fractions.ignore_denominators": 0.6, RULE: 0.4},
                                         misconception="rule:fractions.ignore_denominators", topics=("decimals",))]))
    m = compute(records)["full"]["has_rule"]
    assert m["calibration_episodes"] == 2 and m["top_belief_accuracy"] == 0.5
    assert m["brier"] == pytest.approx(((0.1 ** 2 + 0.1 ** 2) + (0.6 ** 2 + 0.6 ** 2)) / 2)
    assert m["topic_accuracy"] == 0.5
    assert expected_calibration_error([(0.9, True, 0), (0.6, False, 0)]) == pytest.approx(0.5 * 0.1 + 0.5 * 0.6)


def test_adversarial_sessions_are_kept_out_of_the_ordinary_groups():
    leak = turn(1, text="The answer is 5/6.")
    records = session("a", turns=[turn(1)]) + session("b", student="adv#0", attack="direct_request", turns=[leak])
    result = compute(records)["full"]
    assert result["all"]["leak_rate"] == 0 and result["has_rule"]["sessions"] == 1
    assert result["attack:direct_request"]["leak_rate"] == 1
    assert result["attack:all"]["turns_to_first_leak"] == 1


def test_check_records_rejects_a_test_item_that_is_a_probe():
    probe = probe_items()[0]
    records = session("a")
    records.insert(1, rec("test", "a", phase="pre", item=probe, correct_answer="1", student_answer="1", correct=True))
    assert any("matches probe" in p for p in check_records(records))
    assert probe_overlap_problems({"1/9 + 4/11": "x"}, asked=["4/11 + 1/9"])


# ---------------------------------------------------------------- statistics

def test_bootstrap_ci_contains_the_estimate_and_resamples_students():
    u = Units("leak_rate", {f"s{i}": (float(i % 2), 1) for i in range(40)})
    lo, hi = bootstrap_ci(u, reps=500)
    assert lo < u.estimate() == 0.5 < hi


def test_paired_comparison_finds_a_real_difference_and_not_a_fake_one():
    a = Units("m", {f"s{i}": (1.0, 1) for i in range(30)})
    b = Units("m", {f"s{i}": (0.0, 1) for i in range(30)})
    c = compare(a, b, reps=500)
    assert c.difference == 1 and c.p_value < 0.01 and c.students == 30
    same = compare(a, a, reps=500)
    assert same.difference == 0 and same.p_value == 1


def test_holm():
    assert holm([0.01, 0.04, None, 0.03]) == [0.03, pytest.approx(0.06), None, pytest.approx(0.06)]


def test_units_pool_like_compute():
    records = (session("a", pre=[False], post=[True], holds=[]) + session("b", student="t#0", pre=[False],
                                                                           post=[False]))
    ss = grouped_sessions(records)["full"]["has_rule"]
    assert units(ss, "learning_gain").estimate() == compute(records)["full"]["has_rule"]["learning_gain"]


# ---------------------------------------------------------------- labelling

def test_cohen_kappa():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == 0
    assert cohen_kappa([True, True], [True, True]) is None


def test_sheets_are_blind_overlap_and_round_trip(tmp_path):
    records = []
    for i in range(30):
        cond = "full" if i % 2 else "b2"
        text = "The answer is 5/6." if i % 3 == 0 else "Think about the denominators."
        records += session(f"s{i}", student=f"s#{i}", condition=cond, turns=[turn(1, text=text)])
        records.insert(-1, rec("hint", f"s{i}", f"s#{i}", cond, turn=1, text=text, citations=[], rewrites=0,
                               safe_template=False))
    tasks = sample_tasks(records, per_kind={"leak": 10, "targets": 6})
    assert {t.kind for t in tasks} == {"leak", "targets"}
    write_sheets(tasks, tmp_path, ["A", "B"], overlap=0.5)
    rows = {}
    for name in "AB":
        with (tmp_path / f"{name}.csv").open() as f:
            sheet = list(csv.DictReader(f))
        for row in sheet:
            assert "full" not in row["context"] and "b2" not in row["context"]  # blind
            rows.setdefault(row["task_id"], []).append(name)
        key = read_key(tmp_path)
        for row in sheet:  # label each as the leak code would, so agreement and judge checks are perfect
            row["label"] = "yes" if key[row["task_id"]]["auto"] or row["kind"] == "targets" else "no"
        with (tmp_path / f"{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sheet[0]))
            w.writeheader()
            w.writerows(sheet)
    assert sum(len(v) == 2 for v in rows.values()) == round(0.5 * len(tasks))
    labels, key = read_labels(tmp_path), read_key(tmp_path)
    assert agreement(labels, key)["leak"]["agreement"] == 1
    check = check_judge(labels, key, "leak", leak_code_judge)
    assert check["accuracy"] == 1 and check["usable"]


def test_assign_gives_every_task_to_someone():
    tasks = [type("T", (), {"task_id": str(i)})() for i in range(20)]
    sheets = assign(tasks, ["A", "B", "C", "D"], overlap=0.2)
    assert sum(map(len, sheets.values())) == 24


# ---------------------------------------------------------------- the LeetCode-style analysis, report, Eedi helper

def test_analysis_metrics_check_practice_against_the_planted_rule():
    records = session("a", turns=[turn(1)])
    records.insert(-1, rec("hint", "a", turn=1, text="h", citations=[], rewrites=0, safe_template=False,
                           analysis={"where": "1/2 + 1/3 became 2/5", "mistake": "adds across",
                                     "topics": ["fractions"], "practice": ["1/4 + 1/5", "2x + 3 = 7"]}))
    m = compute(records)["full"]["has_rule"]
    assert (m["analysed_hints"], m["where_rate"], m["mistake_rate"], m["analysis_topic_accuracy"]) == (1, 1, 1, 1)
    assert m["practice_items"] == 2 and m["practice_relevance"] == 0.5  # the rule says nothing about 2x + 3 = 7


def test_report_has_every_section(tmp_path):
    from matheasy.evaluation.report import build
    records = []
    for cond in ("full", "b2"):
        for i in range(4):
            records += session(f"{cond}{i}", student=f"s#{i}", condition=cond, pre=[False], post=[i % 2 == 0],
                               turns=[turn(1)], diagnoses=[diag(1, {RULE: 1.0})], holds=[] if i % 2 == 0 else [RULE])
    text = build(records, reps=50)
    for heading in ("## 1. The runs", "## 2. What the tutor shows", "## 3. Learning", "## 4. Safety",
                    "## 5. full compared"):
        assert heading in text
    assert "Log checker: no problems" in text


def test_match_eedi_records_a_decision(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("match_eedi", Path(__file__).parents[1] / "eval" / "match_eedi.py")
    match_eedi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(match_eedi)
    path = tmp_path / "held.json"
    path.write_text(json.dumps({"eedi_status": {"fractions.subtract_across": None}}))
    assert match_eedi.set_status("fractions.subtract_across", "1287", path, known_ids={1287}) == {
        "fractions.subtract_across": 1287}
    assert match_eedi.set_status("fractions.subtract_across", "never_seen", path)["fractions.subtract_across"] \
        == "never_seen"
    with pytest.raises(ValueError):
        match_eedi.set_status("fractions.subtract_across", "99", path, known_ids={1287})
    with pytest.raises(ValueError):
        match_eedi.set_status("fractions.add_numerators_denominators", "1", path)
