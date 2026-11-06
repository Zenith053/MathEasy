import dataclasses
import json

import pytest

from matheasy.evaluation import SimStudent, adversarial_roster, check_files, overlapping, roster
from matheasy.llm import FakeBackend, LLMClient, load_config
from matheasy.tutor import runner
from matheasy.tutor.__main__ import keyword_judge


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUNS", tmp_path)
    return tmp_path


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sessions(path):
    return [r for r in read(path) if r["event"] == "session_start"]


def test_shards_split_students_and_every_condition_runs_the_same_ones(runs):
    for shard in ("0/2", "1/2"):
        runner.main(["t", "--shard", shard, "--students", "2", "--sessions", "1", "--per-attack", "1"])
    students = {}
    for path in (runs / "t" / "turns").glob("*.jsonl"):
        condition, shard = path.stem.split("-")
        students.setdefault(shard, {})[condition] = {s["student_id"] for s in sessions(path)}
    assert set(students) == {"shard0of2", "shard1of2"}
    for by_condition in students.values():
        assert by_condition["full"] == by_condition["b2"]
    assert not students["shard0of2"]["full"] & students["shard1of2"]["full"]
    ids = [s.student_id for s in roster(keyword_judge, slip_only=runner.SLIP_ONLY)]
    adversarial = [s.student_id for s in adversarial_roster(keyword_judge, per_attack=1)]
    assert students["shard1of2"]["full"] <= set(ids[1::2]) | set(adversarial[1::2])
    assert any(sid.startswith("adv-") for sid in students["shard1of2"]["full"])


def test_sessions_carry_memory_and_students_over(runs):
    runner.main(["t", "--students", "2", "--sessions", "3", "--conditions", "full", "--per-attack", "0"])
    starts = sessions(runs / "t" / "turns" / "full-shard0of1.jsonl")
    assert [s["session_index"] for s in starts if s["student_id"] == starts[0]["student_id"]] == [0, 1, 2]
    assert any(s["rechecks"] for s in starts if s["session_index"] > 0)
    memory = json.loads((runs / "t" / "memory" / "full-shard0of1.json").read_text())
    assert all(m["session_index"] == 2 for m in memory.values())
    saved = SimStudent.load(runs / "t" / "students" / "full", starts[0]["student_id"], judge=keyword_judge)
    assert saved.session_index == 2


def test_a_stopped_run_carries_on_and_drops_the_unfinished_session(runs, capsys):
    runner.main(["t", "--students", "2", "--sessions", "2", "--conditions", "full", "--per-attack", "0"])
    path = runs / "t" / "turns" / "full-shard0of1.jsonl"
    records = read(path)
    last = records[-1]["session_id"]
    # As if the run stopped part-way through its last session.
    cut = next(i for i, r in enumerate(records) if r["session_id"] == last) + 3
    path.write_text("".join(json.dumps(r) + "\n" for r in records[:cut]))

    capsys.readouterr()
    runner.main(["t", "--students", "2", "--sessions", "2", "--conditions", "full", "--per-attack", "0"])
    assert "1 sessions run, 3 already done" in capsys.readouterr().out
    again = read(path)
    assert [r["session_id"] for r in again if r["event"] == "session_start"].count(last) == 1
    assert len(again) == len(records) and check_files([path]) == []


def test_conditions_are_checked(runs):
    with pytest.raises(SystemExit):
        runner.main(["t", "--conditions", "b1"])  # needs --llm
    with pytest.raises(SystemExit):
        runner.main(["t", "--conditions", "full,b9"])
    with pytest.raises(SystemExit):
        runner.main(["t", "--shard", "2/2"])


def test_b0_and_b1_run_on_the_llm(runs):
    def reply(request):
        if request.system and "research study" in request.system:  # D's feedback judge
            return json.dumps({"targets": False, "reason": "no"})
        return json.dumps({"action": "message", "tool": "", "args": "", "message": "Check each step again.",
                           "student_is_correct": False, "misconception_rule": "", "student_misconception": "",
                           "citation": ""})
    config = dataclasses.replace(load_config(), log_dir=runs / "llm")
    client = LLMClient(config, backend=FakeBackend(reply), run_id="t")
    students = [SimStudent("fractions.add_numerators_denominators#0", "fractions.add_numerators_denominators",
                           judge=keyword_judge)]
    for condition in ("b0", "b1"):
        result = runner.run_condition(condition, students, runs / "t", sessions=1, shard="0of1", client=client,
                                      config_hash=config.hash, retrieve=None)
        assert result["ran"] == 1
        assert check_files([result["log"], client.log_path]) == []
        assert all(r["critic_passed"] is None for r in read(result["log"]) if r["event"] == "turn")


def test_d_s_tests_and_attacks_reach_the_logs(runs):
    runner.main(["t", "--students", "1", "--per-attack", "1", "--sessions", "2", "--conditions", "full"])
    records = read(runs / "t" / "turns" / "full-shard0of1.jsonl")
    attacks = {r["student_id"]: r["attack"] for r in records if r["event"] == "session_start"}
    assert attacks[next(sid for sid in attacks if not sid.startswith("adv-"))] is None
    assert {a for sid, a in attacks.items() if sid.startswith("adv-")} == {"direct_request"}

    tests = [r for r in records if r["event"] == "test"]
    by_session = {}
    for t in tests:
        by_session.setdefault((t["session_id"], t["phase"]), []).append(t["item"])
    sid = next(iter(by_session))[0]
    pre, post = by_session[(sid, "pre")], by_session[(sid, "post")]
    assert pre and post and overlapping(pre, post) == []  # parallel forms, new numbers
    tutored = [r["item"] for r in records if r["event"] == "turn" and r["session_id"] == sid and r["item"]]
    assert overlapping(pre + post, tutored) == []
    assert check_files([runs / "t" / "turns" / "full-shard0of1.jsonl"]) == []


def test_students_relapse_between_sessions_by_default(runs):
    runner.main(["t", "--students", "1", "--per-attack", "0", "--sessions", "1", "--conditions", "full"])
    [path] = (runs / "t" / "students" / "full").glob("*.json")
    assert json.loads(path.read_text())["relapse_rate"] == runner.RELAPSE_RATE
