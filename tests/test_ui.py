"""The analysis page: accounts, sessions, history, slips, and who may see what."""

import json
import threading
import urllib.error
import urllib.request
from http.cookiejar import CookieJar
from http.server import ThreadingHTTPServer

import pytest

from matheasy.diagnosis import SLIP
from matheasy.tutor import make_problem, planner
from matheasy.tutor.state import StudentMemory
from matheasy.ui.__main__ import Tutor, make_handler
from matheasy.ui.db import Database


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "t.db")
    d.add_user("mr_khan", "teach123", "teacher")
    d.add_user("anna", "anna123")
    d.add_user("ben", "ben1234")
    return d


def test_passwords_are_hashed_and_checked(db):
    row = db.conn.execute("SELECT hash FROM users WHERE username = 'anna'").fetchone()
    assert b"anna123" not in row["hash"]
    assert db.check("anna", "anna123") == "student" and db.check("mr_khan", "teach123") == "teacher"
    assert db.check("anna", "wrong") is None and db.check("nobody", "x") is None
    with pytest.raises(ValueError):
        db.add_user("anna", "another1")          # taken
    with pytest.raises(ValueError):
        db.add_user("cara", "short")             # too short


def test_sessions_start_and_end(db):
    token = db.start_session("anna")
    assert db.session_user(token) == ("anna", "student")
    db.end_session(token)
    assert db.session_user(token) is None and db.session_user(None) is None


def test_memory_and_attempts_round_trip(db):
    memory = StudentMemory("anna", p_knows={"fractions": 0.7})
    db.save_memory(memory)
    assert db.load_memory("anna").p_knows == {"fractions": 0.7}
    db.add_attempt("anna", "1/2 + 1/3", "2/5", False, wrong_line=1, topics=["fractions"])
    assert db.attempts("anna")[0]["topics"] == ["fractions"] and db.attempts("ben") == []


def test_a_mistake_on_mastered_concepts_is_a_slip():
    memory = StudentMemory("anna", p_knows={"fractions": 0.99})
    assert planner.is_slip(memory, ["fractions"], "rule:fractions.add_numerators_denominators")
    memory.p_holds["rule:fractions.add_numerators_denominators"] = 0.9    # suspected: a habit, not a slip
    assert not planner.is_slip(memory, ["fractions"], "rule:fractions.add_numerators_denominators")
    assert not planner.is_slip(StudentMemory("ben"), ["fractions"], None)  # not mastered yet


def test_a_slip_costs_little_mastery_and_is_not_a_mistake():
    normal, slipped = StudentMemory("a", p_knows={"fractions": 0.99}), StudentMemory("b", p_knows={"fractions": 0.99})
    planner.answered(normal, make_problem("p", "1/2 + 1/3"), "2/5", False, topics=["fractions"])
    planner.answered(slipped, make_problem("p", "1/2 + 1/3"), "2/5", False, topics=["fractions"], slip=True)
    assert slipped.p_knows["fractions"] > normal.p_knows["fractions"]
    assert normal.mistakes == {"fractions": 1} and slipped.mistakes == {}


@pytest.fixture
def server(db):
    tutor = Tutor(llm=False, eedi=False, db=db)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tutor))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", tutor
    httpd.shutdown()


def client(base):
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))

    def call(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with opener.open(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    return call


def test_signing_in_and_who_sees_what(server):
    base, tutor = server
    anna, teacher = client(base), client(base)
    assert anna("/api/me")[0] == 401                                   # signed out
    assert anna("/api/login", {"username": "anna", "password": "nope"})[0] == 401
    assert anna("/api/login", {"username": "anna", "password": "anna123"})[0] == 200
    status, r = anna("/api/analyze", {"question": "1/2 + 1/3 + 1/6", "answer": "3/6 + 2/6 + 1/6\n6/18", "student": "ben"})
    assert status == 200 and r["wrong_line"] == 2 and r["student"]["name"] == "anna"   # always her own
    assert anna("/api/student?name=ben")[1]["name"] == "anna"          # cannot read Ben
    assert anna("/api/reset", {"student": "anna"})[0] == 403           # only a teacher resets
    teacher("/api/login", {"username": "mr_khan", "password": "teach123"})
    assert set(teacher("/api/me")[1]["students"]) == {"anna", "ben"}
    history = teacher("/api/student?name=anna")[1]["history"]
    assert len(history) == 1 and history[0]["wrong_line"] == 2
    anna("/api/logout", {})
    assert anna("/api/me")[0] == 401


def test_the_page_flags_a_slip_on_a_mastered_concept(server):
    base, tutor = server
    memory = tutor.store.load("anna")
    memory.session_index = 0
    memory.p_knows["fractions"] = 0.99
    anna = client(base)
    anna("/api/login", {"username": "anna", "password": "anna123"})
    r = anna("/api/analyze", {"question": "1/2 + 1/3", "answer": "2/5"})[1]
    assert r["slip"] and r["misconception"] is None
    assert r["student"]["slips"] == 1 and r["student"]["history"][0]["slip"]
    assert memory.times_diagnosed == {SLIP: 1}


def test_a_word_or_stray_letters_are_not_an_answer():
    from matheasy.ui.__main__ import not_maths
    assert not_maths("{hello}", "1/2 + 1/3")              # a small model reads 'hello' as a named quantity
    assert not_maths("abc", "1/2 + 1/3")                  # letters when the question has none
    assert not_maths("2x + {cat}", "2(x + 3)")
    assert not not_maths("5/6", "1/2 + 1/3")
    assert not not_maths("x = 4", "2x + 3 = 11")
    assert not not_maths("{rate} * 3", "{rate} * 2 + 1")  # a quantity the question names


def test_the_page_progress_needs_sign_in_and_is_empty_when_idle(server):
    base, tutor = server
    anna = client(base)
    assert anna("/api/progress")[0] == 401
    anna("/api/login", {"username": "anna", "password": "anna123"})
    assert anna("/api/progress") == (200, {"stage": ""})
    anna("/api/analyze", {"question": "1/2 + 1/3", "answer": "2/5"})
    assert anna("/api/progress") == (200, {"stage": ""})          # reset after the analysis


def test_signing_up_creates_a_student_and_signs_them_in(server, db):
    base, _ = server
    cara = client(base)
    status, r = cara("/api/signup", {"username": "cara", "password": "cara123", "role": "teacher"})
    assert status == 200 and r == {"user": "cara", "role": "student"}    # a sign-up is never a teacher
    assert db.role("cara") == "student" and db.check("cara", "cara123") == "student"
    assert cara("/api/me")[1]["user"] == "cara"                          # signed in straight away
    assert b"cara123" not in db.conn.execute("SELECT hash FROM users WHERE username = 'cara'").fetchone()["hash"]
    assert client(base)("/api/signup", {"username": "cara", "password": "other12"})[0] == 400   # taken
    assert client(base)("/api/signup", {"username": "dev", "password": "123"})[0] == 400        # too short
