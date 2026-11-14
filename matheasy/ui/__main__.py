"""A local page for trying the tutor's analysis: sign in, type a question and the working, press Analyze.

    python -m matheasy.ui.db add-user mr_khan --role teacher   # first, create accounts
    python -m matheasy.ui.db add-user anna
    python -m matheasy.ui                 # code only: rules, solve back, probes; template hints
    python -m matheasy.ui --llm           # + the LLM fallback, LLM hints and the critic's review (GEMINI_API_KEY)
    python -m matheasy.ui --llm --eedi    # + Eedi misconceptions from the shortlister (GPU; loads for minutes)
    python -m matheasy.ui --llm --no-llm-diagnosis   # the LLM everywhere except diagnosis (T3)
    python -m matheasy.ui --vllm          # the LLM is an open model on our own GPU (config/llm_vllm.yaml)
    -> http://localhost:8000  (8001 with --vllm, since the vLLM server uses 8000)

--vllm needs a running vLLM server (Linux or WSL; the command is in config/llm_vllm.yaml).
--vllm-url and --vllm-model point it at another server or model, e.g. a smaller one that
fits a laptop GPU. In A's test (eval/llm_compare.py) Gemini was more accurate, so the
project's runs use Gemini; vLLM shows the tutor working with no API key and no quota.

Each Analyze runs what the tutor runs: every line of the working is read (transcribe) and
checked against the one before; the first wrong step is diagnosed, then shown with where it
went wrong, the mistake, its topics, a hint that passed the critic, textbook passages to
revise and similar problems. The student's memory (the tutor's planner) is updated and the
attempt saved, so mastery per concept and slips build up over time. A wrong answer on
concepts the student has mastered, with no misconception they are suspected of, is a slip.

Accounts, sessions, attempts and memory live in runs/ui/matheasy.db (db.py). Anyone can sign
up for a student account on the page; teacher accounts are created from the command line. A
student sees only their own work; a teacher picks any student. Served on 127.0.0.1 over plain HTTP: put it
behind HTTPS before letting anyone reach it over a network.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import threading
import time
import traceback
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import httpx

from ..diagnosis import (
    SLIP, ParseError, StepIsCorrect, check_step, diagnose, load_ranker, parse, similar_items, topic_of, unknown_runs,
    uses_slip,
)
from ..evaluation.students import correct_answer
from ..llm import BudgetExceeded, LLMCallError, LLMClient, LLMOutputError
from ..llm.config import REPO_ROOT, LLMConfig, load_config
from ..tutor import planner
from ..tutor.__main__ import textbook_retrieve
from ..tutor.analysis import analyse, mistake, topic_name
from ..tutor.critic import TutorCritic, leaks_answer
from ..tutor.describe import misconception_text
from ..tutor.hinter import LLMHinter
from ..tutor.hinting import hint_request, write_hint
from ..tutor.orchestrator import SAFE_TEMPLATE, read_answer
from ..tutor.placeholders import TemplateHinter
from ..tutor.state import Problem
from ..tutor.store import InMemoryStore
from .db import Database

PAGE = Path(__file__).with_name("index.html")
COOKIE = "matheasy_session"
LLM_ERRORS = (BudgetExceeded, LLMCallError, LLMOutputError)
# Hint rewrites after the critic rejects one: the runner allows 2 (tutor/hinting.py); the page allows 1,
# so the worst wait is two hint-and-critic rounds, not three. Then the safe template, as in the runner.
PAGE_REWRITES = 1
BRACED = re.compile(r"\{([^{}]*)\}")
LETTER = re.compile(r"[A-Za-z]")


def not_maths(line: str, question: str) -> bool:
    """A reading that can't be an answer to `question`: a named quantity ({hello}) the question never
    mentions, or letters when the question has none. The reader may turn any word into {word}, and a
    small model reads 'hello' that way, so this check runs with the LLM on too."""
    q = question.lower()
    if any(word.strip().lower() not in q for word in BRACED.findall(line)):
        return True
    return bool(LETTER.search(BRACED.sub("", line))) and not LETTER.search(question)


class DbStore(InMemoryStore):
    """The tutor's store, kept in the database: every change is saved straight away."""

    def __init__(self, db: Database):
        super().__init__()
        self.db = db

    def load(self, student_id):
        if student_id not in self._memories:
            self._memories[student_id] = self.db.load_memory(student_id)
        return self._memories[student_id]

    def begin_session(self, memory, session_index):
        super().begin_session(memory, session_index)
        self.db.save_memory(memory)

    def record(self, memory, misconception_id, topics=()):
        super().record(memory, misconception_id, topics)
        self.db.save_memory(memory)

    def observe(self, memory, problem, answer, correct, topics=(), slip=False):
        super().observe(memory, problem, answer, correct, topics, slip)
        self.db.save_memory(memory)

    def forget(self, student_id):
        self._memories.pop(student_id, None)


class Tutor:
    """The pieces the page uses, built once: the database, the LLM session, the Eedi ranker, the textbook search."""

    def __init__(self, llm: bool, eedi: bool, db: Optional[Database] = None, llm_diagnosis: bool = True,
                 llm_config: Optional[LLMConfig] = None):
        self.llm_diagnosis = llm_diagnosis  # False: only the diagnoser's LLM is off (T3)
        self.db = db or Database()
        self.store = DbStore(self.db)
        self.lock = threading.Lock()  # one analysis at a time: memory updates must not interleave
        self.stage = ""  # what the running analysis is doing, for the page's progress list (/api/progress)
        self.max_rewrites = PAGE_REWRITES
        self.llm_config = (llm_config or load_config()) if llm else None
        self.session = LLMClient(self.llm_config, run_id="ui").session("ui", "full", budget_tokens=None) if llm else None
        self.retrieve = textbook_retrieve()
        self.hinter = LLMHinter(self.retrieve) if llm else TemplateHinter(self.retrieve)
        self.critic = TutorCritic()
        self.ranker = None
        if eedi:
            print("loading the Eedi shortlister (takes a few minutes on a small GPU)...")
            self.ranker = load_ranker()
            print("Eedi shortlister:", "ready" if self.ranker else "did not load (needs torch, a GPU and data/)")

    def status(self) -> dict:
        cfg = self.llm_config
        return {"llm": self.session is not None, "eedi": self.ranker is not None, "textbook": self.retrieve is not None,
                "llm_diagnosis": self.session is not None and self.llm_diagnosis,
                "vllm": cfg is not None and cfg.provider == "vllm", "model": cfg.model() if cfg else None}

    def memory(self, student: str):
        memory = self.store.load(student)
        if memory.session_index < 0:
            self.store.begin_session(memory, 0)
        return memory

    def student(self, student: str) -> dict:
        """What the tutor remembers about this student: mastery per concept (weakest first),
        misconceptions seen, slips, and recent attempts."""
        memory = self.memory(student)
        rows = []
        for t in sorted(set(memory.p_knows) | set(memory.mistakes)):
            p = memory.p_knows.get(t, planner.KNOWS_AT_START)
            state = "mastered" if planner.mastered(memory, t) else "weak" if memory.mistakes.get(t) else "learning"
            rows.append({"topic": topic_name(t), "mastery": round(p, 3), "mistakes": memory.mistakes.get(t, 0), "state": state})
        rows.sort(key=lambda r: (r["mastery"], -r["mistakes"], r["topic"]))
        seen = [{"id": m, "text": misconception_text(m) or m, "times": n,
                 "suspected": round(memory.p_holds.get(m, 0.0), 2), "topics": [topic_name(t) for t in memory.topics_of.get(m, [])]}
                for m, n in sorted(memory.times_diagnosed.items(), key=lambda kv: -kv[1]) if m != SLIP]
        return {"name": student, "answered": len(memory.worked), "topics": rows, "misconceptions": seen,
                "slips": memory.times_diagnosed.get(SLIP, 0), "mastered_at": planner.MASTERED,
                "history": self.db.attempts(student, limit=15)}

    def reset(self, student: str) -> dict:
        self.db.clear_student(student)
        self.store.forget(student)
        return self.student(student)

    def analyze(self, question: str, answer_text: str, student: str) -> dict:
        with self.lock:
            try:
                result = self._analyze(question, answer_text, student)
            finally:
                self.stage = ""
            if result["kind"] in ("correct", "wrong"):
                self.db.add_attempt(student, question.strip(), answer_text.strip(), result["kind"] == "correct",
                                    slip=result.get("slip", False), wrong_line=result.get("wrong_line"),
                                    step=result.get("step"), status=result.get("status"),
                                    misconception=result.get("misconception"), topics=result.get("topics", ()))
            result["student"] = self.student(student)
        return result

    def _analyze(self, question: str, answer_text: str, student: str) -> dict:
        """The student's working, one step per line: each line is checked against the one before,
        and the first wrong step is diagnosed (previous line -> wrong line)."""
        start = time.time()
        question = question.strip()
        try:
            node = parse(question)
        except ParseError as e:
            return {"kind": "error", "message": f"The question doesn't parse: {e}"}
        written = [line.strip() for line in answer_text.splitlines() if line.strip()]
        if not written:
            return {"kind": "unread", "message": "Type the working, one step per line.", "choices": []}
        lines = []
        self.stage = "read"
        for i, text in enumerate(written, 1):
            if self.session is None and unknown_runs(text):
                # Without the LLM nothing can tell a word ("hello") from letters multiplied (h × e × l × l × o).
                return {"kind": "unread", "message": f"Line {i}: type it as maths, e.g. 5/6 or x = 4.", "choices": []}
            reading = read_answer(text, question, self.session)
            if reading.status != "ok":
                return {"kind": "unread", "message": f"Line {i}: {_unread(reading)}", "choices": list(reading.choices)}
            if not_maths(reading.line, question):
                return {"kind": "unread", "message": f"Line {i}: type it as maths, e.g. 5/6 or x = 4.", "choices": []}
            lines.append(reading.line)
        try:
            correct: Optional[str] = correct_answer(question)
        except Exception:
            correct = None
        memory = self.memory(student)
        chain = [question, *lines]
        self.stage = "check"
        try:
            ok = [check_step(a, b) for a, b in zip(chain, chain[1:])]
        except ParseError as e:
            return {"kind": "error", "message": f"A line doesn't parse: {e}"}
        steps = [{"line": text, "read_as": line if line != text else None, "ok": good}
                 for text, line, good in zip(written, lines, ok)]
        topic = topic_of(node)
        problem = Problem("ui", topic, question, correct or "")
        if all(ok):
            self.store.observe(memory, problem, lines[-1], True)
            return {"kind": "correct", "answer": lines[-1], "steps": steps, "seconds": round(time.time() - start, 1)}

        k = ok.index(False)  # the first wrong step: chain[k] -> chain[k + 1]
        before, after = chain[k], chain[k + 1]
        self.stage = "diagnose"
        try:
            d = diagnose(before, after, question=question, session=self.session if self.llm_diagnosis else None,
                         eedi_ranker=self.ranker if self.llm_diagnosis else None)
        except StepIsCorrect:
            return {"kind": "correct", "answer": lines[-1], "steps": steps}
        except LLM_ERRORS as e:
            return {"kind": "error", "message": f"LLM call failed: {e}"}
        d = planner.raise_fact(memory, d)  # the same number fact wrong again: a weak fact, not a slip
        # T9: the diagnosed rule's mistake made once, and the same step done right elsewhere in this answer
        uses = uses_slip(chain, d.misconception)
        slip = d.misconception == SLIP or bool(uses) or planner.is_slip(memory, d.topics, d.misconception)
        # what the student is told: a number fact's "said" leaves out the right result ("wrote 8 for 6 + 3")
        why_slip = next((t.get("said") or t["reason"] for t in d.trace if t.get("stage") == "slip"), None) or uses
        self.store.observe(memory, problem, lines[-1], False, topics=d.topics, slip=slip)
        if slip:
            self.store.record(memory, SLIP, topics=d.topics)
        elif d.misconception:
            self.store.record(memory, d.misconception, topics=d.topics)
        analysis = analyse(d, problem, memory.worked)
        words = None if slip else mistake(d, problem)
        right = tuple(f"{a} → {b}" for a, b in zip(chain[:k], chain[1:k + 1]))  # the lines before the wrong one
        self.stage = "hint"
        hint, citation, rejected, dropped = self._hint(problem, after, d, slip, right)
        likely = None
        if dropped:  # the critic said the student's work rules this misconception out
            # A rule that reproduces the step exactly is checked by code, so the page still shows it,
            # marked as likely: an LLM critic (a small model especially) can be wrong about the fit.
            likely = words if d.status == "clear" else None
            words = None
        self.stage = "revise"

        e = d.explanation
        practice = list(analysis.practice) or similar_items(d.topics, seen=[question], n=3)
        return {
            "kind": "wrong", "answer": lines[-1], "steps": steps, "wrong_line": k + 1, "step": f"{before} → {after}",
            "status": d.status, "source": _source(d), "fit": e.eedi_fit if e else None, "slip": slip,
            "why_slip": why_slip,
            "misconception": None if slip else d.misconception, "mistake": words, "likely": likely,
            "reading": None if slip or words else _reading(e, problem), "where": analysis.where,
            "topics": [topic_name(t) for t in analysis.topics],
            "hint": hint, "citation": citation, "hint_rejected": rejected, "revise": [] if slip else self._revise(words, d, topic, citation),
            "similar": practice,
            "probe": d.probe.question if d.probe and not slip else None,
            "working": _working(e),
            "working_checked": e.status if e else None,
            "seconds": round(time.time() - start, 1),
        }

    def _hint(self, problem: Problem, answer: str, d, slip: bool, right: tuple[str, ...]):
        """A hint that passed the critic, written as in the orchestrator (tutor/hinting.py); else the
        safe template. Also every rejected attempt and the critic's reasons, so the page can show
        them, and whether the critic ruled the misconception out."""
        written = write_hint(self.hinter, self.critic, hint_request(problem, answer, d, slip=slip, right=right),
                             problem, llm=self.session, max_rewrites=self.max_rewrites)
        rejected = list(written.attempts)
        if written.error is not None:
            rejected.append({"hint": None, "reasons": [f"LLM call failed: {written.error}"]})
        if written.hint is None:
            return SAFE_TEMPLATE, None, rejected, written.dropped
        hint = written.hint
        return hint.text, hint.citations[0] if hint.citations else None, rejected, written.dropped

    def _revise(self, words: Optional[str], d, topic: str, citation: Optional[str]) -> list[dict]:
        """Textbook passages to read, the hint's citation first. Searched with the mistake's words
        and its topics; with neither, there is nothing to search with (bare maths finds noise)."""
        topics = " ".join(topic_name(t) for t in d.topics if t != "other")
        if not words and d.status == "ambiguous":  # rules tie: their descriptions say what to revise
            words = " ".join(filter(None, (misconception_text(m) for m in d.beliefs)))
        if self.retrieve is None or not (words or topics):
            return []
        passages = self.retrieve(f"{words or ''} {topics}".strip(), k=3)
        passages.sort(key=lambda p: p["ref"] != citation)
        return [{"ref": p["ref"], "text": p["text"][:700]} for p in passages[:2]]


def _reading(e, problem: Problem) -> Optional[str]:
    """The LLM's own reading of the mistake when code could not confirm it: for teachers only (the page
    shows it with the technical details), and never if it gives the answer away."""
    if e is None or not e.mistake or leaks_answer(e.mistake, problem.correct_answer, problem.question):
        return None
    return f"{e.mistake} ({e.status}{': ' + e.reason if e.reason else ''})"


def _working(e) -> list[dict]:
    """The LLM's reconstructed working, each wrong step with what code found in it."""
    if e is None:
        return []
    wrong = {w.after: w for w in e.wrong_steps}
    rows = []
    for line in e.steps:
        w = wrong.get(line)
        rows.append({"line": line, "wrong": w is not None,
                     "rule": (misconception_text(w.rule) or w.rule) if w and w.rule else None,
                     "slip": w.slip if w else None})
    return rows


def _source(d) -> str:
    if any(t.get("from") == "working" for t in d.trace) and d.misconception != SLIP:
        return "working"  # a rule reproduces a step of the LLM's reconstructed working
    if d.explanation is not None and d.explanation.eedi and d.misconception:
        return "eedi"  # picked from Eedi, even when recorded under the one rule it belongs to
    if d.misconception and d.misconception.startswith("rule:"):
        return "rule"
    if d.misconception and d.misconception.startswith("eedi:"):
        return "eedi"
    if d.misconception and d.misconception.startswith("fact:"):
        return "fact"
    if d.status == "ambiguous":
        return "ambiguous"
    if d.explanation is not None:
        return "llm"
    return "none"


def _unread(reading) -> str:
    return {"unsure": "That could be read more than one way. Which did you mean?",
            "not_maths": "Type the answer as maths, e.g. 5/6 or x = 4.",
            "unfaithful": "Please type the answer again.",
            }.get(reading.status, f"Couldn't read the answer: {reading.reason or reading.status}")


def make_handler(tutor: Tutor):
    db = tutor.db

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body, kind: str = "application/json", cookie: Optional[str] = None):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            if cookie is not None:
                self.send_header("Set-Cookie", cookie)
            self.end_headers()
            self.wfile.write(data)

        def _token(self) -> Optional[str]:
            jar = SimpleCookie(self.headers.get("Cookie", ""))
            return jar[COOKIE].value if COOKIE in jar else None

        def _user(self):
            return db.session_user(self._token())

        def _body(self) -> dict:
            return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

        def _whose(self, user, asked: Optional[str]) -> Optional[str]:
            """The student this request is about: a student only ever themselves; a teacher any student."""
            name, role = user
            if role == "student":
                return name
            asked = (asked or "").strip()
            return asked if asked and db.role(asked) == "student" else None

        def do_GET(self):
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if url.path == "/api/status":
                return self._send(200, tutor.status())
            user = self._user()
            if user is None:
                return self._send(401, {"error": "Please sign in."})
            if url.path == "/api/progress":  # the running analysis's stage; "" when none is running
                return self._send(200, {"stage": tutor.stage})
            if url.path == "/api/me":
                return self._send(200, {"user": user[0], "role": user[1],
                                        "students": db.students() if user[1] == "teacher" else [user[0]]})
            if url.path == "/api/student":
                who = self._whose(user, parse_qs(url.query).get("name", [None])[0])
                if who is None:
                    return self._send(404, {"error": "No such student."})
                return self._send(200, tutor.student(who))
            self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                data = self._body()
                if self.path == "/api/login":
                    name = str(data.get("username", "")).strip()
                    role = db.check(name, str(data.get("password", "")))
                    if role is None:
                        return self._send(401, {"error": "Wrong username or password."})
                    token = db.start_session(name)
                    return self._send(200, {"user": name, "role": role},
                                      cookie=f"{COOKIE}={token}; HttpOnly; SameSite=Strict; Path=/")
                if self.path == "/api/signup":
                    # Anyone can create a student account; teachers are made from the command line only.
                    name, password = str(data.get("username", "")).strip(), str(data.get("password", ""))
                    try:
                        db.add_user(name, password, "student")
                    except ValueError as e:
                        return self._send(400, {"error": str(e).capitalize() + "."})
                    token = db.start_session(name)
                    return self._send(200, {"user": name, "role": "student"},
                                      cookie=f"{COOKIE}={token}; HttpOnly; SameSite=Strict; Path=/")
                if self.path == "/api/logout":
                    db.end_session(self._token())
                    return self._send(200, {"ok": True}, cookie=f"{COOKIE}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0")
                user = self._user()
                if user is None:
                    return self._send(401, {"error": "Please sign in."})
                who = self._whose(user, data.get("student"))
                if who is None:
                    return self._send(404, {"error": "Pick a student."})
                if self.path == "/api/reset":
                    if user[1] != "teacher":
                        return self._send(403, {"error": "Only a teacher can reset a student."})
                    return self._send(200, tutor.reset(who))
                if self.path == "/api/analyze":
                    return self._send(200, tutor.analyze(str(data.get("question", "")), str(data.get("answer", "")), who))
                self._send(404, {"error": "not found"})
            except Exception as e:  # show the error on the page rather than dropping the connection
                traceback.print_exc()
                self._send(200, {"kind": "error", "message": f"{type(e).__name__}: {e}"})

        def log_message(self, *args):
            pass

    return Handler


def vllm_config(ap: argparse.ArgumentParser, url: Optional[str], model: Optional[str]) -> LLMConfig:
    """config/llm_vllm.yaml, with the server and model changed if asked, once the server answers."""
    config = load_config(REPO_ROOT / "config" / "llm_vllm.yaml")
    config = dataclasses.replace(config, base_url=url or config.base_url,
                                 models={**config.models, "backbone": model or config.model()})
    try:
        served = [m["id"] for m in httpx.get(config.base_url.rstrip("/") + "/models", timeout=5).json()["data"]]
    except (httpx.HTTPError, ValueError, KeyError) as e:
        ap.error(f"no vLLM server answers at {config.base_url} ({type(e).__name__}). Start one first "
                 "(see config/llm_vllm.yaml), or pass --vllm-url")
    if config.model() not in served:
        ap.error(f"the vLLM server serves {served}, not {config.model()}: pass --vllm-model")
    return config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", action="store_true", help="LLM fallback, hints and critic (needs GEMINI_API_KEY)")
    ap.add_argument("--eedi", action="store_true", help="Eedi misconceptions (needs --llm, a GPU and data/)")
    ap.add_argument("--no-llm-diagnosis", dest="llm_diagnosis", action="store_false",
                    help="with --llm, turn off only the diagnoser's LLM (T3); reading, hints and the critic keep it")
    ap.add_argument("--vllm", action="store_true",
                    help="like --llm, but on an open model served by vLLM (config/llm_vllm.yaml); no API key")
    ap.add_argument("--vllm-url", help="the vLLM server, if not the one in config/llm_vllm.yaml (.../v1)")
    ap.add_argument("--vllm-model", help="the model the vLLM server serves, if not the one in config/llm_vllm.yaml")
    ap.add_argument("--port", type=int, default=None, help="default 8000, or 8001 with --vllm")
    args = ap.parse_args()
    config = None
    if args.vllm:
        args.llm = True
        config = vllm_config(ap, args.vllm_url, args.vllm_model)
    elif args.vllm_url or args.vllm_model:
        ap.error("--vllm-url and --vllm-model need --vllm")
    port = args.port or (8001 if args.vllm else 8000)
    if args.eedi and not args.llm:
        ap.error("--eedi needs --llm: the LLM fallback picks the Eedi misconception")
    if args.eedi and not args.llm_diagnosis:
        ap.error("--eedi needs the diagnoser's LLM: drop --no-llm-diagnosis")
    tutor = Tutor(args.llm, args.eedi and args.llm_diagnosis, llm_diagnosis=args.llm_diagnosis, llm_config=config)
    if not tutor.db.users():
        print("No accounts yet. Create one first:  python -m matheasy.ui.db add-user <name> --role teacher")
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(tutor))
    print(f"MathEasy analysis page: http://localhost:{port}  ({tutor.status()})  Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
