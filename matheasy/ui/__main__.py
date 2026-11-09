"""A local page for trying the diagnosis: type a question and a student's answer, press Analyze.

    python -m matheasy.ui                 # code only: rules, solve back, probes; template hints
    python -m matheasy.ui --llm           # + the LLM fallback, LLM hints and the critic's review (GEMINI_API_KEY)
    python -m matheasy.ui --llm --eedi    # + Eedi misconceptions from the shortlister (GPU; loads for minutes)
    -> http://localhost:8000

Each Analyze runs what the tutor runs on a wrong answer: read the answer (transcribe), check
it, diagnose(), then the analysis (where it went wrong, the mistake, its topics, similar
problems from the probe bank), a hint that has passed the critic, and the textbook passage
to revise. Nothing is saved; no student memory is kept between analyses.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from ..diagnosis import (
    ParseError, StepIsCorrect, check_step, diagnose, load_ranker, parse, similar_items, topic_of, unknown_runs,
)
from ..evaluation.students import correct_answer
from ..llm import BudgetExceeded, LLMCallError, LLMClient, LLMOutputError
from ..tutor.__main__ import textbook_retrieve
from ..tutor.analysis import analyse, mistake, topic_name
from ..tutor.critic import TutorCritic
from ..tutor.describe import misconception_text
from ..tutor.hinter import LLMHinter
from ..tutor.orchestrator import MAX_REWRITES, SAFE_TEMPLATE, read_answer
from ..tutor.placeholders import TemplateHinter
from ..tutor.state import HintRequest, Problem

PAGE = Path(__file__).with_name("index.html")
LLM_ERRORS = (BudgetExceeded, LLMCallError, LLMOutputError)


class Tutor:
    """The pieces the page uses, built once: the LLM session, the Eedi ranker, the textbook search."""

    def __init__(self, llm: bool, eedi: bool):
        self.session = LLMClient(run_id="ui").session("ui", "full", budget_tokens=None) if llm else None
        self.retrieve = textbook_retrieve()
        self.hinter = LLMHinter(self.retrieve) if llm else TemplateHinter(self.retrieve)
        self.critic = TutorCritic()
        self.ranker = None
        if eedi:
            print("loading the Eedi shortlister (takes a few minutes on a small GPU)...")
            self.ranker = load_ranker()
            print("Eedi shortlister:", "ready" if self.ranker else "did not load (needs torch, a GPU and data/)")

    def status(self) -> dict:
        return {"llm": self.session is not None, "eedi": self.ranker is not None, "textbook": self.retrieve is not None}

    def analyze(self, question: str, answer_text: str) -> dict:
        start = time.time()
        question, answer_text = question.strip(), answer_text.strip()
        try:
            node = parse(question)
        except ParseError as e:
            return {"kind": "error", "message": f"The question doesn't parse: {e}"}
        if self.session is None and unknown_runs(answer_text):
            # Without the LLM nothing can tell a word ("hello") from letters multiplied (h × e × l × l × o).
            return {"kind": "unread", "message": "Type the answer as maths, e.g. 5/6 or x = 4.", "choices": []}
        reading = read_answer(answer_text, question, self.session)
        if reading.status != "ok":
            return {"kind": "unread", "message": _unread(reading), "choices": list(reading.choices)}
        answer = reading.line
        try:
            correct: Optional[str] = correct_answer(question)
        except Exception:
            correct = None
        if check_step(question, answer):
            return {"kind": "correct", "answer": answer, "seconds": round(time.time() - start, 1)}

        try:
            d = diagnose(question, answer, question=question, session=self.session, eedi_ranker=self.ranker)
        except StepIsCorrect:
            return {"kind": "correct", "answer": answer}
        except LLM_ERRORS as e:
            return {"kind": "error", "message": f"LLM call failed: {e}"}
        topic = topic_of(node)
        problem = Problem("ui", topic, question, correct or "")
        analysis = analyse(d, problem)
        words = mistake(d, problem)
        hint, citation = self._hint(problem, answer, words, d)

        e = d.explanation
        practice = list(analysis.practice) or similar_items(d.topics, seen=[question], n=3)
        return {
            "kind": "wrong", "answer": answer, "read_as": answer if answer != answer_text else None,
            "status": d.status, "source": _source(d), "fit": e.eedi_fit if e else None,
            "misconception": d.misconception, "mistake": words, "where": analysis.where,
            "topics": [topic_name(t) for t in analysis.topics],
            "hint": hint, "citation": citation, "revise": self._revise(words, d, topic, citation),
            "similar": practice,
            "probe": d.probe.question if d.probe else None,
            "working": [{"line": s, "wrong": any(w.after == s for w in e.wrong_steps)} for s in e.steps] if e else [],
            "working_checked": e.status if e else None,
            "seconds": round(time.time() - start, 1),
        }

    def _hint(self, problem: Problem, answer: str, words: Optional[str], d) -> tuple[str, Optional[str]]:
        """A hint that passed the critic, as in the orchestrator; else the safe template."""
        located = d.located
        request = HintRequest(problem.topic, problem.question, answer, words,
                              f"{located.source} became {located.wrong}" if located else None)
        feedback: list[str] = []
        for _ in range(MAX_REWRITES + 1):
            try:
                hint = self.hinter.hint(request, feedback or None, llm=self.session)
                verdict = self.critic.review(hint.text, "hint", problem, request=request, llm=self.session)
            except LLM_ERRORS:
                break
            if verdict.passed:
                return hint.text, hint.citations[0] if hint.citations else None
            feedback = list(verdict.reasons)
        return SAFE_TEMPLATE, None

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


def _source(d) -> str:
    if d.misconception and d.misconception.startswith("rule:"):
        return "rule"
    if d.misconception and d.misconception.startswith("eedi:"):
        return "eedi"
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
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, kind: str):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/status":
                return self._send(200, json.dumps(tutor.status()).encode(), "application/json")
            if self.path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/analyze":
                return self._send(404, b"not found", "text/plain")
            try:
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                result = tutor.analyze(str(data.get("question", "")), str(data.get("answer", "")))
            except Exception as e:  # show the error on the page rather than dropping the connection
                traceback.print_exc()
                result = {"kind": "error", "message": f"{type(e).__name__}: {e}"}
            self._send(200, json.dumps(result).encode(), "application/json")

        def log_message(self, *args):
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", action="store_true", help="LLM fallback, hints and critic (needs GEMINI_API_KEY)")
    ap.add_argument("--eedi", action="store_true", help="Eedi misconceptions (needs --llm, a GPU and data/)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if args.eedi and not args.llm:
        ap.error("--eedi needs --llm: the LLM fallback picks the Eedi misconception")
    tutor = Tutor(args.llm, args.eedi)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(tutor))
    print(f"MathEasy analysis page: http://localhost:{args.port}  ({tutor.status()})  Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
