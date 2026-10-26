"""Week-1 end-to-end check: python -m matheasy.tutor [--llm]

Runs one session of D's simulated student (holding B's first fraction rule)
through the tutor, then checks the run log with D's checker. By default the
student's feedback judge is a keyword match so no API key is needed; --llm uses
D's LLM judge through the shared client (needs GEMINI_API_KEY).
"""

import re
import sys

from ..evaluation import LLMFeedbackJudge, SimStudent, check_files
from ..llm import LLMClient
from . import Orchestrator, TurnLogger, make_problem, run_session
from .placeholders import BasicCritic, InMemoryStore, TemplateHinter

RULE = "fractions.add_numerators_denominators"


def keyword_judge(message: str, misconception: str) -> bool:
    """Demo only: a message targets the misconception if it quotes its description."""
    return re.split(r"[:,]", misconception)[0].lower() in message.lower()


def main() -> None:
    llm = LLMClient(run_id="demo")
    session_id = "demo-session"
    judge = (LLMFeedbackJudge(llm.session(session_id, "full", budget_tokens=None))
             if "--llm" in sys.argv else keyword_judge)
    log = TurnLogger(run_id="demo", config_hash=llm.config.hash)
    log.path.unlink(missing_ok=True)

    tutor = Orchestrator(TemplateHinter(), BasicCritic(), InMemoryStore(), log)
    student = SimStudent("demo-student", RULE, judge=judge)
    result = run_session(
        tutor, student, [make_problem("p1", "1/2 + 1/3"), make_problem("p2", "2/3 + 1/4")],
        session_id=session_id, condition="full", session_index=0,
        pre_items=["1/4 + 2/5"], post_items=["2/7 + 1/3"], llm_session=llm.session(session_id, "full"),
    )

    for u in result.state.transcript:
        who = f"tutor ({u.kind})" if u.speaker == "tutor" else "student"
        print(f"{who:>22}: {u.text}")
    print(f"\npre-test {result.pre_correct}/1, post-test {result.post_correct}/1, ended by {result.ended_by}")
    problems = check_files([log.path])
    print(f"log: {log.path}\nD's checker: {'logs OK' if not problems else problems}")


if __name__ == "__main__":
    main()
