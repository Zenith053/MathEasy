"""LLM judges used by the evaluation (owner: D), and the questions human labellers answer.

Every metric that needs an LLM to read free text has a judge here (or in
students.py), and the same yes/no question is put to human labellers, so the judge
can be checked against them before its metric is used (labels.py, Team Plan
section 7, proposal section 12):

    kind            judge                      question                                    metric it feeds
    targets         students.LLMFeedbackJudge  Does the message target the misconception?  how sim students learn (H2)
    diagnosis_text  LLMTextJudge               Does the stated diagnosis describe it?     B0's top-1 accuracy (H1)
    leak            metrics.states_answer      Does the message give the answer away?     leak rate (safety)
    citation        (human only)               Does the cited section support the hint?   citation quality (RAG)

The leak metric is code, not an LLM; the human labels check that code. Every
judge call goes through the shared LLM client on a session with budget_tokens=None,
so judging never spends a tutor's budget, and is logged under its own role.
"""

from __future__ import annotations

QUESTIONS = {
    "targets": "Does the tutor's message target this misconception: does it explain why this way of working is "
               "wrong, or teach the correct method for exactly the step the misconception gets wrong? "
               "Encouragement, 'check your work', a hint about a different mistake, or only stating the answer "
               "do not count.",
    "diagnosis_text": "Does the tutor's stated diagnosis describe the same misconception as the planted one "
                      "(the same wrong way of working, even in different words)?",
    "leak": "Does the tutor's message give away the final answer to the open question, before the student has "
            "reached it? Restating the question, or numbers that are in the question, do not count.",
    "citation": "Does the cited textbook section support what the hint says (it teaches the method the hint "
                "points to)?",
}

_TEXT_SYSTEM = """\
You check a maths tutor's diagnosis in a research study. A simulated student holds this \
misconception:

{misconception}

The tutor wrote what it thinks the student's mistake is. Decide whether the tutor's \
diagnosis describes the same misconception: the same wrong way of working, even if worded \
differently. A diagnosis that is vaguer ("a fraction error"), names a different wrong \
method, or only says the answer is wrong does not count."""

_SCHEMA = {"type": "object", "properties": {"same": {"type": "boolean"}}, "required": ["same"]}


class LLMTextJudge:
    """metrics.TextJudge for B0's free-text diagnosis: text_judge(stated, description) -> bool.

        judge = LLMTextJudge(client.session("judge-b0", "b0", budget_tokens=None))
        metrics.compute(records, text_judge=judge)

    metrics.compute() also asks whether a statement describes "a careless slip" (metrics.SLIP_TEXT), to
    score B0's over-claiming. Answers are cached, so the same statement is judged once."""

    role = "eval_text_judge"

    def __init__(self, session):
        self.session = session
        self._cache: dict[tuple[str, str], bool] = {}

    def __call__(self, stated: str, description: str) -> bool:
        key = (stated.strip(), description.strip())
        if key not in self._cache:
            result = self.session.call(
                self.role,
                [{"role": "user", "content": f"Tutor's diagnosis:\n{stated}"}],
                system=_TEXT_SYSTEM.format(misconception=description),
                json_schema=_SCHEMA,
            )
            self._cache[key] = bool(result.parsed.get("same"))
        return self._cache[key]


__all__ = ["LLMTextJudge", "QUESTIONS"]
