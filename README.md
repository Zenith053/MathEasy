# MathEasy

A multi-agent maths tutor that diagnoses *why* a student's answer is wrong before it gives a hint. See [docs/Proposal.pdf](docs/Proposal.pdf) for the research plan and [docs/Diagnose Before You Hint_ Team Plan.pdf](<docs/Diagnose Before You Hint_ Team Plan.pdf>) for who owns what.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
```

## Calling the LLM

Every model call goes through the shared client in `matheasy/llm`. Don't import the Gemini SDK anywhere else. The client pins the model, enforces each session's token budget, caches deterministic calls, and logs every call for the metrics.

Put your Gemini API key in the environment, then check it works:

```bash
export GEMINI_API_KEY=...
.venv/bin/python -m matheasy.llm
```

Use it like this:

```python
from matheasy.llm import LLMClient, BudgetExceeded

llm = LLMClient()                                      # reads config/llm.yaml
session = llm.session("student-017", condition="full")  # one tutor session

try:
    result = session.call(
        "hinter",                                       # your role, for the logs
        [{"role": "user", "content": "..."}],
        system="...",
    )
    print(result.text)
except BudgetExceeded:
    ...                                                 # fall back to the safe template
```

- **Budget:** a tutor session gets `budget.session_tokens` from the config. Calls outside the tutor (simulated students, LLM judges) use `llm.session(..., budget_tokens=None)` so they don't eat the tutor's budget.
- **JSON replies:** pass `json_schema={...}` and read `result.parsed`.
- **Other models:** `model="large"` picks the H3 comparison model. Names refer to entries under `models:` in `config/llm.yaml`.
- **No API key:** set `provider: fake` in your local copy of the config, or pass `backend=FakeBackend(...)` in tests.
- **Logs:** one JSONL file per run in `runs/llm_logs/`. Every line carries the config hash so we can check all shards used the same settings.

`config/llm.yaml` is shared. Change it only by team agreement.

## Maths tools, rules and diagnosis

Everything in `matheasy/diagnosis` (owner: B). These signatures are fixed; later versions keep the same calls.

```python
from matheasy.diagnosis import parse, equivalent, check_step, locate, run_rule, load_bank, diagnose

parse("1/2 + 1/3")                  # tree in the order the student wrote it; ParseError if unreadable
parse("3 1/2"), parse("25%")        # mixed numbers and percentages; "50% of 20" means 50% × 20
equivalent("2(x + 3)", "2x + 6")    # True: exact, also for equations and comparisons; 25% is 0.25, not 25
check_step("1/2 + 1/3", "2/5")      # False
e = locate("1/2 + 1/3 + x", "2/5 + x")
e.source, e.wrong, e.expected       # '1/2 + 1/3', '2/5', '5/6': the changed part (None if the step is correct)
e = locate("2x + 4 = 10", "x + 4 = 5")
e.wrong_term, e.expected_term       # '4', '2': the exact wrong term ("(nothing)" for a missing or extra term)
e.to_dict()                         # before, after, correct, source, wrong, expected, difference,
                                    # wrong_term, expected_term

run_rule("fractions.add_numerators_denominators", "1/2 + 1/3")   # '2/5'; None if the rule doesn't apply

d = diagnose("1/2 + 1/3", "2/5", student_id="student-017")
d.status, d.misconception           # 'clear', 'rule:fractions.add_numerators_denominators'
d.probe                             # follow-up question to ask, or None (always None until week 3)
d.to_log()                          # JSON-safe record for the run logs
```

- **Misconception IDs:** `rule:<MalruleLib name>` (runnable), `eedi:<MisconceptionId>` (text only), `slip`. See `matheasy/diagnosis/ids.py`.
- **Bank:** `matheasy/diagnosis/bank.yaml`. `load_bank().for_eedi(n)` maps an Eedi misconception to its rules.
- **Held-back rules** in `eval/held_back_rules.json` can never be registered as rules or put in the bank; both refuse.
- **What `diagnose()` needs from A and C** (`matheasy/diagnosis/interfaces.py`): a shortlister `shortlist(error: LocatedError, k) -> list[Candidate]`, and a student store with `misconception_prior(student_id, misconception_id) -> float | None`. Pass them as `diagnose(..., shortlister=..., store=...)`. `EediShortlister()` wraps A's `shortlist()`: it builds the query with A's `build_query()` from `locate()`'s output and returns `eedi:` IDs. It needs A's Eedi data. Without a shortlister, a stand-in ranks the bank's rules that reproduce the answer.
- **Call order:** `check_step()` first; `diagnose()` only on a wrong step (it raises `StepIsCorrect` otherwise). When it returns a probe, ask it and call `diagnose()` again with `follow_ups=[FollowUp(probe, answer)]`; at most 2.
- **Rules list and probe number ranges:** [docs/diagnosis_rules.md](docs/diagnosis_rules.md).
- **Standard notation:** [docs/input_notation.md](docs/input_notation.md). Everything passed to `parse()` (student answers after `transcribe()`, test items, probes) should be written this way; ambiguous forms such as `a^bc` or `x/2y` must use brackets.
- **Reading a student's line:** `transcribe(line, question=..., session=...)` returns it in standard notation, calling the LLM only for lines that are not already standard. Why it works this way: [docs/design_decisions.md](docs/design_decisions.md).

## Simulated students and run logs

Everything in `matheasy/evaluation` (owner: D). The week-1 student handles B's fraction rules.

```python
from matheasy.evaluation import SimStudent, LLMFeedbackJudge

judge = LLMFeedbackJudge(llm.session(session_id, condition, budget_tokens=None))
student = SimStudent("student-001", "fractions.add_numerators_denominators", judge=judge)

reply = student.respond("Try this one: 1/2 + 1/3", item="1/2 + 1/3")
reply.text, reply.answer            # 'I got 2/5.', '2/5'
reply.to_log()                      # ground truth: followed_rule, holds_misconception, ...
student.misconception_id, student.group   # for the session_start record
```

- **The rule decides every answer.** The student keeps its misconception until a tutor message targets it (the judge decides), then answers correctly. Use one `SimStudent` object for all of a student's sessions.
- **`item`:** the runner passes the maths it asked as plain text, so the student never has to read it out of the tutor's prose.
- **Required log fields:** `matheasy/evaluation/log_fields.py` lists every event the runner must write (`session_start`, `turn`, `diagnosis`, `hint`, `tool_call`, `test`, `session_end`), with the fields each needs and the metrics that read them. Check a run with `.venv/bin/python -m matheasy.evaluation runs/`.
- **Tests vs probes:** `overlapping(test_items, probe_items)` lists test items that match a probe (same numbers and operations, in any order).
