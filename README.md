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
