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

## Textbook search and the Eedi shortlister

Everything in `matheasy/retrieval` and `matheasy/shortlister` (owner: A). Results are in `docs/report_A_draft.md`.

### Textbook search: build it before any run

The hinter cites OpenStax *Prealgebra 2e* and *Elementary Algebra 2e* (CC BY 4.0) through `retrieve()`. The passages are not in git. **Each machine builds them once:**

```bash
.venv/bin/python -m matheasy.retrieval.openstax                    # -> data/openstax/passages.jsonl (1,272 passages)
.venv/bin/python -m matheasy.retrieval.retrieve "adds the numerators and the denominators"   # check: prints passages
```

**If the file is missing, nothing fails, but hints quietly lose their citations.**
- The runner and the page check only whether the file exists (`textbook_retrieve()` returns `None` without it), so a missing file shows no error.
- In the final runs, build it on every machine first, or that shard is not comparable with the others.
- The book versions are pinned in `openstax.py`, so every machine gets the same passages.
- The page's status line shows `"textbook": true` when the file is found.

`retrieve(query, k=3)` returns passages with their reference, score and `similarity`. `is_weak(passages)` flags a match that is probably off-topic. `python -m matheasy.retrieval.retrieve --eval` prints the right-section rate (hybrid: 70% first, 97% in the top 3, on 30 queries).

### Eedi data and the shortlister: only for `--eedi`

The tutor runs without them. They are needed only to link a step no rule explains to an Eedi misconception (`--eedi` in the page and the runner), or to retrain or score the shortlister.

**Eedi data.** Accept the competition rules on your own Kaggle account, then:

```bash
kaggle competitions download -c eedi-mining-misconceptions-in-mathematics -p data/eedi/raw   # unzip it there
.venv/bin/python -m matheasy.shortlister.prepare_eedi    # subset and split (1,038 / 145 / 289 wrong answers)
.venv/bin/python -m matheasy.shortlister.query           # training queries
```

**The trained shortlister (v2).** Put the adapter folder at `data/shortlister/adapter-3B-v2/`. It is shared privately in the team, never in git, because it is built from Eedi data.
- `--eedi` needs a CUDA GPU (about 2 GB) and PyTorch. Without them, `load_ranker()` returns `None`, and diagnosis runs without Eedi names.
- The first start embeds the candidate names, which takes about 8 minutes on a 4 GB GPU. Later starts reuse them (`docs/design_decisions.md` 7.5).

**Never commit or re-upload Eedi-derived files:** raw data, `data/eedi/*`, adapters, name vectors or name groups. `data/` is git-ignored. Only Eedi question IDs (`eedi_split.json`) and misconception IDs (`candidates.json`, `bank.yaml`) are in the repo.

Training and scoring run on Kaggle GPUs: `shortlister/train_shortlister.ipynb` trains, `score_test.py` and `score_live.py` score.

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
d.probe                             # follow-up question to ask when rules tie, or None
d.topics                            # every topic involved, whether rules or the LLM found the mistake
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

Everything in `matheasy/evaluation` (owner: D).

```python
from matheasy.evaluation import SimStudent, LLMFeedbackJudge, roster

judge = LLMFeedbackJudge(llm.session(session_id, condition, budget_tokens=None))
student = SimStudent("student-001", "fractions.add_numerators_denominators", judge=judge)

student.begin_session(session_index)    # before each session (run_session() does this)
reply = student.respond("Try this one: 1/2 + 1/3", item="1/2 + 1/3")
reply.text, reply.answer            # 'I got 2/5.', '2/5'
reply.to_log()                      # ground truth: followed_rule, holds_misconception, slipped, ...
student.misconception_id, student.group   # for the session_start record

students = roster(judge, copies=1, slip_only=10)   # one student per rule, plus slip-only students
student.save("runs/students"); SimStudent.load("runs/students", "student-001", judge=judge)
```

- **The rule decides every answer.** The student keeps its misconception until a tutor message targets it (the judge decides), then answers correctly.
- **Any rule, by name:** B's rules run through `run_rule()`; the 10 held-back rules, which B's runner refuses, through D's code in `matheasy/evaluation/held_back.py` (MalruleLib's wrong answers). The tutor must never import that module.
- **Slips:** `slip_rate` is the chance of a careless slip on an item the student would otherwise get right; slip-only students (`misconception=None`) use 0.3. A slip is never an answer any rule gives, so it can't look like a misconception.
- **Several sessions:** C's `run_session()` calls `begin_session(index)` before each session and logs `relapsed` in `session_start`. Keep the same object, or `save()` and `load()` it between processes; a loaded student carries on exactly as if it had never stopped. `relapse_rate` (default 0) is the chance a fixed misconception comes back at the start of a later session.
- **`item`:** the runner passes the maths it asked as plain text, so the student never has to read it out of the tutor's prose. Equations are answered `x = 4`. Comparisons are written as a relation (`1/3 > 1/4`, sign ignored) and answered with the relation the student believes; that format still needs agreement with B and C.
- **Metrics:** `python -m matheasy.evaluation.metrics runs/turn_logs` prints every week-2 metric per condition and student group: top-1 diagnosis accuracy, follow-ups per diagnosis, over-claiming on slips, claims on misconceptions the tutor can't know, leak rate, unsafe-action rate, critic rewrites and safe templates, tool-call validity and recovery. Learning gain, retention and calibration come in week 3.
- **Required log fields:** `matheasy/evaluation/log_fields.py` lists every event the runner must write (`session_start`, `turn`, `diagnosis`, `hint`, `tool_call`, `test`, `session_end`), with the fields each needs and the metrics that read them. Check a run with `.venv/bin/python -m matheasy.evaluation runs/`.
- **Tests vs probes:** `overlapping(test_items, probe_items)` lists test items that match a probe (same numbers and operations, in any order).

## The tutor, baselines and session runner

Everything in `matheasy/tutor` (owner: C). Run a session end to end with `.venv/bin/python -m matheasy.tutor` (offline) or `--llm` (Gemini).

The tutor answers a wrong answer LeetCode style (see "What we are trying to build" in [docs/design_decisions.md](docs/design_decisions.md)):

```
Where it went wrong: 1/2 + 1/3 became 2/5.
The mistake: Adds the numerators and adds the denominators, e.g. 1/2 + 1/3 = 2/5.
Topics: fractions.
<the hinter's hint, citing a textbook section>
Practise next: 6/12 + 3/6 + 3/7; 7/9 + 4/5 + 4/6.
```

The four parts come from B's diagnosis in code (`analysis.py`). Each is left out if it would give away the answer to any item still open in the session. Whether the baselines and the runner stay is not settled yet; they are kept until the team decides.

Experiments go through the runner: any conditions, on one shard of students, over several sessions. Shards split students, never conditions, and a stopped run carries on when the same command is run again. Each shard includes D's adversarial students (`--per-attack`, 6 per technique by default), students relapse between sessions (`--relapse`, default 0.3), and pre/post tests are D's `prepost_forms()`: the same ideas with new numbers, never a tutored item or a probe.

```bash
.venv/bin/python -m matheasy.tutor.runner trial --conditions full,b2 --sessions 2                # offline, ~90 s
.venv/bin/python -m matheasy.tutor.runner final --llm --shard 0/4 --conditions full,b0,b1,b2      # one member's shard
.venv/bin/python -m matheasy.evaluation runs/final && .venv/bin/python -m matheasy.evaluation.metrics runs/final/turns
.venv/bin/python -m matheasy.evaluation.stats runs/final/turns                                      # D's intervals and comparisons
```

```python
from matheasy.llm import LLMClient
from matheasy.retrieval import retrieve
from matheasy.tutor import (Orchestrator, B0Tutor, LLMHinter, TutorCritic, JsonStudentStore,
                            TurnLogger, make_problem, run_session)

llm = LLMClient()
log = TurnLogger(run_id="run-1", config_hash=llm.config.hash)    # runs/turn_logs/run-1.jsonl

full = Orchestrator(LLMHinter(retrieve), TutorCritic(), JsonStudentStore("runs/students.json"), log)
b2 = Orchestrator(LLMHinter(retrieve), TutorCritic(), JsonStudentStore("runs/students-b2.json"), log, mode="b2")
b0 = B0Tutor(log)

result = run_session(full, student, [make_problem("p1", "1/2 + 1/3")],
                     session_id="s-001", condition="full", session_index=0,
                     pre_items=[...], post_items=[...], llm_session=llm.session("s-001", "full"))
result.ended_by, result.pre_correct, result.post_correct
```

- **One session** = pre-test, tutoring on each problem, post-test, with one token budget (`llm_session`). It writes D's `session_start`, `turn`, `test` and `session_end` records; the tutor writes `diagnosis` and `hint`.
- **Every condition reads input the same way** (`read_answer()`): code finds the maths in the message, then B's `transcribe()` reads what the student meant (a model call only for a line not in standard notation). An ambiguous line is asked about once. Offline there is no transcriber, so the parser's literal reading is used.
- **B's LLM fallback:** the orchestrator passes `question` and the session to `diagnose()`. When no rule explains a step, the hinter aims at the fallback's reading, but only if code verified it, it is a misconception, and its words don't give the answer away.
- **Every full-tutor message passes `TutorCritic`**: code checks first (leak check with B's `equivalent()`), then an LLM review of hints. After 2 rejected rewrites, or when the budget runs out, a safe template is sent.
- **The hinter never sees a correct answer**: `HintRequest` has no field for one. It cites one of the textbook sections A's `retrieve()` returns.
- **Tool wrappers** (`ToolBox(standard_tools(retrieve), log)`) check arguments against a schema, retry an unexpected error once, and log D's `tool_call` record.
- **The planner** (`planner.py`, behind the student store) keeps, per topic (B's labels), P(knows skill) and a count of mistakes, so `weak_topics()` and `mastered()` say what the student has and has not mastered. It also keeps P(holds misconception) per misconception, and re-checks a fixed one in the next session, then after 2, 4, ... sessions. `misconception_prior()` gives B's diagnoser P(holds). `planner.plan()` picks each session's problems: re-check items first, then one practice problem from B's probe bank on the weakest topic, then the pool, weakest topic first.
- **B0** has no tools, retrieval, memory or critic; its diagnosis is free text in `stated_misconception`.
- **B1** (`B1Agent(log, ToolBox(standard_tools(retrieve), log))`) is one LLM agent calling the same tools itself (at most 4 per message), with no hypothesis testing, memory or critic. It names a misconception by B's rule name (`list_rules` tool); a name that is not one of B's rules is logged as unknown.
