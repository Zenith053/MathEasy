# Evaluation: week-3 deliverables and what C needs to connect

Owner: D. What the evaluation now provides, how to run it, and the small changes C's runner needs to use it. D does not edit the tutor or runner code; C owns those changes.

## What is new

| Piece | Where | What it does |
|---|---|---|
| Pre/post tests | `evaluation/prepost.py` | For each student and session, test items that make the same mistake show up with new numbers. Each pre-test item has a parallel post-test item. Items never match a probe, a tutored item or each other. Next session's pre-test is the retention check |
| Adversarial students | `evaluation/adversarial.py` | Zhao et al. (2026)'s six techniques: direct request, emotional threat, intentional wrong answer, contextual manipulation, interpersonal influence, request shaping. They still answer by their rule. Attack lines have no digits, and the answer comes last |
| Metrics | `evaluation/metrics.py` | Added: normalised learning gain, retention across sessions, relapse rate, calibration of the final belief (ECE and Brier), topic accuracy, turns to first leak. Adversarial sessions are reported in their own `attack:<technique>` groups |
| Statistics | `evaluation/stats.py` | 95% intervals that resample students, paired comparisons between conditions (bootstrap interval and permutation p-value), Holm correction |
| Judges and labelling | `evaluation/judges.py`, `evaluation/labels.py`, `docs/labelling_protocol.md` | LLM judge for B0's text diagnosis; blind labelling sheets; agreement; judge-vs-human checks |
| LeetCode-style analysis metrics | `evaluation/metrics.py` | For C's `hint.analysis`: how often each part is shown, whether the topics shown include the planted one, and `practice_relevance`: the share of recommended practice problems on which the student's own misconception gives a wrong answer (checked by running the rule) |
| Results report | `evaluation/report.py` | One command turns the merged logs into the Markdown tables for the results section |
| Eedi matching | `eval/match_eedi.py` | Suggests Eedi misconceptions for each held-back rule and records the decision in `eedi_status` |
| Checker | `evaluation/log_fields.py` | Now also fails a run whose test items match any probe |
| Open questions from `design_decisions.md` | `items.py`, orchestrator | `item_key()` handles `Mixed` and `Word`. Clarification questions are already outside `follow_ups`, because the orchestrator counts probes only |

## Changes for C's runner (`tutor/runner.py`, `tutor/session.py`)

1. **Use D's tests.** Since 38644a7 the runner's stand-in tests leave out probes. They are still the same items the tutor teaches on, and the pre- and post-test are identical, so the post-test measures recall of tutored problems, not the idea on new numbers. Replace `materials()`' test items with:

   ```python
   from matheasy.evaluation import prepost_forms
   forms = prepost_forms(student.misconception, sessions=args.sessions, student_id=student.student_id,
                         avoid=pool)                     # pool: the problems the tutor teaches on
   # session k: pre_items=forms[k].pre, post_items=forms[k].post
   ```

2. **Log the attack.** In `run_session()`, write `attack=getattr(student, "attack", None)` instead of `attack=None` in `session_start`.

3. **Adversarial sessions.** Run them in every shard and under every condition, like ordinary students:

   ```python
   from matheasy.evaluation import adversarial_roster, LLMAttackLine
   students = adversarial_roster(judge, per_attack=6)[i::n]
   # with --llm, optionally: student.attack_line = LLMAttackLine(client.session(sid, condition, budget_tokens=None))
   ```

   Pre/post tests are optional for them. If asked, test questions must start with `"Test question"` (as now), so the student answers them plainly.

4. **Retention needs relapse.** Pass `relapse_rate` to `roster()` (D suggests 0.3) for multi-session runs. Otherwise a fixed misconception never comes back, and retention is 1 by construction.

## How to run

```bash
python -m matheasy.evaluation runs/final                         # checker (fields, held-back leaks, tests vs probes)
python -m matheasy.evaluation.metrics runs/final/turns           # every metric, per condition and group
python -m matheasy.evaluation.stats runs/final/turns             # intervals and full vs each baseline
python -m matheasy.evaluation.labels sample runs/final/turns --out labels/
python -m matheasy.evaluation.report runs/final --out docs/results.md   # the results tables for the report
```

## The LeetCode-style plan (design_decisions.md, 30 Oct)

The new plan drops H1–H3. The metrics above still measure what the tutor shows the student:

| What the tutor shows | Metric |
|---|---|
| What the mistake is | `top1_accuracy`, `over_claiming_rate` |
| Where it went wrong | `where_rate` |
| Which topics it involves | `topic_accuracy` and `analysis_topic_accuracy`, for every student group, held-back students included |
| What to practise | `practice_rate`, `practice_relevance` |
| Whether the student then gets it right | `learning_gain`, `fixed_rate`, `retention` |
| That the answer is never given away | `leak_rate`, and per attack technique |

Calibration and the comparisons with B0, B1 and B2 matter only if the baselines stay. That is still open in `design_decisions.md`, and the team needs to decide.

**Not done:** `eedi_status` in `eval/held_back_rules.json` is still empty. It needs the Eedi `misconception_mapping.csv` (Kaggle, under the competition terms) in `data/eedi/raw/`. Then run `eval/match_eedi.py suggest`, decide each rule, and record it with `eval/match_eedi.py set`. Until then, the ten held-back students count as `unchecked`.
