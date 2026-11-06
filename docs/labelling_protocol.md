# Labelling protocol

Owner: D. For week 4: every member labels a share of the tutors' outputs, blind to which condition produced them. The labels check the parts of the evaluation that read free text, before any of their numbers are reported (Team Plan section 7: "any metric judged by an LLM is checked against human labels before it is used").

Code: `matheasy/evaluation/labels.py`. Questions and judge prompts: `matheasy/evaluation/judges.py`.

## What is labelled

Each task is one yes/no question about one piece of tutor output.

| Kind | Question (as on the sheet, shortened) | What it checks |
|---|---|---|
| `leak` | Does the message give away the final answer to the open question, before the student reached it? | The leak metric's code (`metrics.states_answer`) |
| `targets` | Does the message target the planted misconception (explain why the method is wrong, or teach the right method for that step)? | The simulated students' LLM judge (`LLMFeedbackJudge`), which decides when a student learns |
| `diagnosis_text` | Does the tutor's stated diagnosis describe the planted misconception? | B0's text judge (`LLMTextJudge`), which scores B0's diagnosis |
| `citation` | Does the cited textbook section support the hint? | Citation quality (human only) |

Default sample per condition: 40 `leak`, 40 `targets`, 30 `diagnosis_text`, 20 `citation`. For `leak`, half of the tasks come from turns the code flags as leaks and half from the rest, because leaks are rare and a random sample would have too few to check.

## Steps

1. **D makes the sheets** after the final runs are merged:

   ```bash
   python -m matheasy.evaluation.labels sample runs/final/turns --out labels/
   ```

   This writes `labels/A.csv` … `labels/D.csv` and `labels/key.jsonl`.

2. **D sends each member their own CSV only.** The key holds the condition, session and turn of each task. Nobody opens the key until all labels are in.

3. **Each member fills in the `label` column** with `yes` or `no`, using the `note` column for anything unclear. Rules:
   - Answer the question on the sheet only, from the context on the sheet.
   - Do not guess the condition, and do not look the session up in the run logs.
   - Leave `label` blank if you cannot decide. Blank labels are left out; do not write "maybe".
   - Work through the sheet in its given order. It is already shuffled.

4. **About 20% of tasks appear on two members' sheets**, so agreement can be measured. Label them independently, without discussing them.

5. **D computes agreement and checks each judge:**

   ```bash
   python -m matheasy.evaluation.labels agree labels/                          # Cohen's kappa between members
   python -m matheasy.evaluation.labels check labels/ --judge leak             # the leak code against humans
   python -m matheasy.evaluation.labels check labels/ --judge targets          # LLM judge (needs GEMINI_API_KEY)
   python -m matheasy.evaluation.labels check labels/ --judge diagnosis_text   # LLM judge (needs GEMINI_API_KEY)
   ```

## Decision rule

- **Agreement between members:** report Cohen's kappa per kind. Below 0.6, the question is unclear. Revise its wording in `judges.QUESTIONS` before trusting either the humans or the judge.
- **A judge is used for a reported metric only if its kappa with the human labels is at least 0.7** (`labels.JUDGE_KAPPA_MIN`). Zhao et al. (2026) report 0.81–0.88 for their leak judges.
- **If a judge fails:** report the metric from the human-labelled sample only, with its interval, and say so in the report.
- Tasks where two members disagree are left out of the judge check and listed in the report's appendix.

## Workload

About 520 tasks for four conditions, with 20% double-labelled, comes to roughly 155 per member. At about 20 seconds a task, that is under an hour each.
