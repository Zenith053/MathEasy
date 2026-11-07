# Week-3 trial run

Owner: D. This is the Team Plan's week-3 gate: "a trial run on a few students passes D's log checker". Run on 7 November 2026, offline, with no API key: the template hinter, the critic's code checks, and students that judge feedback by keyword.

```bash
python -m matheasy.tutor.runner trial --conditions full,b2 --sessions 2     # 85 s
python -m matheasy.evaluation runs/trial                                    # logs OK
python -m matheasy.evaluation.report runs/trial --out runs/trial/results.md
```

**Gate: passed.**

- 428 sessions: the whole roster and 36 adversarial students (6 per technique), 2 sessions each, under the full tutor and B2.
- Every record passes the checker, and no test item matches a probe.
- One config hash across the whole run.
- Held-back students are now split by `eedi_status`: 8 known only as text, 2 never seen.

B0 and B1 need `--llm`, so they are not in this trial.

## What the numbers say (offline, so not results)

| | full | b2 |
|---|---|---|
| Top-1 diagnosis, students with a rule | 0.98 [0.92, 1.00] | 0.68 [0.51, 0.85] |
| Practice that exercises the student's own mistake | 1.00 | 0.89 |
| Learning gain, students with a rule | 0.59 [0.46, 0.71] | 0.52 [0.39, 0.65] |
| Calibration error (ECE) | 0.08 | 0.13 |
| Leak rate, ordinary and under every attack | 0.00 | 0.00 |

The paired difference in top-1 accuracy on students with a rule is +0.30 [0.11, 0.46], from probes settling ties that B2 guesses. It is not significant after Holm correction (0.24) at this size.

## Findings for the owners

| Finding | Owner |
|---|---|
| **Students "known only as text" get top-1 accuracy 0.00 and topic accuracy 0.10 in both conditions.** No rule fits them, so the tutor must name an `eedi:` misconception through the shortlister, and offline nothing reaches it (the diagnoser uses `RuleShortlister`). Until the trained shortlister or the shortlist cache is connected to `diagnose()`, the held-back groups can only score 0. | A and B |
| **Leak rate 0 under all six attacks is expected offline:** the template hinter cannot leak. The leak test needs a run with `--llm`. | C (to run), D (to check) |
| **Comparisons must stay within groups.** The full tutor fixes students faster, so it has fewer episodes on easy students. That makes its pooled `all` figures lean on the hard groups: `topic_accuracy` over `all` looked 0.10 worse while no group was worse by more than 0.01. `report.py` now compares within groups only. | D (done) |

## Next

A trial with `--llm` on a few students per condition, with B0 and B1 included, before the freeze. Each member first tests their API key with the runner.
