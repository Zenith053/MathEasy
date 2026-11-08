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

## Trial with the LLM (run by A, 7–8 November)

```bash
python -m matheasy.tutor.runner trial-llm-2 --llm --conditions full,b0,b1,b2 --shard 0/6 --sessions 2
```

The plan was 18 students (12 from the roster and one per attack technique), 2 sessions each, under all four conditions: 144 sessions.

**Stopped at 63 sessions: full 36, B0 27, B1 0, B2 0.** The free tier's daily limit ended it. Every finished record passes the checker, with one config hash.

### What the numbers say (one student per group, so not results)

| | full | b0 |
|---|---|---|
| Leak rate, ordinary students | 0.00 | 0.08 [0.00, 0.21] |
| Leak rate under attack | 0.00 under all six techniques | 0.00 under the two reached |
| Learning gain, students with a rule | 0.87 [0.60, 1.00] | 0.75 [0.47, 1.00] |
| Top-1 diagnosis, the student known only as text | 0.00 | – |
| New API requests per session (not from the cache) | 7.6 | 10.0 |
| Tutor tokens per session: median, max | 1,381, 21,080 | 3,565, 9,310 |

- **Hints cite the textbook:** 34 of the full tutor's 48 hints carry a citation from `retrieve()`, and the other 14 are safe templates.
- **The 0% finding above holds with the LLM.**

### Findings for the owners

| Finding | Owner |
|---|---|
| **The final runs do not fit the free tier.** Its limits are 15 requests a minute and 500 a day, shared by every key in one Google project. With about 214 sessions per condition and 7.6–10 new requests per session (B1 not measured), the final runs need about 6,000–7,000 requests: about 13 project-days. The options are billing, or fewer students or conditions. | Team |
| **One quota error or dropped connection ends the run.** `LLMCallError` propagates out of the runner. The client's retries did not get past a per-minute limit whose error said to retry in about 30 seconds. Rerunning the same command resumes and loses only the unfinished session, which is how this trial was finished. | C |
| **4 of 23 full-tutor sessions went over the 20,000-token session budget** (largest 21,080). Is the budget enforced, and should it change now that there is a measurement? | C |
| **B0's top-1 and topic accuracy are blank** in the report, though B0's logs hold 190 diagnosis records. Is B0's diagnosis meant to be scored? | C and D |
| **`retrieve()`'s similarity is not logged,** so whether `is_weak()` would have caught a bad citation can't be checked from the logs. | A and C |
| **A laptop that sleeps pauses the run.** Most of a night was lost here. Runs need a machine that stays awake. | Everyone |

## Next

- B1 still needs a trial with `--llm`, on a few students: about 130 requests.
- B2 passed offline, and its LLM calls are the full tutor's, mostly served from the cache.
- Each member first tests their API key with the runner.
- The quota decision comes before the freeze.
