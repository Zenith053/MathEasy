# vLLM: an open model on our own GPU, next to Gemini

*Swaraj (A), 14 Nov 2026. For B, C and D.*

**In short:** the tutor can now run on an open model (Qwen) served by vLLM on our own GPU, as well as on Gemini. The project's runs stay on Gemini: in a fair test, Gemini was as accurate or more accurate on every measure. vLLM is there for the showcase, and as a backup with no API key and no quota. `config/llm.yaml` is unchanged, so its hash and every logged run are unaffected.

## What I added

| File | What it does |
|---|---|
| `matheasy/llm/backends.py` | `VLLMBackend`: sends requests to any OpenAI-compatible server (vLLM) and reads the replies, keeping the JSON-schema replies. |
| `matheasy/llm/config.py` | Provider `vllm` with a `base_url`. The `MATHEASY_LLM_CONFIG` environment variable picks a config file for every program, with no code changes. |
| `config/llm_vllm.yaml` | The same settings as `llm.yaml`, except the provider, server, model and timeout. |
| `matheasy/ui/` | `python -m matheasy.ui --vllm`: a "vLLM on" badge, and "LLM" now means Gemini. Only one of the two is on at a time. |
| `eval/llm_compare.py` | The Gemini-vs-vLLM test: `run` writes a scores-only JSON, and `compare` gives paired intervals. |
| `tests/test_vllm_backend.py` | 8 tests. The full suite passes (1317). |

**C:** this touches your `matheasy/llm` code. Please review `VLLMBackend` and the config changes. The Gemini path has not changed.

## How to use it

```
python -m matheasy.ui --llm                      # Gemini   -> badges: LLM on,  vLLM off
python -m matheasy.ui --vllm                     # vLLM     -> badges: LLM off, vLLM on (page on port 8001)
python -m matheasy.ui --vllm --vllm-model <id>   # another model, e.g. a smaller one for a laptop
MATHEASY_LLM_CONFIG=config/llm_vllm.yaml python -m matheasy.tutor.runner ...   # any other program
```

`--vllm` needs a running vLLM server, and the page refuses to start without one. vLLM runs only on Linux. On Windows that means WSL (see the setup section below).

## Accuracy: Gemini vs vLLM

Only the model changed; everything else was held fixed:
- the same prompts, temperature 0 and seed;
- the student side fixed: the keyword judge and template attacks;
- no textbook and no name groups.

Gemini ran on my laptop. Qwen2.5-7B-Instruct-AWQ ran on vLLM 0.8.5 on a Kaggle T4. Both used the v2 shortlister.

| Test | Gemini 3.1 flash-lite | Qwen 7B on vLLM |
|---|---|---|
| Right topic for a step no rule explains (22) | **19** | 13 |
| Right Eedi name picked (22) | **5** | 3 |
| Tutor top-1, all 46 students | **0.69** | 0.62 |
| Tutor top-1, text-only students | **0.24** | 0.12 |
| Tutor top-1, has_rule students | 1.00 | 1.00 |
| Slip called a misconception (slip_only, 2 students) | **0.25** | 1.00 |
| Hints that leaked the answer | 0 | 0 |
| Replies that weren't valid JSON | 2 of 724 | 1 of 637 |

- **The topic test is the only borderline-significant difference.** The paired difference was +0.27, with a 95% bootstrap interval of [0.05, 0.50], but the permutation p-value was 0.07.
- **The other differences point the same way**, but there are too few cases to show a significant difference.
- **Where the gap appears:** where the LLM decides (topics, Eedi picks, text-only students). Where rules decide, both models score the same.
- **The limits of this test:** a small n, one session per student, one open model, 4-bit, on a free T4.

The scores are in `runs/scores-gemini.json` and `runs/scores-vllm.json`. They're git-ignored and contain no Eedi IDs or names.

## Latency

| Setup | What was timed | Calls | Median | Range |
|---|---|---|---|---|
| Gemini 3.1 flash-lite (Google API) | the comparison run's real project calls | 724 (200 new) | **2.47 s** | |
| Qwen 7B AWQ, vLLM on a Kaggle T4 | the comparison run's real project calls | 637 (179 new) | **3.30 s** | |
| Gemini 3.1 flash-lite (Google API) | 10 short JSON hints, today | 10 | **2.62 s** | 1.03-6.63 s |
| Qwen 3B AWQ, vLLM on a laptop RTX 4050 | the same 10 short JSON hints, today | 10 | **0.74 s** | 0.47-4.46 s |

All 20 of today's replies were valid JSON.

- **On real project calls, Gemini was a bit faster than the 7B model on a T4.** A T4 is an old, slow GPU.
- **For short replies, the 3B model on the laptop is about 3× faster than Gemini.** There's no network trip, the model is smaller, and the 4050 is newer than a T4.
- **The cost of that speed:** the 3B model is smaller than the 7B one I tested, and I haven't measured its accuracy. Expect it to be weaker than Gemini.
- Gemini's time also depends on Google's load and the free tier's limits, so expect slow outliers.

### A full Analyze on the page

One Analyze makes several LLM calls: reading the lines, the diagnosis, the hints and the critic's review. I timed the page's own Analyze on 6 wrong answers with each model, once each, today, with no cache:
- the page's 4 examples;
- `3/4 - 1/4 → 2/0`;
- `2(x + 3) → 2x + 3`.

| Example | Gemini: s (calls) | Qwen 3B, laptop vLLM: s (calls) |
|---|---|---|
| 1/2 + 1/3 → 2/5 | 12.3 (2) | 8.0 (2) |
| 2x + 3 = 11, two lines | 34.9 (3) | 26.9 (6) |
| (x + 3)² − x² | 48.4 (2) | 7.2 (6) |
| 7 × 8 + 6 → 60 | 41.1 (7) | 5.0 (3) |
| 3/4 − 1/4 → 2/0 | 185.0 (3) | 4.1 (3) |
| 2(x + 3) → 2x + 3 | 3.6 (2) | 4.4 (4) |
| **Median / total** | **38 s / 325 s** | **6 s / 56 s** |

- **vLLM was much faster here.** Most of Gemini's time was spent waiting on Google, not working: one example took 185 s for 3 calls. The free tier was busy, and earlier tests had already used some of today's quota. On a good day, expect Gemini at roughly 2.5 s per call.
- **Gemini's hints were better.**
  - On Gemini, all 6 answers got a specific hint that passed the critic.
  - On the 3B model, the critic rejected every hint for 2 of the 6, so the student got the generic "Let's slow down…" message.
  - One of the 3B model's hints was off target: for `2/0` it talked about dividing by zero.
- This is one run of 6 examples, so treat it as a demo figure, not a measurement.

## Running it on a Windows laptop (what I did)

1. Install WSL with `wsl --install --web-download` in an admin Command Prompt. The plain `wsl --install` failed with "Forbidden (403)".
2. In Ubuntu:
   - `sudo apt install python3-venv build-essential`. vLLM needs a C compiler.
   - `python3 -m venv ~/vllm-env`.
   - `pip install vllm`.
3. Start the server in Ubuntu:
   ```
   VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen2.5-3B-Instruct-AWQ --port 8000 \
       --max-model-len 4096 --gpu-memory-utilization 0.6 --generation-config vllm
   ```
   - `VLLM_USE_FLASHINFER_SAMPLER=0` is needed without the CUDA toolkit (`nvcc`).
   - `--generation-config vllm` makes the server use our settings, not Qwen's own defaults.
4. In Windows: `python -m matheasy.ui --vllm --vllm-model Qwen/Qwen2.5-3B-Instruct-AWQ`.

**What caught me out:**
- **Disk space:** the vLLM install takes about 9 GB in WSL and the 3B model about 2 GB, all on C:. Ubuntu's `/tmp` is in memory and holds about 4 GB, so install with `TMPDIR=~/tmp pip install --no-cache-dir vllm`.
- **Downloads:** WSL's network freezes on long downloads; use `--resume-retries 10`.
- **GPU memory:** a 6 GB GPU can't hold the 7B model next to the Eedi shortlister. Use the 3B model and leave Eedi off, or use Gemini for Eedi demos.
- **On a T4 (Kaggle):** use `vllm==0.8.5`. Newer versions fall back to an older engine there that ignores the JSON schema, and most replies stopped being JSON.

## What I'd like from the team

- **C:** review the backend and config changes, and say if you'd rather they lived elsewhere.
- **All:**
  - Agree that the final runs stay on Gemini.
  - Decide whether the report should mention vLLM. It could be a short "open-model alternative" paragraph using the tables above.
