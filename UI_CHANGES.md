# The analysis page: what changed and how to run it

*Swaraj (A), 14 Nov 2026, commit `57a2192`. For B, C and D.*

**In short:** the analysis page (`python -m matheasy.ui`) is now a practice workspace:
- a problem set on the left;
- an editor for the working on the right;
- live progress while it analyses;
- separate views for students and teachers.

I also fixed four problems found in testing: words like "hello" were read as maths, the mistake card could be empty, waits could reach 30 s, and the generic-hint question needed measuring. All 1320 tests pass.

---

## How to run it

First, once only, create accounts:
```
python -m matheasy.ui.db add-user mr_khan --role teacher
python -m matheasy.ui.db add-user anna
```
Anyone can also sign up as a student on the page.

| What you want | Command | Open |
|---|---|---|
| Code only (rules, no LLM) | `python -m matheasy.ui` | http://localhost:8000 |
| Gemini | `python -m matheasy.ui --llm` | http://localhost:8000 |
| Gemini + Eedi | `python -m matheasy.ui --llm --eedi` | http://localhost:8000 |
| vLLM (open model on our own GPU) | `python -m matheasy.ui --vllm --vllm-model Qwen/Qwen2.5-3B-Instruct-AWQ` | http://localhost:8001 |
| vLLM + Eedi | add `--eedi` to the line above | http://localhost:8001 |

- **Gemini** needs `GEMINI_API_KEY` set.
- **`--eedi`** needs a GPU and `data/shortlister/adapter-3B-v2`, and loading it takes 2-3 minutes.
- **`--vllm`** needs a vLLM server running first. On Windows, that means WSL: see [WSL.md](WSL.md). The short version, in PowerShell:
  ```
  wsl -d Ubuntu -- bash -lc "source ~/vllm-env/bin/activate && VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen2.5-3B-Instruct-AWQ --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.55 --generation-config vllm"
  ```
  Wait for `Application startup complete`, then start the page.

**To check what's on:** sign in, click your avatar (top right) and look under **System**. It shows LLM (Gemini), vLLM, Eedi and Textbook, each on or off.

**To stop:** press Ctrl+C in each window.

---

## What the page looks like now

**Top bar**
- The **∑ MathEasy** logo.
- **Practice** and **Progress** views. A dot appears on Progress after an Analyze.
- The **☀/☾** theme button. Dark is the default, and the browser remembers your choice.
- The **user menu**: name and role, a **Show technical details** switch, the **System** status and **Sign out**.

**Practice view: the Problem panel (left)**
- A question box. You can pick a problem or type your own.
- **See an example** buttons, which fill in a sample student answer and analyse it.
- A small progress box: answers, slips, misconceptions and the weakest topics.
- The **problem set**:
  - topic filters (All / Fractions / Arithmetic / Algebra) and a "2 / 7 solved" count;
  - a level on each problem (Easy, Medium, Hard), set by hand;
  - a mark on each problem: ✓ solved, ● tried, ○ not tried.
- Clicking a problem only loads it. It no longer answers it for the student.

**Practice view: the Working panel (right)**
- An editor with line numbers. It grows with the working (4-12 lines).
- After an Analyze, the **first wrong line gets a red band**, and correct lines get green numbers.
- **▶ Analyze**, or Ctrl+Enter.

**Result**
- **While it works**, it shows the server's real stages: Reading your lines → Checking each step → Finding the mistake → Writing a hint → Finding what to revise.
- **Then the verdict**: ✓ Correct, ✗ Wrong at line N, or ⚠ Slip, with the time taken. Next comes the mistake, then the hint, then the details.
- **What to do next:**
  - after a correct answer, a **Next problem →** button;
  - after a wrong one, **Fix line N and try again**, which selects that line.

**Progress view**
- The mastery chart, with bars coloured by state (red = weak, blue = learning, green = mastered) and a legend.
- The misconceptions seen and the recent work.

**Students and teachers**
- **Students** don't see rule IDs, sources (Rule / Eedi / LLM), status words, the LLM's reconstructed working or the hints the critic rejected.
- **Teachers** see all of these by default. Anyone can switch them with **Show technical details**.

**On a phone:** **Problem / Working / Result** tabs show one panel at a time, and the Working tab shows the question too.

**Accessibility:** every control is a real button, so Tab and Enter reach them all, and focus is outlined.

---

## Fixes from testing

I put 21 real answers through the tutor (wrong, correct and broken inputs), then compared Gemini with Qwen 3B on the 12 wrong ones.

| Problem | Fix | Result |
|---|---|---|
| "hello" was analysed as a wrong answer when the LLM was on. A small model reads it as `{hello}`, a named quantity. | `not_maths()` in `ui/__main__.py`: a line is not maths if it names a word the question doesn't, or uses letters when the question has none. It runs with the LLM on too. | "hello" now gets "type it as maths". |
| The mistake card was empty when the critic dropped a rule's mistake, even when the rule reproduced the step exactly. | The page still shows it, tagged **likely**, with a short note. The LLM's unconfirmed reading is shown with technical details only. Students see only readings that code checked, as in `tutor/analysis.py`. | Qwen 3B: the mistake is shown for 5 of 12 wrong answers, up from 1. |
| A single Analyze could take 30 s (3 hint attempts, each reviewed by the critic). | `write_hint(..., max_rewrites=)` is a new optional setting (default 2, unchanged for the runner). The page passes 1, so 2 attempts, then the safe template. | Qwen 3B: slowest 28.5 s → 9.4 s, median 6.9 s → 5.4 s, with no more generic hints. On Gemini, 1 of 12 hints passed only on its 3rd attempt and would now get the safe template. |
| Is the critic too strict? Qwen 3B got the generic hint for 5 of 12 wrong answers. | Measured Gemini on the same 12. | **Gemini: 0 of 12 generic hints.** The critic is fine; the small model writes hints it can't pass. No change to the critic. |

---

## Files changed

| File | Owner | Change |
|---|---|---|
| `matheasy/ui/index.html` | B (page) | Rewritten: the layout, problem set, editor marks, live progress, user menu, student and teacher views, phone tabs, chart colours. |
| `matheasy/ui/__main__.py` | B (page) | `/api/progress` (the running stage, sign-in required), `not_maths()`, the `likely` and `reading` fields, and `PAGE_REWRITES = 1`. |
| `matheasy/tutor/hinting.py` | C | `write_hint` takes an optional `max_rewrites` (default `MAX_REWRITES`, unchanged). |
| `tests/test_ui.py` | | Tests for the not-maths check and the progress endpoint. |
| `tests/test_hinting.py` | | A test for the rewrite cap. |

The API the page uses hasn't changed, apart from the new `/api/progress` and two new fields in `/api/analyze` results: `likely` and `reading`.

---

## Still open

- **Naming the misconception for text-only students** is the biggest gap: 0.24 top-1 on Gemini. This is A's shortlister work: retrain on queries shaped like the tutor's, then run the final evaluation on Gemini.
- **Number words** such as "five sixths" aren't read as 5/6.
- **The problem set** has 7 problems with hand-set levels.
- **On a 6 GB GPU**, vLLM + Eedi nearly fills the memory. If an Analyze fails with "out of memory", use Gemini for Eedi demos.
