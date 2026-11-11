# Report draft: fine-tuning and retrieval (Role A)

Draft for the final report, written by A, for B and C to fit into the final version. Every number comes from the repo and the private Kaggle runs named in each section. It follows `docs/design_decisions.md`, so H3, which the team plan gave A, is left out (section 3).

## 1. The misconception shortlister

### 1.1 Task

When no rule explains a student's wrong step, the tutor needs a short list of likely misconceptions to reason about. Eedi's competition data names about 2,600 misconceptions. The shortlister ranks all of them for one wrong step and returns the top 25. Live, the LLM fallback chooses from the top 20 in-topic names (`diagnosis/eedi.py`).

### 1.2 Data

- **Source.** Eedi "Mining Misconceptions in Mathematics" (Kaggle competition data, used under its rules; never committed or re-uploaded).
- **Subset.** Questions on fractions, decimals and algebraic expressions: 1,472 labelled wrong answers.
- **Split by question,** so no test question is seen in training:

| Split | Wrong answers |
|---|---|
| Train | 1,038 |
| Validation | 145 |
| Test | 289, of which 91 have a misconception with no training question ("unseen") |

- **The query.** `build_query()` turns a wrong step into `Topic | Before | Student wrote | Correct | Changed part | Difference`. The same function is used in training and in the live tutor. Eedi rows do not always fit it: only 527 of 1,472 (36%) are plain calculations that the tutor's step locator can read. The rest are word problems, diagrams or multiple choice, so for them the query falls back to a text diff.

### 1.3 Method

- **Model.** A bi-encoder: the query and each misconception name are embedded by the same model, and names are ranked by cosine similarity. The model is Qwen2.5-3B-Instruct (1.5B for comparison), last-token pooling, with an instruction prefix on queries.
- **Fine-tuning.** QLoRA: the base model is loaded in 4-bit (NF4), and a LoRA adapter (rank 16, alpha 32, dropout 0.05) is trained on all attention and MLP projections.
- **Loss.** InfoNCE with temperature 0.05. Each query's negatives are the other names in its batch, plus 3 "hard negatives": the names most similar to the right one.
- **Settings (runs 1–4).** 3 epochs, batch 8, learning rate 1e-4 with 10% warm-up, queries up to 512 tokens. The epoch with the best validation MAP@25 was kept.
- **Extra training data.**
  - **512 LLM-written examples** (run 3): wrong steps written by the shared LLM for misconceptions with no training question. Only misconception names were sent to the LLM.
  - **939 rule-generated pairs** (run 4): B's 32 mapped rules applied to generated questions. These give live-format queries with a known Eedi label.
- **Compute.** Each run took about 1½–2 hours on one Kaggle T4 GPU.

### 1.4 Results on held-out questions

Scored on the 289 test rows (`shortlister/score_test.py`). The 95% intervals come from resampling rows 2,000 times.

| Run | Trained on | MAP@25 | Recall@10 | MAP@25, seen | MAP@25, unseen |
|---|---|---|---|---|---|
| BM25 | no training (word matching) | 0.038 [0.021, 0.059] | 0.059 | 0.024 | 0.070 |
| 1 | 3B, Eedi rows | 0.292 [0.254, 0.330] | 0.612 | 0.338 | 0.190 |
| 2 | 1.5B, Eedi rows | 0.275 [0.239, 0.312] | 0.578 | 0.318 | 0.180 |
| 3 | 3B, plus LLM-written examples | 0.323 [0.283, 0.363] | 0.644 | 0.335 | **0.295** |
| 4 | 3B, plus LLM examples and rule pairs | 0.329 [0.288, 0.372] | 0.630 | **0.381** | 0.216 |

What each change did, measured on the same rows (paired, MAP@25):

| Change | All rows | Seen | Unseen |
|---|---|---|---|
| Fine-tuning (run 1 against BM25) | **+0.253 [0.215, 0.293]** | +0.315 | +0.120 |
| 3B against 1.5B | +0.017 [−0.015, 0.049] | +0.020 | +0.010 |
| LLM-written examples (run 3 against run 1) | +0.031 [−0.001, 0.061] | −0.003 | **+0.105 [0.055, 0.157]** |
| Rule pairs (run 4 against run 3) | +0.006 [−0.031, 0.043] | +0.046 [0.001, 0.091] | **−0.080 [−0.141, −0.017]** |

- **Fine-tuning is what matters.** A fine-tuned 3B model puts the right misconception in the top 10 for 61% of wrong answers. Word matching manages 6%.
- **Size did not matter much.** 1.5B is not clearly worse than 3B.
- **LLM-written examples help where they were aimed.** They lift misconceptions that have no training question by about 0.1, and leave the others unchanged.
- **Rule pairs trade one group for the other.** They help misconceptions seen in training and hurt unseen ones, so the total does not move.
- These intervals are not corrected for multiple comparisons, so results near 0 are weak evidence.

### 1.5 Training queries and live queries differ

The tutor writes queries from its own step locator (plain notation such as `2/5`, the exact wrong term). Eedi's are LaTeX with a text diff. The same 156 validation and test rows were scored in both formats (`shortlister/score_live.py`):

| Model | Eedi format: MAP@25 / Recall@10 | Tutor format: MAP@25 / Recall@10 |
|---|---|---|
| Run 3 | 0.360 / 0.718 | 0.160 / 0.436 |
| Run 4 | 0.371 / 0.679 | 0.186 / 0.397 |

The tutor's format costs about 0.2 MAP@25 (run 3: 95% interval 0.15 to 0.25). So scores in Eedi's format overstate how well the model works inside the tutor.

### 1.6 v2: retrained for the tutor's queries (by B, 9 November)

B retrained the same model to close this gap (`docs/diagnosis_fallback.md`). The changes were:
- training on the 626 calculation rows only, with a tutor-format copy of 294 of them;
- choosing the epoch by Recall@25 on both formats;
- 6 epochs, batch 16.

On 137 validation and test steps in the tutor's format, ranking all names:

| | Run 3 | v2 |
|---|---|---|
| Right misconception first | 0.07 | 0.20 |
| In the top 10 | 0.47 | 0.61 |
| In the top 25 | 0.62 | 0.78 |

- Eedi-format recall did not drop: top 25 went from 0.87 to 0.90.
- With 137 rows, these figures carry roughly ±7 points of uncertainty.
- v2 is the model the tutor uses.

### 1.7 How the shortlist is used live

The 20 best in-topic names go to the LLM fallback, which picks one and says whether it fits exactly or is only the closest. B's pick test (`eval/eedi_pick.py`, 22 cases) gave these results:
- The right name was among the 20 options 16 times.
- The LLM's pick matched it exactly 5 times, and 16 picks were judged acceptable.
- Exact picks were acceptable 13 of 14 times, close picks 3 of 8.
- Many misses are near-duplicate names (to-do T7).

### 1.8 Cost

- **Shortlister.** One forward pass per query on a 4 GB GPU. Embedding the 972 in-topic candidate names takes about 8 minutes once, then they are cached, and a restart takes about 30 seconds.
- **LLM.** The shortlister makes no LLM call itself; the pick is part of the fallback's one call.
- **Pending.** A shortlist cache built from v2 is still needed, so the final runs can skip the Eedi step's GPU.

## 2. Textbook retrieval (RAG)

### 2.1 Task

Every hint cites a textbook section. `retrieve()` returns the best passages for the diagnosed misconception, each with its reference, for example "Prealgebra 2e, 4.2 Multiply and Divide Fractions > Simplify Fractions".

### 2.2 Method

- **Corpus.** OpenStax *Prealgebra 2e* and *Elementary Algebra 2e* (CC BY), split by section and subsection into 1,272 passages.
- **Hybrid search.**
  - BM25 matches exact words.
  - Static embeddings (model2vec `potion-base-8M`, CPU only) match meaning, for example "multiplies only the first term in the bracket" finds the Distributive Property.
  - The two rankings are merged by reciprocal-rank fusion (k = 60).
- **UK to US terms.** The tutor and Eedi use UK words and OpenStax is American, so US terms are added to the query. For example, "brackets" adds "parentheses distributive property".
- **Weak-match flag.** Each result carries its cosine similarity. `is_weak()` flags a result list whose best similarity is below 0.45. Right-section results scored at least 0.56, and 10 off-topic queries at most 0.39. The flag catches off-topic queries only: a wrong section on the right topic can still score high.

### 2.3 Results

The "right section retrieved" metric: 30 misconception queries, each with a hand-labelled right section (`retrieval/eval_sections.json`).

| Search | Right section first | In the top 3 |
|---|---|---|
| BM25 only | 53% | 87% |
| Embeddings only | 70% | 87% |
| **Hybrid** | **70%** | **97%** |

- The one hybrid miss is "keeps the common denominator when multiplying fractions".
- **In the tutor:** in the LLM trial (7–8 November), 34 of the full tutor's 48 hints cited a section. The other 14 were the critic's safe template, which carries no citation.

## 3. H3 is not reported

The team plan's H3 compared the fine-tuned shortlister with a much larger prompted model. It was dropped on 30 October with the other hypotheses (`docs/design_decisions.md`, "What we are trying to build"), and no large-model comparison was run. Sections 1.4–1.8 report the fine-tuning on its own terms.

## 4. Limitations

- **Small test set.** 289 test rows, and 91 unseen, so differences under about 0.04 MAP@25 are not resolved.
- **Tutor-format scores cover only plain calculations,** the 36% of Eedi that fits the tutor's format.
- **Single labels.** Eedi gives each wrong answer one misconception, so a reasonable near-duplicate counts as wrong.
- **Small retrieval eval.** 30 queries, labelled by A.
