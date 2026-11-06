"""
How much does a trained adapter lose on live-format queries? (Role A, needs a GPU.)

Scores the same val and test rows twice, in the training format and in the live
format (live_queries.py), so the comparison is like for like. The full val and
test scores are printed first as a check that the adapter loaded correctly: they
should match the run's results.json. Only scores are written, no Eedi text.

Usage: python -m matheasy.shortlister.score_live --adapter DIR --out results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from .shortlist import map_at_k, recall_at_k

DATA = Path(__file__).resolve().parents[2] / "data" / "eedi"
INSTRUCT = "Instruct: Given a student's wrong step, retrieve the misconception that explains it\nQuery: "
MAX_Q, MAX_M = 512, 96
KEY = ["QuestionId", "WrongAnswer"]
N_BOOT = 2000
SEED = 20261016


def load(adapter):
    """The base model and adapter, set up as in train_shortlister.ipynb when it evaluated."""
    base = json.loads((Path(adapter) / "results.json").read_text())["model"]
    tok = AutoTokenizer.from_pretrained(adapter)
    tok.padding_side = "right"
    model = AutoModel.from_pretrained(
        base,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16,
                                               bnb_4bit_use_double_quant=True),
        dtype=torch.float16, device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    return tok, PeftModel.from_pretrained(model, adapter).eval()


@torch.no_grad()
def embedder(tok, model):
    def embed(texts, max_len, size):
        out = []
        for i in range(0, len(texts), size):
            chunk = [t + tok.eos_token for t in texts[i:i + size]]
            batch = tok(chunk, max_length=max_len, truncation=True, padding=True, return_tensors="pt").to(model.device)
            hidden = model(**batch).last_hidden_state
            last = batch["attention_mask"].sum(1) - 1
            out.append(F.normalize(hidden[torch.arange(len(chunk)), last].float(), dim=-1))
        return torch.cat(out)
    return embed


def scores(ranked, true):
    return {"n": len(true), "MAP@25": map_at_k(ranked, true, 25), "Recall@10": recall_at_k(ranked, true, 10)}


def paired_gap(train_ranked, live_ranked, true):
    """Mean of (live - training) reciprocal rank, with a 95% bootstrap interval over rows."""
    rr = lambda r, t: 1 / (r.index(t) + 1) if t in r[:25] else 0.0
    gap = np.array([rr(b, t) - rr(a, t) for a, b, t in zip(train_ranked, live_ranked, true)])
    rng = np.random.default_rng(SEED)
    means = gap[rng.integers(0, len(gap), (N_BOOT, len(gap)))].mean(axis=1)
    return {"gap": float(gap.mean()), "ci95": [float(x) for x in np.percentile(means, [2.5, 97.5])]}


def main(adapter, out):
    queries = pd.read_csv(DATA / "queries.csv")
    queries = queries[queries.Split.isin(["val", "test"])]
    live = pd.read_csv(DATA / "live_queries.csv")
    live = live[live.Split.isin(["val", "test"])]
    names = pd.read_csv(DATA / "raw" / "misconception_mapping.csv")
    names["MisconceptionName"] = names.MisconceptionName.str.strip()

    tok, model = load(adapter)
    embed = embedder(tok, model)
    ids = names.MisconceptionId.tolist()
    m = embed(names.MisconceptionName.tolist(), MAX_M, 64)

    def rank(texts):
        q = embed([INSTRUCT + t for t in texts], MAX_Q, 16)
        return [[ids[j] for j in row] for row in (q @ m.T).topk(25, dim=1).indices.cpu().tolist()]

    queries["Ranked"] = rank(queries.Query.tolist())
    live["Ranked"] = rank(live.Query.tolist())
    both = live.merge(queries[KEY + ["Query", "Ranked"]], on=KEY, suffixes=("_live", "_train"))
    assert len(both) == len(live), "every live row must have a training-format row"

    def compare(rows):
        true = rows.MisconceptionId.tolist()
        return {"training format": scores(rows.Ranked_train.tolist(), true),
                "live format": scores(rows.Ranked_live.tolist(), true),
                "live - training": paired_gap(rows.Ranked_train.tolist(), rows.Ranked_live.tolist(), true)}

    results = {"full": {s: scores(q.Ranked.tolist(), q.MisconceptionId.tolist()) for s, q in queries.groupby("Split")},
               "live-format rows": {"val + test": compare(both),
                                    **{s: compare(rows) for s, rows in both.groupby("Split")},
                                    **{t: compare(rows) for t, rows in both.groupby("Topic")}}}
    print(json.dumps(results, indent=1))
    Path(out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", default="live_format_results.json")
    a = ap.parse_args()
    main(a.adapter, a.out)
