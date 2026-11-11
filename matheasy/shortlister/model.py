"""
The trained shortlister: ranks every Eedi misconception name for a query (Role A).

A bi-encoder from train_shortlister.ipynb: the query and the names are embedded by
the same model and ranked by cosine similarity. It has the same signature as the
BM25 shortlist(), so it plugs straight into B's EediShortlister:

  ranker = ModelShortlister("data/shortlister/adapter-3B-v2")
  diagnose(before, after, shortlister=EediShortlister(ranker))

Uses the same 4-bit setup as training on an NVIDIA GPU, half precision on an Apple
silicon GPU (MPS: about 6 GB, where full precision would not fit in 16 GB), and full
precision on a CPU (slower, but one query is one forward pass). Needs torch,
transformers and peft (and bitsandbytes for the 4-bit setup), and
data/eedi/raw/misconception_mapping.csv.

The names' vectors are computed once and saved in the adapter's folder
(name_vectors_<key>.pt, git-ignored with the rest of data/). The key covers the adapter,
the names, the token limit and GPU or CPU, so a changed model or name list never reuses
old vectors. Loading then only needs the model itself, not a pass over every name.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

DATA = Path(__file__).resolve().parents[2] / "data" / "eedi"
INSTRUCT = "Instruct: Given a student's wrong step, retrieve the misconception that explains it\nQuery: "
MAX_Q, MAX_M = 512, 96  # token limits, as in training
QUERY_BATCH, NAME_BATCH = 16, 64


def device() -> str:
    """Where the model runs: an NVIDIA GPU, an Apple silicon GPU, or the CPU."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load(adapter):
    """The base model named in the adapter's results.json, with the adapter on top."""
    base = json.loads((Path(adapter) / "results.json").read_text(encoding="utf-8"))["model"]
    tok = AutoTokenizer.from_pretrained(adapter)
    tok.padding_side = "right"
    if device() == "mps":  # bitsandbytes' 4-bit needs CUDA; half precision fits instead
        model = AutoModel.from_pretrained(base, dtype=torch.float16)
        return tok, PeftModel.from_pretrained(model, adapter).to("mps").eval()
    if torch.cuda.is_available():
        model = AutoModel.from_pretrained(
            base,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.float16,
                                                   bnb_4bit_use_double_quant=True),
            dtype=torch.float16, device_map={"": 0})
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    else:
        model = AutoModel.from_pretrained(base, dtype=torch.float32)
    return tok, PeftModel.from_pretrained(model, adapter).eval()


class ModelShortlister:
    def __init__(self, adapter, candidates=None, name_batch=None):
        """`candidates`: rank only these MisconceptionIds (e.g. candidates.candidate_ids()).
        `name_batch`: smaller for a small GPU (8 fits in 4 GB)."""
        self.tok, self.model = _load(adapter)
        names = pd.read_csv(DATA / "raw" / "misconception_mapping.csv")
        if candidates is not None:
            names = names[names.MisconceptionId.isin(candidates)]
        self.ids = names.MisconceptionId.tolist()
        self.names = names.MisconceptionName.str.strip().tolist()
        self.vectors = self._name_vectors(Path(adapter), name_batch or NAME_BATCH)

    def _name_vectors(self, adapter: Path, batch: int):
        """The names' vectors, from the saved file when it matches, else computed and saved."""
        fingerprint = json.dumps([(adapter / "adapter_config.json").read_text(encoding="utf-8"),
                                  (adapter / "results.json").read_text(encoding="utf-8"),
                                  MAX_M, device(), self.ids, self.names])
        path = adapter / f"name_vectors_{hashlib.sha256(fingerprint.encode()).hexdigest()[:16]}.pt"
        if path.exists():
            saved = torch.load(path, map_location=self.model.device)
            if saved["ids"] == self.ids:
                return saved["vectors"]
        vectors = self._embed(self.names, MAX_M, batch)
        try:
            torch.save({"ids": self.ids, "vectors": vectors.cpu()}, path)
        except OSError:
            pass  # a read-only folder: just compute them again next time
        return vectors

    @torch.no_grad()
    def _embed(self, texts, max_len, size):
        """Last-token pooling, L2-normalised."""
        out = []
        for i in range(0, len(texts), size):
            chunk = [t + self.tok.eos_token for t in texts[i:i + size]]
            batch = self.tok(chunk, max_length=max_len, truncation=True, padding=True,
                             return_tensors="pt").to(self.model.device)
            hidden = self.model(**batch).last_hidden_state
            last = batch["attention_mask"].sum(1) - 1
            out.append(F.normalize(hidden[torch.arange(len(chunk)), last].float(), dim=-1))
        return torch.cat(out)

    def rank(self, queries, k=25):
        """For each query, its top k as [(MisconceptionId, name, similarity), ...], best first."""
        q = self._embed([INSTRUCT + t for t in queries], MAX_Q, QUERY_BATCH)
        top = (q @ self.vectors.T).topk(k, dim=1)
        return [[(self.ids[j], self.names[j], s) for j, s in zip(row_ids, row_scores)]
                for row_ids, row_scores in zip(top.indices.cpu().tolist(), top.values.cpu().tolist())]

    def __call__(self, query, k=25):
        return self.rank([query], k)[0]
