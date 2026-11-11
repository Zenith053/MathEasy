"""Linking a step no rule explains to an Eedi misconception.

    ranker = load_ranker()                       # the trained shortlister on this machine's GPU, or None
                                                 # (an NVIDIA GPU, or Apple silicon's MPS)
    d = diagnose(before, after, question=q, session=s, eedi_ranker=ranker)
    d.misconception                              # 'eedi:1287' when the LLM picked an Eedi name

When no rule reproduces a step, the ranker (A's shortlister) scores the in-topic Eedi
names against the step's query, the same query format it was trained on, and the top
EEDI_K go to the LLM fallback (explain.py) as options. The LLM always picks the closest
one, since a near concept helps a student more than no answer, and says whether it fits
"exact" or is only "close" (Explanation.eedi_fit). A picked name becomes the diagnosis, so the
tutor shows it, aims the hint at it, searches the textbook with it and records its ID
in the student's memory, exactly as for a rule.

Without a ranker, diagnose() works as before: rules, solve back, then the LLM's own words.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, Sequence

from .ids import eedi_id
from .maths import LocatedError

EEDI_K = 20      # names the LLM chooses from: more names, more chance the right one is there
RANK_DEPTH = 80  # names asked of the ranker, so that EEDI_K in-topic ones are left after filtering
# v2: trained on calculation steps in the tutor's own query format (docs/diagnosis_fallback.md).
# Run 3, the earlier model, is data/shortlister/adapter-3B-llm: pass it to load_ranker() to compare.
ADAPTER = Path(__file__).resolve().parents[2] / "data" / "shortlister" / "adapter-3B-v2"


class EediRanker(Protocol):
    """A's shortlist() signature: [(MisconceptionId, name, score), ...] best first."""
    def __call__(self, query: str, k: int = 25) -> list[tuple[int, str, float]]: ...


def options(located: LocatedError, ranker: EediRanker, k: int = EEDI_K) -> list[tuple[str, str]]:
    """The k best in-topic Eedi names for this step, as [(eedi ID, name)], best first."""
    from matheasy.shortlister.candidates import candidate_ids

    from .interfaces import query

    keep = candidate_ids()
    found = [(eedi_id(int(mid)), name) for mid, name, _score in ranker(query(located), k=RANK_DEPTH) if int(mid) in keep]
    return found[:k]


def load_ranker(adapter: Path | str = ADAPTER, *, name_batch: int = 8) -> Optional[EediRanker]:
    """The trained shortlister, ranking only in-topic names, if this machine has a GPU (NVIDIA,
    or Apple silicon's MPS) and the adapter; else None. Loading takes about a minute and 2 GB
    of GPU memory (about 6 GB on MPS, which has no 4-bit), so load it once per process and
    pass it to every diagnose() call."""
    try:
        import torch

        gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
        if not gpu or not Path(adapter).exists():
            return None
        from matheasy.shortlister.candidates import candidate_ids
        from matheasy.shortlister.model import ModelShortlister
    except ImportError:
        return None
    return ModelShortlister(adapter, candidates=candidate_ids(), name_batch=name_batch)


def named(options_given: Sequence[tuple[str, str]], pick) -> Optional[str]:
    """The eedi ID the LLM picked by its number (1 = the first option); None for 0 or anything else."""
    if isinstance(pick, bool) or not isinstance(pick, int) or not 1 <= pick <= len(options_given):
        return None
    return options_given[pick - 1][0]
