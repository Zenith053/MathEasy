"""Pieces of maths in free text (Role C), for reading student replies and checking hints."""

from __future__ import annotations

import re

# A run of maths in free text: numbers, single-letter variables (not words, not the s in
# "it's") and operators, e.g. the "x = 4" in "I got x = 4." or "1/3 > 1/4" in "I think 1/3 > 1/4".
_LETTER = r"(?<![A-Za-z'’])[A-Za-z](?![A-Za-z'’])"
_ATOM = rf"(?:\d+(?:\.\d+)?|{_LETTER}|[-+*/×÷^=<>≤≥()%−·])"
_MATHS_RUN = re.compile(rf"{_ATOM}(?:\s*{_ATOM})*")
# "I" and "a" next to a number are words, not variables: "a 3/4", "5/6 I think".
EDGE_WORDS = re.compile(r"^[AaI]\s+(?=[\d(-])|\s+[AaI]$")


def maths_runs(text: str) -> list[str]:
    """Each piece of maths in `text` that has a number in it, in order."""
    runs = (EDGE_WORDS.sub("", run.strip()) for run in _MATHS_RUN.findall(text))
    return [run for run in runs if re.search(r"\d", run)]
