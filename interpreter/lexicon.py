# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The probability lexicon: fixed word ↔ band mapping, IPCC style.

The writer uses the words; the validator (Phase 4) checks each word sits in
the band of the probability figure it is attached to; the report's appendix
prints this table so the reader can check the writer.
"""

from __future__ import annotations

from typing import NamedTuple


class Band(NamedTuple):
    word: str
    lo: float  # inclusive
    hi: float  # exclusive (except the top band, inclusive at 1.0)


# Ordered highest first. Boundaries per the plan §7.4: a probability p maps
# to the FIRST band with lo <= p (< hi), so the boundary values (0.99, 0.90,
# 0.66, 0.33, 0.10, 0.01) belong to the band above them.
LEXICON: tuple[Band, ...] = (
    Band("virtually certain", 0.99, 1.0000001),
    Band("very likely", 0.90, 0.99),
    Band("likely", 0.66, 0.90),
    Band("about as likely as not", 0.33, 0.66),
    Band("unlikely", 0.10, 0.33),
    Band("very unlikely", 0.01, 0.10),
    Band("exceptionally unlikely", 0.0, 0.01),
)

WORDS: tuple[str, ...] = tuple(b.word for b in LEXICON)


def word_for(p: float) -> str:
    """The lexicon word for a probability in [0, 1]."""
    p = max(0.0, min(float(p), 1.0))
    for band in LEXICON:
        if band.lo <= p < band.hi:
            return band.word
    return LEXICON[-1].word  # pragma: no cover - unreachable by construction


def band_for(word: str) -> Band | None:
    w = (word or "").strip().lower()
    for band in LEXICON:
        if band.word == w:
            return band
    return None


def word_matches(word: str, p: float) -> bool:
    """Does this lexicon word sit in the band of this probability?"""
    band = band_for(word)
    if band is None:
        return False
    p = max(0.0, min(float(p), 1.0))
    return band.lo <= p < band.hi


def markdown_table() -> str:
    """The appendix table (also embedded in the system prompt)."""
    lines = ["| word | probability band |", "|---|---|"]
    for band in LEXICON:
        if band.word == "virtually certain":
            rng = "≥ 99%"
        elif band.word == "exceptionally unlikely":
            rng = "< 1%"
        else:
            rng = f"{band.lo * 100:.0f}–{band.hi * 100:.0f}%"
        lines.append(f"| {band.word} | {rng} |")
    return "\n".join(lines)
