# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl as a second reader, and what it found that the main system did not.

Sibyl re-forecasts the ten most volatile questions by reading the open web
with an agent. The main pipeline reads structured connectors. They are looking
at different things, which is the whole point of running both, and until this
module Sibyl appeared in the report as a single line on page fourteen.

Three things a reader wants:

* **Where the second reader disagrees**, and in which direction. Compared on
  expected value, because that is the quantity the rest of the report is
  ordered by; the divergence score says how far apart the distributions are
  but not which way.
* **What the second reader found that the main system did not.** Sibyl's trial
  belief traces are prose about specific events. Any sentence in them whose
  content does not appear in the main pipeline's own grounding is, by
  construction, something the structured connectors did not carry. Nobody else
  in the sector publishes this.
* **Whether Sibyl's own trials settled the question.** Sibyl runs K
  independent trials; when they disagree with each other, careful research did
  not reach an answer. That is a different signal from disagreeing with Fred,
  and it is worth printing as its own flag.

Pure functions over plain values. The extraction is substring and token
matching, deliberately: it is a prompt for a reader's attention, not a claim
that a fact is new to the world, and a cleverer method would invite the
report to make the stronger claim.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

TAG_AGREES = "second opinion agrees"
TAG_MORE_ALARMED = "second opinion is more alarmed"
TAG_MORE_CAUTIOUS = "second opinion is more cautious"
TAG_NOT_COVERED = "no second opinion"

# The model may not drop this, and it is not the model's to write. Sibyl has
# never been scored: it is a second reader, and a second reader who has never
# been marked cannot break a tie.
SIBYL_CAVEAT = (
    "Sibyl reads the open web; the main system reads structured data feeds. "
    "Sibyl has no scored track record yet, so where the two disagree the "
    "disagreement is a reason to look again, never a reason to prefer one. "
    "It is a second reader, not a tiebreaker."
)


def direction_tag(
    fred_ev: float | None,
    sibyl_ev: float | None,
    *,
    ratio: float,
) -> str:
    """Which way the second reader leans, on expected value.

    A ratio rather than a difference, because the metrics span deaths and
    millions of people in need and one absolute gap cannot serve both. When
    either side is missing there is no comparison to report, so the pair reads
    as uncovered rather than as agreement.
    """
    if fred_ev is None or sibyl_ev is None:
        return TAG_NOT_COVERED
    try:
        f, s = float(fred_ev), float(sibyl_ev)
    except (TypeError, ValueError):
        return TAG_NOT_COVERED
    if f <= 0 and s <= 0:
        return TAG_AGREES
    if f <= 0:
        return TAG_MORE_ALARMED
    if s <= 0:
        return TAG_MORE_CAUTIOUS
    if s >= f * ratio:
        return TAG_MORE_ALARMED
    if f >= s * ratio:
        return TAG_MORE_CAUTIOUS
    return TAG_AGREES


def unsettled(inter_trial_js: float | None, *, share: float) -> bool:
    """Did Sibyl's own trials fail to converge?

    ``js_divergence_inter_trial`` lives on [0, ln 2] like every other JSD in
    the repo, so the threshold is a share of that maximum.
    """
    if inter_trial_js is None:
        return False
    import math

    try:
        return float(inter_trial_js) / math.log(2.0) >= float(share)
    except (TypeError, ValueError, ZeroDivisionError):
        return False


# ---------------------------------------------------------------------------
# Novel evidence
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_PROPER = re.compile(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]{2,}")
_DIGIT = re.compile(r"\d")

# Words that carry no evidence on their own. Kept short on purpose: a long
# stop list starts deciding what counts as a fact.
_STOPWORDS = frozenset("""
the and for with that this from have has had been are was were will would
could should may might can into over under about after before during between
their there these those which while than then them they its it's not but our
your his her out per via such also more most less least very much many some
any all one two both each other another same each own how why when where what
who whom whose because since until upon among across against toward towards
within without through around above below near far new old high low large
small said says say told reported reports report reporting according
government national international agency agencies country countries region
regional area areas people population humanitarian response responses
situation situations continue continued continues remain remains remained
expect expected expects likely unlikely risk risks level levels data
information sources source evidence analysis assessment assessments
""".split())


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD.finditer(text or "")}


def _content_tokens(text: str) -> list[str]:
    return [
        t for t in (m.group(0).lower() for m in _WORD.finditer(text or ""))
        if len(t) >= 4 and t not in _STOPWORDS
    ]


# Sibyl's traces mix two kinds of prose: what it LEARNED about the world, and
# how it went about forecasting. The first live run printed "Nudged q0.99 up
# modestly to 150,000 to respect the fat conditional tail", "further research
# would not materially move the distribution, so submitting" and "Fetching the
# dedicated war article should give a month-by-month casualty timeline" under a
# heading promising things the main system did not know. None of those is a
# finding; two are a to-do list.
#
# So a sentence is dropped when it talks about the forecast rather than about
# the situation. Matched on vocabulary that only appears in forecasting talk,
# and on the first person, which a factual finding never needs.
_PROCESS_MARKERS = re.compile(
    r"""\b(
        q0\.\d+ | quantile | quantiles | posterior | prior[s]? | draws? |
        pooled | median | percentile | distribution | calibrat\w* |
        submitting | submit | nudged? | seeded | anchor\w* | my\ (?:central|
        estimate|forecast|read) | i\ (?:therefore|stay|keep|set|put|assume|
        will|would|should|expect\ to)
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)
# A research to-do rather than a statement: no finite verb, or an opening that
# only makes sense as a question to go and answer.
_TODO_OPENERS = re.compile(
    r"^(any\b|whether\b|check\b|confirm\b|fetch\w*\b|search\b|verify\b|"
    r"look\b|find\b|need\ to\b|todo\b|month-by-month\b)",
    re.IGNORECASE,
)
_BARE_URL = re.compile(r"^\s*https?://\S+\s*$", re.IGNORECASE)


def _is_process_talk(sentence: str) -> bool:
    """Is this about the forecast rather than about the world?"""
    text = sentence.strip()
    if _BARE_URL.match(text):
        return True  # a link is a citation, not a finding
    if _TODO_OPENERS.match(text):
        return True
    return bool(_PROCESS_MARKERS.search(text))


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    for chunk in re.split(r"[\n\r]+", text or ""):
        for sentence in _SENTENCE_SPLIT.split(chunk):
            cleaned = " ".join(sentence.split())
            if 40 <= len(cleaned) <= 320:
                out.append(cleaned)
    return out


def novel_facts(
    trial_texts: Iterable[str],
    known_text: str,
    *,
    limit: int = 2,
    min_novelty: float = 0.4,
) -> list[str]:
    """Sentences from Sibyl's trials whose content is absent from the pack.

    Scored on the share of a sentence's content words that do not appear
    anywhere in the main pipeline's own evidence. A sentence has to be
    SPECIFIC to qualify at all: it names something or counts something. A
    general observation that drought is bad is novel to no one.
    """
    known_tokens = _tokens(known_text)
    known_lower = (known_text or "").lower()
    scored: list[tuple[float, int, str]] = []
    seen: set[str] = set()

    for text in trial_texts:
        for sentence in _sentences(str(text or "")):
            fingerprint = re.sub(r"\W+", "", sentence.lower())[:120]
            if not fingerprint or fingerprint in seen:
                continue
            content = _content_tokens(sentence)
            if len(content) < 4:
                continue
            specific = bool(_PROPER.search(sentence)) or bool(_DIGIT.search(sentence))
            if not specific:
                continue
            if _is_process_talk(sentence):
                continue
            if sentence.lower() in known_lower:
                continue
            unknown = [t for t in content if t not in known_tokens]
            novelty = len(unknown) / len(content)
            if novelty < min_novelty:
                continue
            seen.add(fingerprint)
            # Longer sentences carry more, so length breaks ties, but novelty
            # decides: the point is what the other reader saw.
            scored.append((novelty, len(content), sentence))

    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [s for _, _, s in scored[:max(0, int(limit))]]


def trial_texts(trials: Any) -> list[str]:
    """Every piece of prose in a ``trials_json`` structure, flattened.

    The shape has changed once already (belief traces moved inside the trial
    objects), so this walks whatever it is given rather than reaching for
    named keys, and takes any string long enough to be a sentence.
    """
    out: list[str] = []

    def _walk(node: Any, depth: int = 0) -> None:
        if depth > 6:
            return
        if isinstance(node, str):
            if len(node) >= 40:
                out.append(node)
        elif isinstance(node, dict):
            for value in node.values():
                _walk(value, depth + 1)
        elif isinstance(node, (list, tuple)):
            for value in node:
                _walk(value, depth + 1)

    _walk(trials)
    return out


def summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Counts for the section's opening line, from the rows themselves."""
    return {
        "n_covered": len(rows),
        "n_more_alarmed": sum(1 for r in rows if r.get("tag") == TAG_MORE_ALARMED),
        "n_more_cautious": sum(1 for r in rows if r.get("tag") == TAG_MORE_CAUTIOUS),
        "n_agrees": sum(1 for r in rows if r.get("tag") == TAG_AGREES),
        "n_unsettled": sum(1 for r in rows if r.get("unsettled")),
        "n_with_novel_evidence": sum(1 for r in rows if r.get("novel_evidence")),
    }
