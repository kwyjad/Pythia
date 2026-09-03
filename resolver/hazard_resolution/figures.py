# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Turn extracted figures into ladder candidates — deterministically.

Everything the model was forbidden to do happens here, by rule:

* **Households become people** (:func:`convert_units`), and only when the
  document said households. The multiplier comes from
  ``reliefweb.household_conversion`` and the conversion is recorded on the
  candidate, so a reader can undo it.
* **Duplicates collapse** (:func:`deduplicate`). The same sentence
  republished across five situation reports is one statement, and the same
  figure for the same area restated in different words is one figure.
* **Precedence decides** (:func:`preference_key`). The rulebook orders
  attributions — government > UN agency > IFRC/NGO > media — and within a
  tier the latest-dated figure wins. This is the rung's own internal
  precedence: unlike the record-based rungs, "the largest number" is not
  the right answer when several bodies are quoting different assessments
  of the same flood.
* **Implausible figures are rejected** (:func:`apply_ceiling`), not
  flagged. This is a deliberate difference from :mod:`reconcile`, which
  flags a ceiling breach and keeps the value: there, a curated source
  disagreeing with modelled exposure is a finding worth a human's time;
  here, a figure above the exposed population is much more likely a
  mis-transcription — a national cumulative total, or a figure for a
  different emergency in the same document — and admitting it would let a
  transcription error outrank a curated EM-DAT record.

No LLM, no network, no clock beyond what the caller passes: the same
figures always produce the same candidates.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from resolver.hazard_resolution import cell_ledger
from resolver.hazard_resolution.candidates import VALUE_AFFECTED, Candidate
from resolver.hazard_resolution.extract import (
    SOURCE,
    UNIT_HOUSEHOLDS,
    UNIT_PEOPLE,
    ExtractedFigure,
)
from resolver.hazard_resolution.rulebook import Rulebook
from resolver.hazard_resolution.rules import within_sanity_ceiling

LOG = logging.getLogger(__name__)

#: Rank given to an attribution that matches no configured tier. It sorts
#: after every listed tier but is NOT discarded: an unattributed figure in
#: an OCHA situation report is still the best evidence in the room when
#: nothing else is available.
UNRANKED = 10_000


def authority_tier(stated_by: str, rulebook: Rulebook) -> tuple[int, str]:
    """Classify an attribution into a precedence tier. Returns (rank, tier).

    Lowercase substring matching against the rulebook's keyword lists,
    first tier that matches wins. Deliberately simple: the alternative is a
    judgement call, and judgement calls are what this module exists to keep
    out of the resolution path.
    """

    text = str(stated_by or "").strip().lower()
    if not text:
        return UNRANKED, "unattributed"
    for rank, entry in enumerate(rulebook.get("reliefweb.authority_precedence")):
        keywords = [str(k).strip().lower() for k in entry.get("keywords") or []]
        if any(keyword and keyword in text for keyword in keywords):
            return rank, str(entry.get("tier"))
    return UNRANKED, "unrecognised"


def household_multiplier(iso3: str, rulebook: Rulebook) -> tuple[float, str]:
    """The households-to-people multiplier for a country. Returns (value, origin)."""

    by_iso3 = rulebook.get("reliefweb.household_conversion.by_iso3") or {}
    specific = by_iso3.get(iso3.upper())
    if specific is not None:
        return float(specific), "by_iso3"
    return float(rulebook.get("reliefweb.household_conversion.default_multiplier")), "default"


def convert_units(
    figure: ExtractedFigure, iso3: str, rulebook: Rulebook
) -> tuple[float, dict[str, Any] | None]:
    """Convert a households figure to people. Returns (value, conversion).

    ``conversion`` is None when nothing was converted, so a people figure
    carries no conversion record and cannot be mistaken for one.
    """

    if figure.unit != UNIT_HOUSEHOLDS:
        return float(figure.value), None
    multiplier, origin = household_multiplier(iso3, rulebook)
    converted = float(figure.value) * multiplier
    return converted, {
        "from_unit": UNIT_HOUSEHOLDS,
        "to_unit": UNIT_PEOPLE,
        "stated_value": float(figure.value),
        "multiplier": multiplier,
        "multiplier_origin": origin,
        "converted_value": converted,
    }


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _newest_first(value: str) -> int:
    """A date as a term that sorts NEWEST first in an ascending key.

    Negated day ordinals, so a later date gives a smaller number. An
    absent or unparseable date yields 0, which sorts after every real date
    — correct, because an undated figure should never outrank a dated one.
    """

    from resolver.hazard_resolution.sources import parse_date

    parsed = parse_date(value)
    return -parsed.toordinal() if parsed is not None else 0


def preference_key(
    figure: ExtractedFigure, rulebook: Rulebook
) -> tuple[int, int, int, int, str]:
    """Sort key implementing the rung's precedence. Lower sorts better.

    Authority first, then the latest-dated figure, then the newest
    document, then the document's own source rank, then its id so ties
    never wobble between runs.
    """

    rank, _ = authority_tier(figure.stated_by, rulebook)
    return (
        rank,
        _newest_first(figure.date),
        _newest_first(figure.doc_date),
        int(figure.doc_source_rank),
        str(figure.doc_id),
    )


def deduplicate(
    figures: list[tuple[ExtractedFigure, float]], rulebook: Rulebook
) -> tuple[list[tuple[ExtractedFigure, float]], list[dict[str, Any]]]:
    """Collapse restatements of one figure. Returns (kept, dropped).

    Two passes, because a figure is republished in two different ways:
    verbatim (the same sentence copied into the next situation report) and
    paraphrased (the same number for the same area, worded differently).
    The best-ranked member of each group survives.
    """

    dropped: list[dict[str, Any]] = []

    def _collapse(
        items: list[tuple[ExtractedFigure, float]],
        key,
        reason: str,
    ) -> list[tuple[ExtractedFigure, float]]:
        groups: dict[Any, list[tuple[ExtractedFigure, float]]] = {}
        for item in items:
            groups.setdefault(key(item), []).append(item)
        kept: list[tuple[ExtractedFigure, float]] = []
        for group in groups.values():
            group.sort(key=lambda pair: preference_key(pair[0], rulebook))
            kept.append(group[0])
            for figure, _value in group[1:]:
                dropped.append(
                    {
                        "reason": reason,
                        "doc_id": figure.doc_id,
                        "value": figure.value,
                        "unit": figure.unit,
                        "quote": figure.quote,
                        "superseded_by_doc_id": group[0][0].doc_id,
                    }
                )
        return kept

    kept = _collapse(
        figures, lambda pair: _normalise(pair[0].quote), "duplicate_quote"
    )
    kept = _collapse(
        kept,
        lambda pair: (round(pair[1], 3), _normalise(pair[0].area)),
        "duplicate_value_and_area",
    )
    kept.sort(key=lambda pair: preference_key(pair[0], rulebook))
    return kept, dropped


def apply_ceiling(
    figures: list[tuple[ExtractedFigure, float]],
    exposure_ceiling: float | None,
    rulebook: Rulebook,
    ceiling_basis: dict[str, Any] | None = None,
) -> tuple[list[tuple[ExtractedFigure, float]], list[dict[str, Any]]]:
    """Drop figures above the GDACS exposure ceiling. Returns (kept, rejected).

    ``ceiling_basis`` (from :func:`candidates.exposure_ceiling_basis`) names
    the event and the response field the ceiling came from. It is optional
    only so existing callers keep working: without it a rejection records
    the number and not its origin, and a ceiling of 2 against a reported
    40,000 cannot be told apart from a genuine mis-transcription.
    """

    basis = ceiling_basis or {}
    kept: list[tuple[ExtractedFigure, float]] = []
    rejected: list[dict[str, Any]] = []
    for figure, value in figures:
        if within_sanity_ceiling(value, exposure_ceiling, rulebook):
            kept.append((figure, value))
            continue
        rejected.append(
            {
                "reason": "exceeds_gdacs_exposure_ceiling",
                "doc_id": figure.doc_id,
                "value": value,
                "unit": figure.unit,
                "quote": figure.quote,
                "exposed_population": exposure_ceiling,
                "ceiling_multiplier": rulebook.get("sanity.ceiling_multiplier"),
                "ceiling_source": basis.get("source"),
                "ceiling_source_ref": basis.get("source_ref"),
                "ceiling_field": basis.get("field"),
                "ceiling_events_seen": basis.get("n_events"),
                "ceiling_events_with_exposure": basis.get("n_events_with_exposure"),
            }
        )
    if rejected:
        LOG.warning(
            "[figures] rejected %d extracted figure(s) above the GDACS exposure "
            "ceiling (%.0f x %s) — an extracted figure above modelled exposure is "
            "far more likely a mis-transcription than a discovery",
            len(rejected),
            float(rulebook.get("sanity.ceiling_multiplier")),
            f"{exposure_ceiling:.0f}" if exposure_ceiling is not None else "unknown",
        )
    return kept, rejected


def _source_ref(figure: ExtractedFigure) -> str:
    """A stable per-figure identifier: the document plus the quote's digest.

    Several figures can come from one document, so the document id alone
    would collide on the candidate table's uniqueness constraint. Hashing
    the quote keeps the reference stable across runs — the same sentence
    always yields the same reference.
    """

    digest = hashlib.sha1(_normalise(figure.quote).encode("utf-8")).hexdigest()[:8]
    return f"{figure.doc_id}#{digest}"


def build_candidates(
    extraction,
    *,
    iso3: str,
    ym: str,
    hazard: str,
    rulebook: Rulebook,
    exposure_ceiling: float | None = None,
    ceiling_basis: dict[str, Any] | None = None,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    """The whole deterministic pipeline: figures in, ranked candidates out.

    ``preference_rank`` is 0 for the figure the rulebook's precedence
    prefers, and the reconciler reads it instead of falling back to
    "largest wins" on this rung.
    """

    iso3 = iso3.upper()
    rejected: list[dict[str, Any]] = list(extraction.rejected)

    converted: list[tuple[ExtractedFigure, float, dict[str, Any] | None]] = []
    for figure in extraction.figures:
        value, conversion = convert_units(figure, iso3, rulebook)
        converted.append((figure, value, conversion))

    conversions = {id(figure): conversion for figure, _v, conversion in converted}
    pairs = [(figure, value) for figure, value, _c in converted]

    pairs, ceiling_rejects = apply_ceiling(
        pairs, exposure_ceiling, rulebook, ceiling_basis
    )
    rejected.extend(ceiling_rejects)

    pairs, duplicates = deduplicate(pairs, rulebook)
    rejected.extend(duplicates)

    candidates: list[Candidate] = []
    for rank, (figure, value) in enumerate(pairs):
        tier_rank, tier = authority_tier(figure.stated_by, rulebook)
        conversion = conversions.get(id(figure))
        candidates.append(
            Candidate(
                iso3=iso3,
                ym=ym,
                hazard=hazard,
                value=float(value),
                value_type=VALUE_AFFECTED,
                source=SOURCE,
                source_ref=_source_ref(figure),
                stated_by=figure.stated_by or "(unattributed)",
                doc_url=figure.doc_url or None,
                extraction_model=figure.model,
                preference_rank=rank,
                detail={
                    "quote": figure.quote,
                    "unit_as_stated": figure.unit,
                    "stated_value": figure.value,
                    "area": figure.area,
                    "figure_date": figure.date,
                    "cumulative_or_new": figure.cumulative_or_new,
                    "authority_tier": tier,
                    "authority_rank": tier_rank,
                    "doc_id": figure.doc_id,
                    "doc_title": figure.doc_title,
                    "doc_date": figure.doc_date,
                    "household_conversion": conversion,
                },
            )
        )

    LOG.info(
        "[figures] %s %s %s: %d extracted figure(s) -> %d candidate(s), %d rejected",
        iso3, hazard, ym, len(extraction.figures), len(candidates), len(rejected),
    )
    _record_figures(
        iso3=iso3, ym=ym, hazard=hazard,
        candidates=candidates, rejected=rejected,
        exposure_ceiling=exposure_ceiling, ceiling_basis=ceiling_basis or {},
        rulebook=rulebook,
    )
    return candidates, rejected


def _record_figures(
    *,
    iso3: str,
    ym: str,
    hazard: str,
    candidates: list[Candidate],
    rejected: list[dict[str, Any]],
    exposure_ceiling: float | None,
    ceiling_basis: dict[str, Any],
    rulebook: Rulebook,
) -> None:
    """Append every figure and its fate to the run's evidence stream.

    Kept AND rejected: a ledger listing only the survivors cannot answer
    "what was rejected, against what ceiling", which is the question a
    reader has when the machine resolves less than it should.
    """

    if not cell_ledger.enabled():
        return
    multiplier = rulebook.get("sanity.ceiling_multiplier")
    for candidate in candidates:
        detail = candidate.detail or {}
        cell_ledger.record_figure(
            iso3=iso3, hazard=hazard, ym=ym, outcome="accepted",
            doc_id=str(detail.get("doc_id") or ""),
            doc_url=candidate.doc_url,
            value=candidate.value,
            unit=str(detail.get("unit_as_stated") or ""),
            stated_by=candidate.stated_by,
            quote=str(detail.get("quote") or ""),
            ceiling=exposure_ceiling,
            ceiling_multiplier=multiplier,
            ceiling_source=ceiling_basis.get("source"),
            ceiling_source_ref=ceiling_basis.get("source_ref"),
            ceiling_field=ceiling_basis.get("field"),
            preference_rank=candidate.preference_rank,
            detail={
                "authority_tier": detail.get("authority_tier"),
                "household_conversion": detail.get("household_conversion"),
                "figure_date": detail.get("figure_date"),
            },
        )
    for entry in rejected:
        cell_ledger.record_figure(
            iso3=iso3, hazard=hazard, ym=ym, outcome="rejected",
            doc_id=str(entry.get("doc_id") or ""),
            value=entry.get("value"),
            unit=str(entry.get("unit") or ""),
            quote=str(entry.get("quote") or ""),
            reason=str(entry.get("reason") or ""),
            ceiling=entry.get("exposed_population", exposure_ceiling),
            ceiling_multiplier=entry.get("ceiling_multiplier", multiplier),
            ceiling_source=entry.get("ceiling_source", ceiling_basis.get("source")),
            ceiling_source_ref=entry.get(
                "ceiling_source_ref", ceiling_basis.get("source_ref")
            ),
            ceiling_field=entry.get("ceiling_field", ceiling_basis.get("field")),
            detail={
                k: v
                for k, v in entry.items()
                if k
                not in {
                    "doc_id", "value", "unit", "quote", "reason",
                    "exposed_population", "ceiling_multiplier",
                    "ceiling_source", "ceiling_source_ref", "ceiling_field",
                }
            },
        )
