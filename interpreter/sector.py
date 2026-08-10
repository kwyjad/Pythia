# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Where Fred departs from the prevailing institutional view.

The report spent pages restating what the Humanitarian Needs Overview already
says. Everyone knows Sudan and Somalia are catastrophic. The value in a
forecasting system is where it disagrees with the settled picture, in either
direction, early enough to matter.

We already ingest ACAPS Risk Radar and INFORM Severity. Ranking the same
countries three ways and reading off the gaps is the cheapest honest version
of that comparison.

**These measure different things and are not commensurable.** INFORM Severity
scores a crisis as it stands now, across every driver at once. Risk Radar is
an analyst's forward judgement about specified risks. Fred's excess is how
many more people one hazard is expected to affect next month than history
would suggest. A gap between them is a prompt to look, never a verdict that
one is wrong, and `NOT_COMMENSURABLE` says so in the report's own fixed words
rather than leaving the framing to the model.

Pure functions over plain dicts.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

NOT_COMMENSURABLE = (
    "These three rankings measure different things. The severity scores "
    "describe a crisis as it stands, across every driver at once. Fred's "
    "figure is how many more people one hazard is expected to affect over the "
    "next six months than its own history would suggest. A country can sit "
    "high on one and low on another for good reasons. Read the gaps as a "
    "prompt to look again, not as a claim that anyone is wrong."
)

# Risk Radar states a level in words; INFORM states a number. Mapping the
# words onto a scale is a judgement, so it is written out here where it can be
# read, rather than hidden in a SQL CASE.
RISK_LEVEL_SCORES = {
    "very high": 5.0,
    "high": 4.0,
    "significant": 3.5,
    "medium": 3.0,
    "moderate": 3.0,
    "low": 2.0,
    "very low": 1.0,
}

SOURCE_LABELS = {
    "inform_severity": "INFORM Severity",
    "risk_radar": "ACAPS Risk Radar",
}


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def risk_radar_scores(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Per-country Risk Radar score: the worst risk on file for that country.

    A country carries several risk rows. The one that matters to a reader
    scanning for trouble is the worst of them, so the maximum is taken rather
    than a mean, which would dilute one severe risk with several mild ones.
    """
    out: dict[str, float] = {}
    for row in rows:
        iso3 = str(row.get("iso3") or "").upper()
        if not iso3:
            continue
        score = _as_float(row.get("impact"))
        if score is None:
            score = RISK_LEVEL_SCORES.get(
                str(row.get("risk_level") or "").strip().lower()
            )
        if score is None:
            continue
        out[iso3] = max(out.get(iso3, float("-inf")), score)
    return out


def inform_severity_scores(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Per-country INFORM Severity: the worst crisis in the latest snapshot.

    A country with two crises is as bad as its worst one, for the same reason
    as above. Rows are expected pre-filtered to the newest snapshot date.
    """
    out: dict[str, float] = {}
    for row in rows:
        iso3 = str(row.get("iso3") or "").upper()
        score = _as_float(row.get("severity_score"))
        if not iso3 or score is None:
            continue
        out[iso3] = max(out.get(iso3, float("-inf")), score)
    return out


def fred_scores(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Per-country expected excess: the sum over that country's forecasts.

    Summed, not maximised, because excess is in people and two hazards in one
    country are two calls on the same response capacity. Only positive excess
    counts: a country expecting fewer people affected than usual is not
    something Fred is worried about, and adding a negative would let one
    quiet hazard cancel a loud one.
    """
    out: dict[str, float] = {}
    for row in rows:
        iso3 = str(row.get("iso3") or "").upper()
        excess = _as_float(row.get("excess_nominal"))
        if not iso3 or excess is None or excess <= 0:
            continue
        out[iso3] = out.get(iso3, 0.0) + excess
    return out


def ranks(values: Mapping[str, float]) -> dict[str, int]:
    """1-based rank, largest first, ties broken by ISO3 so runs are stable."""
    ordered = sorted(values.items(), key=lambda kv: (-float(kv[1]), kv[0]))
    return {iso3: i for i, (iso3, _) in enumerate(ordered, start=1)}


def compare(
    fred: Mapping[str, float],
    sector: Mapping[str, float],
    *,
    source: str,
    list_size: int,
    min_rank_gap: int,
) -> dict[str, Any]:
    """Rank gaps between Fred and one sector index.

    Only countries BOTH rank are compared. A country ACAPS does not cover
    cannot produce a gap, and treating its absence as a low sector rank would
    manufacture disagreement out of coverage.
    """
    common = sorted(set(fred) & set(sector))
    if not common:
        return {
            "source": source,
            "source_label": SOURCE_LABELS.get(source, source),
            "n_compared": 0,
            "more_worried": [],
            "less_worried": [],
            "note": "no countries appear in both rankings",
        }
    fred_ranks = ranks({k: fred[k] for k in common})
    sector_ranks = ranks({k: sector[k] for k in common})

    rows: list[dict[str, Any]] = []
    for iso3 in common:
        # Positive gap: Fred ranks the country higher (a smaller rank number)
        # than the sector index does.
        gap = sector_ranks[iso3] - fred_ranks[iso3]
        rows.append({
            "iso3": iso3,
            "fred_rank": fred_ranks[iso3],
            "sector_rank": sector_ranks[iso3],
            "rank_gap": gap,
            "fred_excess": round(float(fred[iso3]), 1),
            "sector_score": round(float(sector[iso3]), 2),
        })

    more = sorted(
        (r for r in rows if r["rank_gap"] >= min_rank_gap),
        key=lambda r: (-r["rank_gap"], r["fred_rank"]),
    )[:list_size]
    less = sorted(
        (r for r in rows if r["rank_gap"] <= -min_rank_gap),
        key=lambda r: (r["rank_gap"], r["sector_rank"]),
    )[:list_size]
    return {
        "source": source,
        "source_label": SOURCE_LABELS.get(source, source),
        "n_compared": len(common),
        "min_rank_gap": min_rank_gap,
        "more_worried": more,
        "less_worried": less,
    }


def build(
    attention_rows: Sequence[Mapping[str, Any]],
    *,
    inform_rows: Iterable[Mapping[str, Any]] = (),
    risk_rows: Iterable[Mapping[str, Any]] = (),
    list_size: int,
    min_rank_gap: int,
) -> dict[str, Any]:
    """The whole comparison block for the pack."""
    fred = fred_scores(attention_rows)
    comparisons = []
    for source, values in (
        ("inform_severity", inform_severity_scores(inform_rows)),
        ("risk_radar", risk_radar_scores(risk_rows)),
    ):
        if not values:
            continue
        comparisons.append(
            compare(
                fred, values, source=source,
                list_size=list_size, min_rank_gap=min_rank_gap,
            )
        )
    return {
        "caveat": NOT_COMMENSURABLE,
        "n_countries_with_excess": len(fred),
        "comparisons": comparisons,
    }
