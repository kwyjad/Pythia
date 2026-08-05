# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The base-rate block the forecaster sees — Phase 6.

The machine's job does not end at writing ``haz_resolutions``. A forecast is
scored against a generating process, and a forecaster that does not know that
process is guessing at the target as well as at the world. This module renders
what the backcast learned — occurrence by calendar month, conditional severity,
and the provenance behind both — into a compact block that
``forecaster/prompts.py`` injects into the SPD prompt of every PA question the
machine resolves.

**Everything a forecaster is told about the rules is generated from
``rulebook.yaml`` at prompt-build time.** The buffer, the trigger level, the
ladder order, the freeze window, the drought rule: each is read from the
rulebook and rendered into the sentence describing it. Changing a threshold
therefore changes the prompt on the next run with no code change, which is the
only arrangement in which the prompt and the rulebook cannot drift apart. A
prompt that describes last quarter's rules is worse than one that describes
none, because it is confidently wrong.

**Three deliberate refusals.**

*It renders only for the metric the base rates are made of.* PA, and only PA.
Pythia's drought questions usually resolve on ``PHASE3PLUS_IN_NEED`` — a STOCK
of people in Phase 3+ — while this machine's drought values are the monthly
INCREASE in that stock. They are different quantities that happen to share a
source, and putting one where the other belongs is precisely the failure the
repo already paid for once (see the GDACS-exposure-as-IFRC-history entry in
CLAUDE.md). The eligibility test is in :func:`is_eligible`, and it is the only
gate a caller needs.

*It distinguishes "assessed and quiet" from "never assessed".* A country-hazard
with occurrence rows but no severity row has been looked at across the backcast
and never resolved to a value — "no historical events in record" is a true
statement about it. A country-hazard with no rows at all was never assessed,
and the honest rendering is no block, not a block full of zeros. Same
distinction the acceptance report's denominator makes, one cell at a time.

*It flags rather than truncates.* ``base_rates.prompt.max_chars`` is a declared
budget, not a cutter: an over-long block is logged and fails a test. A base rate
sliced off mid-table would still read as a complete one.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from resolver.hazard_resolution.rulebook import (
    HAZARD_CODE_BY_RULEBOOK_NAME,
    RULEBOOK_NAME_BY_HAZARD_CODE,
    Rulebook,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)

#: Hazard codes this machine resolves, derived from the rulebook's own hazard
#: map so a fourth hazard cannot be added there and forgotten here.
MACHINE_HAZARDS: frozenset[str] = frozenset(HAZARD_CODE_BY_RULEBOOK_NAME.values())

#: The one metric these base rates describe. ``haz_base_rates_severity`` holds
#: people-affected figures; nothing else may be anchored on them.
MACHINE_METRIC = "PA"

_MONTH_ABBR = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

_HAZARD_WORD = {"TC": "tropical cyclone", "FL": "flood", "DR": "drought"}

#: Ladder rung -> the short name a forecaster will recognise. An unmapped rung
#: renders as its raw key rather than being dropped: a ladder the prompt
#: describes incompletely is worse than one it describes awkwardly.
_RUNG_LABEL = {
    "emdat": "EM-DAT",
    "reliefweb_extracted": "ReliefWeb docs",
    "ifrc_go": "IFRC GO",
    # The lower-bound label stays spelled out even under the token budget: a
    # displacement count read as a people-affected figure is the one
    # misreading this rung can cause.
    "idmc_idu": "IDMC displacement (lower bound)",
}


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def is_eligible(hazard_code: str, metric: str) -> bool:
    """Is this question one the machine's base rates actually describe?

    True only for people-affected questions on a hazard the machine resolves.
    Two exclusions are load-bearing rather than conservative:

    ``PHASE3PLUS_IN_NEED``
        A stock of people already in Phase 3+, where the machine's drought
        value is the monthly increase in that stock. Same source, different
        quantity.
    ``EVENT_OCCURRENCE``
        Resolved from GDACS alert levels for every hazard, whereas this
        machine detects cyclones from IBTrACS track geometry. The occurrence
        rates below would answer a question about a different definition of
        "an event happened".
    """

    return (hazard_code or "").upper() in MACHINE_HAZARDS and (
        metric or ""
    ).upper() == MACHINE_METRIC


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OccurrenceRow:
    """One published (country, hazard, calendar month) occurrence rate."""

    calendar_month: int
    p_occurrence: float
    n_years: int
    source_window: str = ""


@dataclass(frozen=True)
class SeverityRow:
    """One published (country, hazard) conditional severity distribution."""

    q10: float | None
    q25: float | None
    q50: float | None
    q75: float | None
    q90: float | None
    n_events: int
    window_start: int | None
    window_end: int | None
    provenance_mix: Mapping[str, Any]


def _table_exists(con: "duckdb.DuckDBPyConnection", table: str) -> bool:
    try:
        rows = con.execute(
            "SELECT 1 FROM information_schema.tables WHERE lower(table_name) = ?",
            [table.lower()],
        ).fetchall()
    except Exception as exc:  # noqa: BLE001 - a missing catalog is not a crash
        LOG.debug("[prompt_block] table probe failed for %s: %s", table, exc)
        return False
    return bool(rows)


def load_base_rate_inputs(
    con: "duckdb.DuckDBPyConnection", iso3: str, hazard: str
) -> tuple[list[OccurrenceRow], SeverityRow | None]:
    """Read this country-hazard's published base rates out of the DB.

    Returns ``([], None)`` when the tables do not exist yet — the machine is
    not wired into a workflow, so a forecaster DB that predates it is the
    normal case, not an error.
    """

    iso3_up = (iso3 or "").upper().strip()
    hazard_up = (hazard or "").upper().strip()
    if not iso3_up or not hazard_up:
        return [], None

    occurrence: list[OccurrenceRow] = []
    if _table_exists(con, "haz_base_rates_occurrence"):
        rows = con.execute(
            """
            SELECT calendar_month, p_occurrence, n_years, COALESCE(source_window, '')
            FROM haz_base_rates_occurrence
            WHERE iso3 = ? AND hazard = ?
            ORDER BY calendar_month
            """,
            [iso3_up, hazard_up],
        ).fetchall()
        occurrence = [
            OccurrenceRow(
                calendar_month=int(month),
                p_occurrence=float(p),
                n_years=int(n_years),
                source_window=str(window or ""),
            )
            for month, p, n_years, window in rows
        ]

    severity: SeverityRow | None = None
    if _table_exists(con, "haz_base_rates_severity"):
        row = con.execute(
            """
            SELECT q10, q25, q50, q75, q90, n_events, window_start, window_end,
                   provenance_mix_json
            FROM haz_base_rates_severity
            WHERE iso3 = ? AND hazard = ?
            """,
            [iso3_up, hazard_up],
        ).fetchone()
        if row is not None:
            try:
                mix = json.loads(row[8]) if row[8] else {}
            except (TypeError, ValueError):
                mix = {}
            severity = SeverityRow(
                q10=None if row[0] is None else float(row[0]),
                q25=None if row[1] is None else float(row[1]),
                q50=None if row[2] is None else float(row[2]),
                q75=None if row[3] is None else float(row[3]),
                q90=None if row[4] is None else float(row[4]),
                n_events=int(row[5] or 0),
                window_start=None if row[6] is None else int(row[6]),
                window_end=None if row[7] is None else int(row[7]),
                provenance_mix=mix if isinstance(mix, Mapping) else {},
            )

    return occurrence, severity


# ---------------------------------------------------------------------------
# Terse rendering helpers
# ---------------------------------------------------------------------------


def terse_count(value: float | int | None) -> str:
    """A people count in as few characters as stay honest: 12,400 -> ``12k``.

    Two significant figures below 10 units, none above: ``1.2M`` and ``12M``
    both cost four characters, and neither pretends to a precision the
    quantile behind it has.
    """

    if value is None:
        return "n/a"
    try:
        magnitude = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if magnitude < 0:
        return "n/a"
    for divisor, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")):
        if magnitude >= divisor:
            scaled = magnitude / divisor
            return f"{scaled:.0f}{suffix}" if scaled >= 10 else f"{scaled:.1f}{suffix}"
    return f"{magnitude:.0f}"


def terse_pct(fraction: float | None) -> str:
    """A probability as a whole percentage, with ``<1%`` reserved for a rate
    that is small but not zero — rounding a real risk to ``0%`` is the one
    error a rare-event forecaster cannot recover from."""

    if fraction is None:
        return "n/a"
    value = float(fraction) * 100.0
    if value <= 0:
        return "0%"
    if value < 1:
        return "<1%"
    return f"{value:.0f}%"


# ---------------------------------------------------------------------------
# The rules, rendered from the rulebook
# ---------------------------------------------------------------------------


def resolution_process_text(rulebook: Rulebook, hazard: str) -> str:
    """The two sentences describing how this hazard's questions resolve.

    Every number and every name in the returned text is read from the
    rulebook. This is the whole point of the function: the forecaster is told
    the machine's actual parameters, not a paraphrase of them that a later
    threshold change would silently falsify.
    """

    hazard_up = (hazard or "").upper().strip()
    freeze_days = int(rulebook.get("freeze_days"))
    freeze = f"answers freeze {freeze_days}d after month end and never reopen"

    if hazard_up == "DR":
        threshold = int(rulebook.get("drought.ipc_phase_threshold"))
        min_people = rulebook.get("drought.min_delta_people")
        min_pct = rulebook.get("drought.min_delta_pct")
        floor_parts = []
        if float(min_people or 0) > 0:
            floor_parts.append(f"{terse_count(min_people)} people")
        if float(min_pct or 0) > 0:
            floor_parts.append(f"{float(min_pct):g}%")
        floor = f"a rise of {' or '.join(floor_parts)}+" if floor_parts else "any rise"
        return (
            f"HOW THIS RESOLVES: people affected = the rise in IPC/CH Phase "
            f"{threshold}+ population from the previous analysis to the one "
            f"covering the month, counted only where drought indicators show "
            f"drought; {floor} resolves to that number, no rise resolves ZERO, "
            f"a rise the indicators do not attribute to drought resolves no data. "
            f"It is a monthly increase, not the Phase {threshold}+ caseload; {freeze}."
        )

    if hazard_up == "TC":
        buffer_km = rulebook.get("cyclone.buffer_km")
        min_wind = rulebook.get("cyclone.min_wind_kt")
        detection = (
            f"a cyclone month = a best-track storm passing within "
            f"{buffer_km:g} km of the country at {min_wind:g} kt+"
        )
    else:  # FL
        level = str(rulebook.get("flood.gdacs_trigger_level")).lower()
        detection = f"a flood month = a GDACS flood alert at {level}+ over the country"

    ladder = [str(rung) for rung in rulebook.get("ladder")]
    ladder_text = " > ".join(_RUNG_LABEL.get(rung, rung) for rung in ladder)
    ceiling = float(rulebook.get("sanity.ceiling_multiplier"))
    cap_clause = "GDACS exposure"
    if ceiling != 1.0:
        cap_clause = f"{ceiling:g}x GDACS exposure"
    if bool(rulebook.get("sanity.population_cap")):
        cap_clause += " and national population"

    return (
        f"HOW THIS RESOLVES: {detection}; no qualifying event plus a silent "
        f"ReliefWeb sweep resolves ZERO, not missing. A detected month takes the "
        f"first figure on the fixed ladder {ladder_text}, capped by {cap_clause}; "
        f"{freeze}."
    )


def _window_end_year(window: str) -> int | None:
    """The end year of a ``YYYY-MM..YYYY-MM`` source window, or None."""

    try:
        end = str(window).split("..")[1]
        return int(end.split("-")[0])
    except (IndexError, ValueError, AttributeError):
        return None


def _hazard_caveat(
    rulebook: Rulebook,
    hazard: str,
    severity: SeverityRow | None,
    occurrence_window: str = "",
) -> str:
    """The one-line health warning a hazard's record needs, if it needs one."""

    hazard_up = (hazard or "").upper().strip()
    parts: list[str] = []

    name = RULEBOOK_NAME_BY_HAZARD_CODE.get(hazard_up)
    # The backcast lookup lives INSIDE the DR branch: only drought's caveat
    # uses it, and a missing backcast key must not be able to take down the
    # whole TC/FL block through the loader's blanket handler.
    if name and hazard_up == "DR":
        first_year = int(rulebook.get(f"backcast.{name}"))
        # Record depth ends where the OCCURRENCE record ends (the backcast
        # window), not where the severity window does — the two differ, and
        # severity may be absent entirely.
        window_end = _window_end_year(occurrence_window)
        if window_end is None and severity is not None and severity.window_end:
            window_end = int(severity.window_end)
        depth = f" (~{window_end - first_year + 1}y)" if window_end else ""
        parts.append(
            f"CAVEAT: the drought record starts {first_year}{depth} and IPC "
            "coverage is partial by country — a low rate here is weak "
            "evidence of a low rate in the world."
        )
        if str(rulebook.get("drought.month_attribution")) == "analysis_window":
            parts.append(
                "Values repeat across the months an analysis window covers; "
                "never sum them."
            )

    lower_bound_share = 0.0
    if severity is not None:
        try:
            lower_bound_share = float(severity.provenance_mix.get("lower_bound_share") or 0.0)
        except (TypeError, ValueError):
            lower_bound_share = 0.0
    if lower_bound_share > 0:
        parts.append(
            f"{terse_pct(lower_bound_share)} of the severity record is displacement "
            "counts — LOWER BOUNDS, so those quantiles sit below the truth."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------


def render_base_rate_block(
    *,
    iso3: str,
    hazard: str,
    forecast_months: Sequence[int],
    occurrence: Sequence[OccurrenceRow],
    severity: SeverityRow | None,
    rulebook: Rulebook,
    country_name: str = "",
) -> str:
    """Render the block. Pure: no DB, no clock, no network.

    Returns ``""`` when this country-hazard has no published base rates at
    all — see the module docstring on assessed-and-quiet versus never-assessed.
    """

    hazard_up = (hazard or "").upper().strip()
    iso3_up = (iso3 or "").upper().strip()
    if not occurrence and severity is None:
        return ""

    min_events = int(rulebook.get("base_rates.prompt.min_events_for_quantiles"))
    hazard_word = _HAZARD_WORD.get(hazard_up, hazard_up)
    where = f"{country_name} ({iso3_up})" if country_name else iso3_up

    months = [int(m) for m in forecast_months if 1 <= int(m) <= 12]
    if not months:
        months = list(range(1, 13))
    # Preserve the forecast window's own order (it wraps across a year end)
    # while never printing a calendar month twice.
    ordered: list[int] = []
    for month in months:
        if month not in ordered:
            ordered.append(month)

    by_month = {row.calendar_month: row for row in occurrence}

    lines = [
        f"PA RESOLUTION BASE RATES — {hazard_word} people affected, {where}",
    ]

    # The denominator is normally identical across a country's months, so it
    # is factored into the header rather than repeated six times. When it
    # differs it goes back onto each cell: an occurrence rate without its own
    # denominator is unreadable, and that is not a saving worth making.
    shown = [by_month[m] for m in ordered if m in by_month]
    shared_years = shown[0].n_years if shown and len({r.n_years for r in shown}) == 1 else None
    # The header's window comes from the months actually SHOWN (falling back
    # to any published month) — a header must not describe rows it isn't
    # rendering.
    window = next(
        (row.source_window for row in shown if row.source_window),
        next((row.source_window for row in occurrence if row.source_window), ""),
    )

    if not shown:
        # Every forecast-window month was withheld by the min_years floor.
        # A table of six n/a cells under a bare header would read as an
        # assertion of a table that carries no rate — one sentence says the
        # same thing honestly.
        lines.append(
            "P(qualifying event): no month in this forecast window has a "
            "published rate (denominator below base_rates.occurrence.min_years)."
        )
    else:
        header = "P(qualifying event)"
        if window:
            header += f", {window} backcast"
        if shared_years is not None:
            header += f", {shared_years}y assessed"
        lines.append(f"{header}:")
        cells = []
        for month in ordered:
            row = by_month.get(month)
            if row is None:
                # Published only above base_rates.occurrence.min_years: a rate
                # withheld for a thin denominator must not read as a low rate.
                cells.append(f"{_MONTH_ABBR[month]} n/a")
            elif shared_years is not None:
                cells.append(f"{_MONTH_ABBR[month]} {terse_pct(row.p_occurrence)}")
            else:
                cells.append(
                    f"{_MONTH_ABBR[month]} {terse_pct(row.p_occurrence)}(n={row.n_years}y)"
                )
        lines.append("  " + "  ".join(cells))

    if severity is None or severity.n_events <= 0:
        lines.append(
            "Severity | no historical events in record for this country-hazard."
        )
    elif severity.n_events < min_events:
        lines.append(
            f"Severity | {severity.n_events} resolved event(s) on record — too few "
            "for quantiles."
        )
    else:
        span = ""
        if severity.window_start and severity.window_end:
            span = f", {severity.window_start}-{severity.window_end}"
        lines.append(
            f"Severity | people affected in months WITH an event "
            f"(n={severity.n_events} events{span}):"
        )
        lines.append(
            "  q10 "
            + terse_count(severity.q10)
            + "  q25 "
            + terse_count(severity.q25)
            + "  q50 "
            + terse_count(severity.q50)
            + "  q75 "
            + terse_count(severity.q75)
            + "  q90 "
            + terse_count(severity.q90)
        )

    mix = {}
    if severity is not None and isinstance(severity.provenance_mix, Mapping):
        raw_mix = severity.provenance_mix.get("overall")
        if isinstance(raw_mix, Mapping):
            mix = raw_mix
    if mix:
        shares = sorted(mix.items(), key=lambda kv: (-float(kv[1] or 0.0), kv[0]))
        lines.append(
            "Figure sources: "
            + ", ".join(f"{source} {terse_pct(share)}" for source, share in shares)
        )

    caveat = _hazard_caveat(rulebook, hazard_up, severity, occurrence_window=window)
    if caveat:
        lines.append(caveat)

    lines.append(resolution_process_text(rulebook, hazard_up))

    block = "\n".join(lines)

    max_chars = int(rulebook.get("base_rates.prompt.max_chars"))
    if len(block) > max_chars:
        # Flagged, never cut. A block truncated mid-table still reads as a
        # complete base rate, which is how a prompt bug becomes a forecast bug.
        LOG.warning(
            "[prompt_block] %s/%s base-rate block is %d chars, over the "
            "base_rates.prompt.max_chars budget of %d — rendered in full",
            iso3_up, hazard_up, len(block), max_chars,
        )
    return block


def build_base_rate_block(
    iso3: str,
    hazard_code: str,
    metric: str,
    forecast_months: Sequence[int],
    *,
    con: "duckdb.DuckDBPyConnection",
    rulebook: Rulebook | None = None,
    country_name: str = "",
) -> str:
    """Load and render in one call; ``""`` when there is nothing to say.

    The caller owns the connection. Every failure path returns ``""``: an
    absent base rate must degrade to a prompt without one, never to a broken
    forecast run.
    """

    if not is_eligible(hazard_code, metric):
        return ""

    hazard_up = (hazard_code or "").upper().strip()
    try:
        if rulebook is None:
            from resolver.hazard_resolution.rulebook import load_rulebook

            rulebook = load_rulebook()
        occurrence, severity = load_base_rate_inputs(con, iso3, hazard_up)
        return render_base_rate_block(
            iso3=iso3,
            hazard=hazard_up,
            forecast_months=forecast_months,
            occurrence=occurrence,
            severity=severity,
            rulebook=rulebook,
            country_name=country_name,
        )
    except Exception as exc:  # noqa: BLE001 - a prompt inject never breaks a run
        LOG.warning(
            "[prompt_block] base-rate block unavailable for %s/%s: %s",
            iso3, hazard_up, exc,
        )
        return ""
