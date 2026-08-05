#!/usr/bin/env python3
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""``resolve-hazards`` — the resolution machine's CLI.

Phases 1-3: cyclone and flood detection, zero resolutions, the
deterministic impact ladder, and LLM extraction of stated figures from
ReliefWeb documents.

Usage:
    resolve-hazards --hazard cyclone --month 2026-06
    resolve-hazards --hazard flood   --month 2026-06
    resolve-hazards --hazard drought --month 2026-06
    resolve-hazards --hazard flood   --months 2026-04 2026-05 2026-06
    resolve-hazards --hazard cyclone --month 2013-11 --scope ALL --countries PHL
    resolve-hazards --hazard flood   --month 2026-06 --no-extract  # spend nothing

Pipeline for one (hazard, month):

    1. DETECT — did a qualifying hazard occur?
       cyclone: IBTrACS track point >= cyclone.min_wind_kt within
                cyclone.buffer_km of territory
       flood:   GDACS alert >= flood.gdacs_trigger_level
       A haz_triggers row is written for EVERY country, triggered or not.

    2. ZEROS — non-triggered country-months get a ReliefWeb silence sweep:
       silent → RESOLVED_ZERO with evidence_of_absence;
       hits   → triggered=true (trigger_source='reliefweb_sweep'), handed
                to the ladder;
       failed → inconclusive, never a zero (fail-closed).

    3. LADDER — triggered country-months walk rulebook.ladder
       (EM-DAT → ReliefWeb-extracted → IFRC GO → IDMC IDU), take the
       highest rung with a figure, label IDU values as lower bounds, apply
       the GDACS exposure ceiling and national population cap, and flag
       (never rewrite) on conflict. No candidate → NO_DATA, flagged.

       Rung 2 is the only paid step: for a TRIGGERED cell whose higher
       rungs are empty, ReliefWeb documents are fetched and a cheap model
       transcribes the figures they state. Never for a non-triggered cell,
       and never past extraction.max_calls_per_month.

Drought does none of that. It has no discrete event, so it skips the
ladder and resolves against IPC/CH food-security analyses: people affected
is the increase in Phase 3+ population between consecutive analyses,
admitted as DROUGHT impact only where the drought indicators say the
country was in drought. Detection, zeros and freezing work exactly as
above; only the figure's source differs. No LLM call is involved, so a
drought run is free.

Answers written before month-end + freeze_days are marked provisional;
from the deadline on they are final and immutable.

Zeros are suppressed when the detector's store does not demonstrably
cover the month (coverage gate) — months in ingestion gaps stay
unresolved rather than becoming false zeros.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Sequence

LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
COUNTRIES_CSV_PATH = ROOT / "data" / "countries.csv"

_HAZARD_ALIASES = {
    "cyclone": "cyclone",
    "tc": "cyclone",
    "flood": "flood",
    "fl": "flood",
    "drought": "drought",
    "dr": "drought",
}

# Months younger than this many days are guaranteed inside the IBTrACS
# "last3years" rolling window; older months need the ALL archive.  This
# is scope SELECTION (which file to download), not a threshold that
# changes any resolution decision — those all live in the rulebook.
_LAST3YEARS_SAFE_DAYS = 2 * 365


def _parse_month(value: str) -> str:
    if not re.fullmatch(r"\d{4}-\d{2}", value):
        raise argparse.ArgumentTypeError(f"--month must be YYYY-MM, got '{value}'")
    year, month = int(value[:4]), int(value[5:7])
    if not 1 <= month <= 12:
        raise argparse.ArgumentTypeError(f"--month has invalid month: '{value}'")
    if date(year, month, 1) > date.today():
        raise argparse.ArgumentTypeError(f"--month {value} is in the future")
    return value


def _auto_scope(ym: str, rulebook) -> str:
    y, m = (int(p) for p in ym.split("-"))
    age_days = (date.today() - date(y, m, 1)).days
    if age_days <= _LAST3YEARS_SAFE_DAYS:
        return str(rulebook.get("cyclone.ibtracs.default_scope"))
    return "ALL"


def _load_universe() -> list[str]:
    """ISO3 universe from the resolver country registry."""
    import pandas as pd

    df = pd.read_csv(COUNTRIES_CSV_PATH, usecols=["iso3"], encoding="utf-8-sig")
    return sorted({str(c).strip().upper() for c in df["iso3"] if str(c).strip()})


@dataclass
class SweepTally:
    """Counts from the silence-sweep / zero-resolution step."""

    zeros: int = 0
    flipped: int = 0
    inconclusive: int = 0
    frozen: int = 0


def sweep_and_resolve_zeros(
    con,
    result,
    rulebook,
    *,
    hazard_key: str,
    detector_evidence: dict,
    dry_run: bool,
) -> SweepTally:
    """Sweep every non-triggered country-month and write the zeros.

    Shared by both hazards: the sweep, the fail-closed handling and the
    coverage gate are identical, and only the detector's own evidence
    block differs. A non-silent sweep FLIPS the cell to triggered so the
    impact ladder still sees it — a hazard nobody's physical detector
    caught but everybody reported is a real hazard.
    """
    import time

    from resolver.hazard_resolution import detect as detect_mod
    from resolver.hazard_resolution import reliefweb_sweep as sweep_mod
    from resolver.hazard_resolution import resolutions as res_mod

    tally = SweepTally()
    year, month = detect_mod.ym_to_year_month(result.ym)
    delay = float(rulebook.get(f"{hazard_key}.reliefweb_sweep.request_delay_sec"))
    non_triggered = [row for row in result.rows if not row.triggered]

    for idx, row in enumerate(non_triggered):
        sweep = sweep_mod.sweep_country_month(
            row.iso3, result.ym, rulebook, hazard_key=hazard_key
        )
        if idx < len(non_triggered) - 1 and delay > 0:
            time.sleep(delay)

        if sweep["inconclusive"]:
            tally.inconclusive += 1
            if not dry_run:
                detect_mod.record_sweep_on_trigger(
                    con, hazard=result.hazard, iso3=row.iso3,
                    ym=result.ym, sweep_evidence=sweep,
                )
            continue

        if not sweep["silent"]:
            tally.flipped += 1
            LOG.info(
                "[cli] %s %s: ReliefWeb sweep found %d reports despite no "
                "%s trigger — flipping to triggered (reliefweb_sweep)",
                row.iso3, result.ym, sweep["total_hits"], hazard_key,
            )
            if not dry_run:
                detect_mod.flip_trigger_from_sweep(
                    con, hazard=result.hazard, iso3=row.iso3,
                    ym=result.ym, sweep_evidence=sweep,
                )
            continue

        # Silent sweep — a zero, if the detector's coverage allows one.
        if not dry_run:
            detect_mod.record_sweep_on_trigger(
                con, hazard=result.hazard, iso3=row.iso3,
                ym=result.ym, sweep_evidence=sweep,
            )
        if not result.coverage_ok:
            continue

        evidence = {
            **detector_evidence,
            "reliefweb": sweep,
            "retrieved_at": sweep["retrieved_at"],
        }
        if dry_run:
            tally.zeros += 1
            continue
        outcome = res_mod.write_zero_resolution(
            con,
            iso3=row.iso3,
            year=year,
            month=month,
            hazard=result.hazard,
            evidence_of_absence=evidence,
            rulebook=rulebook,
        )
        if outcome == res_mod.WRITE_FROZEN_SKIP:
            tally.frozen += 1
        else:
            tally.zeros += 1

    if not result.coverage_ok and non_triggered:
        LOG.warning(
            "[cli] coverage gate suppressed zeros for %s: %s",
            result.ym, result.coverage_note,
        )
    return tally


def run_impact_ladder(
    con, ym: str, hazard: str, rulebook, *, skip_fetch: bool, dry_run: bool,
    no_extract: bool = False,
):
    """Fetch the ladder's sources and resolve every triggered cell."""
    from resolver.hazard_resolution import impact as impact_mod

    triggered = impact_mod.triggered_iso3s(con, ym, hazard)
    if not triggered:
        LOG.info("[cli] %s %s: no triggered cells — ladder has nothing to do", hazard, ym)
        return impact_mod.LadderRun(hazard=hazard, ym=ym)

    fetches = impact_mod.fetch_ladder_sources(
        con, ym, hazard, rulebook, skip_fetch=skip_fetch or dry_run
    )
    if no_extract:
        LOG.info(
            "[cli] --no-extract: ladder rung 2 (ReliefWeb extraction) is skipped; "
            "no LLM calls will be made"
        )
    return impact_mod.resolve_triggered_cells(
        con,
        ym=ym,
        hazard=hazard,
        iso3s=triggered,
        rulebook=rulebook,
        fetches=fetches,
        dry_run=dry_run,
        extract=not no_extract,
        # --skip-fetch means "reuse the stores"; for rung 2 that is the
        # cached documents, and the extraction cache makes re-reading them
        # free anyway.
        fetch_documents=not skip_fetch,
    )


def _log_summary(hazard: str, ym: str, result, tally: SweepTally, ladder, dry: str) -> None:
    n_triggered = sum(1 for r in result.rows if r.triggered)
    LOG.info("--- resolve-hazards summary (%s %s) ---", hazard, ym)
    LOG.info("  countries assessed:        %d", len(result.rows))
    LOG.info("  triggered (detector):      %d", n_triggered)
    LOG.info("  flipped by sweep:          %d", tally.flipped)
    LOG.info("  resolved zero:             %d%s", tally.zeros, dry)
    LOG.info("  sweep inconclusive:        %d", tally.inconclusive)
    LOG.info("  frozen (skipped, logged):  %d", tally.frozen)
    LOG.info("  detector coverage ok:      %s (%s)", result.coverage_ok, result.coverage_note)
    if ladder is not None:
        LOG.info("  --- impact ladder ---")
        LOG.info("  cells walked:              %d", ladder.cells)
        LOG.info("  resolved value:            %d%s", ladder.resolved_value, dry)
        LOG.info("    of which lower bounds:   %d", ladder.lower_bound)
        LOG.info("  no data (flagged):         %d", ladder.no_data)
        LOG.info("  pending (pre-freeze):      %d", ladder.pending)
        LOG.info("  flagged for review:        %d", ladder.flagged)
        LOG.info("  provisional (pre-freeze):  %d", ladder.provisional)
        LOG.info("  frozen (skipped, logged):  %d", ladder.frozen_skipped)
        LOG.info("  --- rung 2: ReliefWeb extraction ---")
        LOG.info("  cells read:                %d", ladder.cells_extracted)
        LOG.info("  cells skipped:             %d", ladder.cells_extraction_skipped)
        LOG.info("  documents read:            %d", ladder.docs_read)
        LOG.info("  LLM calls:                 %d", ladder.extraction_calls)
        LOG.info("  extraction spend:          $%.4f", ladder.extraction_cost_usd)
        if ladder.extraction_budget_capped:
            LOG.warning(
                "  extraction BUDGET CAPPED — some cells have unread documents"
            )
        if ladder.unavailable_sources:
            LOG.warning(
                "  ladder sources UNAVAILABLE: %s (their rungs are unread, not empty)",
                ",".join(ladder.unavailable_sources),
            )


def run_cyclone_month(
    *,
    ym: str,
    db_url: str | None,
    countries_filter: list[str] | None,
    scope: str,
    skip_fetch: bool,
    no_sweep: bool,
    dry_run: bool,
    no_ladder: bool = False,
    no_extract: bool = False,
    rulebook=None,
    con=None,
) -> int:
    """Run the cyclone path for one month.  Returns an exit code."""
    from resolver.db.duckdb_io import get_db
    from resolver.hazard_resolution import detect as detect_mod
    from resolver.hazard_resolution import ibtracs as ibtracs_mod
    from resolver.hazard_resolution.geometry import load_country_geometries
    from resolver.hazard_resolution.rulebook import load_rulebook
    from resolver.hazard_resolution.schema import ensure_haz_schema

    rulebook = rulebook or load_rulebook()
    con = con if con is not None else get_db(db_url)
    ensure_haz_schema(con)

    # --- Step 1: fetch (idempotent upsert) ---
    if dry_run and not skip_fetch:
        LOG.info("[cli] --dry-run implies --skip-fetch (no DB writes at all)")
        skip_fetch = True
    if not skip_fetch:
        if scope == "auto":
            scope = _auto_scope(ym, rulebook)
        try:
            frame, url = ibtracs_mod.fetch_ibtracs(rulebook, scope=scope)
            ibtracs_mod.store_ibtracs(con, frame, scope, url)
        except Exception as exc:
            summary = ibtracs_mod.store_summary(con)
            if summary["total_storms"] > 0:
                LOG.error(
                    "[cli] IBTrACS fetch failed (%s) — continuing on the existing "
                    "store (%d storms, newest point %s); the coverage gate decides "
                    "whether zeros are safe",
                    exc,
                    summary["total_storms"],
                    summary["max_point_time"],
                )
            else:
                LOG.error("[cli] IBTrACS fetch failed and the store is empty: %s", exc)
                return 1

    # --- Step 2: detection ---
    geoms = load_country_geometries()
    universe = _load_universe()
    if countries_filter:
        wanted = {c.strip().upper() for c in countries_filter}
        unknown = sorted(wanted - set(universe))
        if unknown:
            LOG.warning("[cli] --countries not in the universe: %s", ",".join(unknown))
        universe = [c for c in universe if c in wanted]
    no_geom = sorted(set(universe) - set(geoms))
    if no_geom:
        LOG.info(
            "[cli] %d universe countries lack boundary geometry and are skipped: %s",
            len(no_geom),
            ",".join(no_geom),
        )
    universe = [c for c in universe if c in geoms]
    if not universe:
        LOG.error("[cli] country universe is empty after filters")
        return 1

    result = detect_mod.detect_cyclone_month(
        con, ym, rulebook, geoms, iso3_filter=universe
    )
    if not dry_run:
        detect_mod.write_triggers(con, result, rulebook)

    # --- Step 3: silence sweep + zero resolutions for non-triggered ---
    tally = SweepTally()
    if no_sweep:
        LOG.info("[cli] --no-sweep: skipping ReliefWeb silence checks and zeros")
    else:
        detector_evidence = {
            "ibtracs": {
                **ibtracs_mod.store_summary(con),
                "query": {
                    "ym": ym,
                    "min_wind_kt": rulebook.get("cyclone.min_wind_kt"),
                    "buffer_km": rulebook.get("cyclone.buffer_km"),
                    "n_points_month_global": result.n_points_month,
                    "n_points_qualifying_global": result.n_points_qualifying,
                    "n_storms_in_buffer": 0,
                    "coverage_note": result.coverage_note,
                },
            },
        }
        tally = sweep_and_resolve_zeros(
            con, result, rulebook,
            hazard_key="cyclone",
            detector_evidence=detector_evidence,
            dry_run=dry_run,
        )

    # --- Step 4: impact ladder over the triggered cells ---
    ladder = None
    if no_ladder:
        LOG.info("[cli] --no-ladder: skipping the impact ladder")
    elif not dry_run:
        ladder = run_impact_ladder(
            con, ym, result.hazard, rulebook, skip_fetch=skip_fetch,
            dry_run=dry_run, no_extract=no_extract,
        )

    _log_summary(
        result.hazard, ym, result, tally, ladder, " (dry-run)" if dry_run else ""
    )
    return 0


def run_flood_month(
    *,
    ym: str,
    db_url: str | None,
    countries_filter: list[str] | None,
    skip_fetch: bool,
    no_sweep: bool,
    dry_run: bool,
    no_ladder: bool = False,
    no_extract: bool = False,
    rulebook=None,
    con=None,
) -> int:
    """Run the flood path for one month.  Returns an exit code.

    Same three steps as the cyclone path with GDACS as the detector: a
    country-month triggers on a GDACS flood alert at or above
    ``flood.gdacs_trigger_level``.
    """
    from resolver.db.duckdb_io import get_db
    from resolver.hazard_resolution import detect as detect_mod
    from resolver.hazard_resolution import gdacs as gdacs_mod
    from resolver.hazard_resolution.rulebook import (
        HAZARD_CODE_BY_RULEBOOK_NAME,
        load_rulebook,
    )
    from resolver.hazard_resolution.schema import ensure_haz_schema
    from resolver.hazard_resolution.sources import raw_store_summary

    rulebook = rulebook or load_rulebook()
    con = con if con is not None else get_db(db_url)
    ensure_haz_schema(con)
    hazard = HAZARD_CODE_BY_RULEBOOK_NAME["flood"]

    # --- Step 1: fetch GDACS events (idempotent upsert) ---
    if dry_run and not skip_fetch:
        LOG.info("[cli] --dry-run implies --skip-fetch (no DB writes at all)")
        skip_fetch = True
    if not skip_fetch:
        outcome = gdacs_mod.fetch_gdacs_events(con, ym, hazard, rulebook)
        if not outcome.ok:
            summary = raw_store_summary(con, "gdacs", hazard)
            if summary["total_records"] > 0:
                LOG.error(
                    "[cli] GDACS fetch failed (%s) — continuing on the existing "
                    "store (%d events); the coverage gate decides whether zeros "
                    "are safe",
                    outcome.error, summary["total_records"],
                )
            else:
                LOG.error(
                    "[cli] GDACS fetch failed and the store is empty: %s",
                    outcome.error,
                )
                return 1

    # --- Step 2: detection ---
    universe = _load_universe()
    if countries_filter:
        wanted = {c.strip().upper() for c in countries_filter}
        unknown = sorted(wanted - set(universe))
        if unknown:
            LOG.warning("[cli] --countries not in the universe: %s", ",".join(unknown))
        universe = [c for c in universe if c in wanted]
    if not universe:
        LOG.error("[cli] country universe is empty after filters")
        return 1

    result = detect_mod.detect_flood_month(con, ym, rulebook, iso3_filter=universe)
    if not dry_run:
        detect_mod.write_triggers(con, result, rulebook)

    # --- Step 3: silence sweep + zero resolutions for non-triggered ---
    tally = SweepTally()
    if no_sweep:
        LOG.info("[cli] --no-sweep: skipping ReliefWeb silence checks and zeros")
    else:
        detector_evidence = {
            "gdacs": {
                **raw_store_summary(con, "gdacs", hazard),
                "query": {
                    "ym": ym,
                    "gdacs_trigger_level": rulebook.get("flood.gdacs_trigger_level"),
                    "n_events_month_global": result.n_points_month,
                    "n_events_qualifying_global": result.n_points_qualifying,
                    "coverage_note": result.coverage_note,
                },
            },
        }
        tally = sweep_and_resolve_zeros(
            con, result, rulebook,
            hazard_key="flood",
            detector_evidence=detector_evidence,
            dry_run=dry_run,
        )

    # --- Step 4: impact ladder over the triggered cells ---
    ladder = None
    if no_ladder:
        LOG.info("[cli] --no-ladder: skipping the impact ladder")
    elif not dry_run:
        ladder = run_impact_ladder(
            con, ym, hazard, rulebook, skip_fetch=skip_fetch,
            dry_run=dry_run, no_extract=no_extract,
        )

    _log_summary(hazard, ym, result, tally, ladder, " (dry-run)" if dry_run else "")
    return 0


def _log_drought_summary(ym: str, run, dry: str) -> None:
    LOG.info("--- resolve-hazards summary (drought %s) ---", ym)
    LOG.info("  countries assessed:        %d", run.cells)
    LOG.info("  drought detected:          %d", run.triggered)
    LOG.info("  resolved value:            %d%s", run.resolved_value, dry)
    LOG.info("  resolved zero:             %d%s", run.resolved_zero, dry)
    LOG.info("  no data (flagged):         %d", run.no_data)
    LOG.info("  pending (pre-freeze):      %d", run.pending)
    LOG.info("  inconclusive (no row):     %d", run.inconclusive)
    LOG.info("  flagged for review:        %d", run.flagged)
    LOG.info("  provisional (pre-freeze):  %d", run.provisional)
    LOG.info("  frozen (skipped, logged):  %d", run.frozen_skipped)
    if run.unavailable_sources:
        LOG.warning(
            "  sources UNAVAILABLE: %s (their answers are unread, not empty)",
            ",".join(run.unavailable_sources),
        )


def run_drought_month(
    *,
    ym: str,
    db_url: str | None,
    countries_filter: list[str] | None,
    skip_fetch: bool,
    dry_run: bool,
    no_ladder: bool = False,
    no_extract: bool = False,
    no_sweep: bool = False,
    rulebook=None,
    con=None,
) -> int:
    """Run the drought path for one month.  Returns an exit code.

    Drought skips the impact ladder, so ``--no-ladder`` / ``--no-extract``
    are accepted and ignored: there is no ladder to skip and nothing to
    spend. ``--no-sweep`` is likewise inapplicable — drought's evidence of
    absence is the indicator feeds plus IPC, not a ReliefWeb sweep.
    """
    from resolver.db.duckdb_io import get_db
    from resolver.hazard_resolution import drought as drought_mod
    from resolver.hazard_resolution import drought_indicators as ind_mod
    from resolver.hazard_resolution import ipc as ipc_mod
    from resolver.hazard_resolution.rulebook import load_rulebook
    from resolver.hazard_resolution.schema import ensure_haz_schema

    rulebook = rulebook or load_rulebook()
    con = con if con is not None else get_db(db_url)
    ensure_haz_schema(con)

    for flag, name in ((no_ladder, "--no-ladder"), (no_extract, "--no-extract")):
        if flag:
            LOG.info(
                "[cli] %s has no effect on drought: it resolves from IPC, not "
                "from the impact ladder, and makes no model calls",
                name,
            )
    if no_sweep:
        LOG.info(
            "[cli] --no-sweep has no effect on drought: its evidence of absence "
            "is the drought indicators plus IPC, not a ReliefWeb sweep"
        )

    # --- Step 1: fetch IPC analyses and the indicator feeds ---
    if dry_run and not skip_fetch:
        LOG.info("[cli] --dry-run implies --skip-fetch (no DB writes at all)")
        skip_fetch = True

    if skip_fetch:
        LOG.info("[cli] --skip-fetch: using the existing IPC and indicator caches")
        fetches = {
            name: {"ok": True, "skipped": True} for name in ("ipc", "drought_indicators")
        }
    else:
        outcomes = {
            "ipc": ipc_mod.fetch_ipc(con, ym, rulebook),
            "drought_indicators": ind_mod.fetch_indicators(con, ym, rulebook),
        }
        for name, outcome in outcomes.items():
            if not outcome.ok:
                LOG.error(
                    "[cli] drought source %s unavailable for %s: %s",
                    name, ym, outcome.error,
                )
        fetches = {name: outcome.as_provenance() for name, outcome in outcomes.items()}
        # Without IPC there is no figure and no baseline for ANY country, so
        # a whole-month run would write nothing but pendings. The caches may
        # still carry earlier analyses, though, so this is only fatal when
        # the store is empty too.
        if not outcomes["ipc"].ok and ipc_mod.store_summary(con)["total_records"] == 0:
            LOG.error(
                "[cli] IPC fetch failed and the analysis store is empty: %s",
                outcomes["ipc"].error,
            )
            return 1

    # --- Step 2: the country universe ---
    universe = _load_universe()
    if countries_filter:
        wanted = {c.strip().upper() for c in countries_filter}
        unknown = sorted(wanted - set(universe))
        if unknown:
            LOG.warning("[cli] --countries not in the universe: %s", ",".join(unknown))
        universe = [c for c in universe if c in wanted]
    if not universe:
        LOG.error("[cli] country universe is empty after filters")
        return 1

    # --- Step 3: decide and persist every country-month ---
    run = drought_mod.run_month(
        con, ym=ym, iso3s=universe, rulebook=rulebook, fetches=fetches, dry_run=dry_run
    )
    _log_drought_summary(ym, run, " (dry-run)" if dry_run else "")
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="resolve-hazards",
        description=(
            "People-affected resolution machine "
            "(cyclone + flood detection, zeros, and the impact ladder)"
        ),
    )
    parser.add_argument(
        "--hazard",
        required=True,
        choices=sorted(_HAZARD_ALIASES),
        type=str.lower,
        help="Hazard to resolve: cyclone/tc, flood/fl or drought/dr",
    )
    months = parser.add_mutually_exclusive_group(required=True)
    months.add_argument(
        "--month",
        type=_parse_month,
        help="Target month, YYYY-MM",
    )
    months.add_argument(
        "--months",
        nargs="+",
        type=_parse_month,
        metavar="YYYY-MM",
        help="Several target months, run in order (each is independent)",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("RESOLVER_DB_URL", ""),
        help="DuckDB URL or path (default: $RESOLVER_DB_URL)",
    )
    parser.add_argument(
        "--countries",
        nargs="*",
        default=None,
        metavar="ISO3",
        help="Restrict to these ISO3 codes (default: full universe)",
    )
    parser.add_argument(
        "--scope",
        default="auto",
        choices=["auto", "last3years", "ALL"],
        help="IBTrACS file to fetch (auto: last3years for recent months, ALL for backcast)",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Reuse the existing raw stores (no downloads)",
    )
    parser.add_argument(
        "--no-sweep",
        action="store_true",
        help="Skip ReliefWeb silence checks (no zeros are written)",
    )
    parser.add_argument(
        "--no-ladder",
        action="store_true",
        help="Skip the impact ladder (detection and zeros only)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help=(
            "Skip ladder rung 2 (LLM extraction from ReliefWeb documents). "
            "No model is called and nothing is spent"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect and sweep but write nothing (implies --skip-fetch)",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("RESOLVER_LOG_LEVEL", "INFO"),
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    hazard = _HAZARD_ALIASES[args.hazard]
    target_months = args.months or [args.month]

    started = datetime.now(timezone.utc)
    common = dict(
        db_url=args.db or None,
        countries_filter=args.countries,
        skip_fetch=args.skip_fetch,
        no_sweep=args.no_sweep,
        no_ladder=args.no_ladder,
        no_extract=args.no_extract,
        dry_run=args.dry_run,
    )

    # Months are independent: one bad month must not cost the others their
    # run, so a non-zero month is recorded and the loop continues.
    failures = 0
    for ym in target_months:
        if hazard == "cyclone":
            rc = run_cyclone_month(ym=ym, scope=args.scope, **common)
        elif hazard == "drought":
            rc = run_drought_month(ym=ym, **common)
        else:
            rc = run_flood_month(ym=ym, **common)
        if rc != 0:
            failures += 1
            LOG.error("[cli] %s %s failed (exit %d)", hazard, ym, rc)

    if len(target_months) > 1:
        LOG.info(
            "[cli] ran %d months for %s: %d ok, %d failed",
            len(target_months), hazard, len(target_months) - failures, failures,
        )
    LOG.info(
        "resolve-hazards finished in %.1fs (exit %d)",
        (datetime.now(timezone.utc) - started).total_seconds(),
        1 if failures else 0,
    )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
