#!/usr/bin/env python3
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""``resolve-hazards`` — the resolution machine's CLI.

Phase 1: cyclone detection and zero resolutions.

Usage:
    python -m resolver.resolution_machine.cli --hazard cyclone --month 2026-06
    resolve-hazards --hazard cyclone --month 2026-06            # poetry script
    resolve-hazards --hazard cyclone --month 2013-11 --scope ALL --countries PHL

Pipeline for one (hazard, month):
    1. fetch IBTrACS into haz_raw_ibtracs (idempotent; --skip-fetch reuses
       the store, --scope ALL backcasts)
    2. detect: track point >= cyclone.min_wind_kt within cyclone.buffer_km
       of territory → haz_triggers rows for EVERY country in the universe
    3. non-triggered country-months get a ReliefWeb silence sweep:
       silent → RESOLVED_ZERO with evidence_of_absence;
       hits   → triggered=true (trigger_source='reliefweb_sweep'),
                left for the impact ladder (Phase 2+)

Zeros are suppressed when the IBTrACS store does not demonstrably cover
the month (coverage gate) — months in ingestion gaps stay unresolved
rather than becoming false zeros.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Sequence

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

_HAZARD_ALIASES = {
    "cyclone": "cyclone",
    "tc": "cyclone",
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
        return str(rulebook["cyclone.ibtracs.default_scope"])
    return "ALL"


def _load_universe(rulebook) -> list[str]:
    """ISO3 universe from the resolver country registry."""
    import pandas as pd

    csv_path = REPO_ROOT / str(rulebook["universe.countries_csv"])
    df = pd.read_csv(csv_path, usecols=["iso3"])
    return sorted({str(c).strip().upper() for c in df["iso3"] if str(c).strip()})


def run_cyclone_month(
    *,
    ym: str,
    db_url: str | None,
    countries_filter: list[str] | None,
    scope: str,
    skip_fetch: bool,
    no_sweep: bool,
    dry_run: bool,
    rulebook=None,
) -> int:
    """Run the Phase-1 cyclone path for one month.  Returns an exit code."""
    from resolver.db.duckdb_io import get_db
    from resolver.resolution_machine import detect as detect_mod
    from resolver.resolution_machine import ibtracs as ibtracs_mod
    from resolver.resolution_machine import reliefweb_sweep as sweep_mod
    from resolver.resolution_machine import resolutions as res_mod
    from resolver.resolution_machine.geometry import load_country_geometries
    from resolver.resolution_machine.rulebook import load_rulebook
    from resolver.resolution_machine.schema import ensure_schema

    rulebook = rulebook or load_rulebook()
    con = get_db(db_url)
    ensure_schema(con)

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
            if summary["total_points"] > 0:
                LOG.error(
                    "[cli] IBTrACS fetch failed (%s) — continuing on the existing "
                    "store (%d points, newest %s); the coverage gate decides "
                    "whether zeros are safe",
                    exc,
                    summary["total_points"],
                    summary["max_iso_time"],
                )
            else:
                LOG.error("[cli] IBTrACS fetch failed and the store is empty: %s", exc)
                return 1

    # --- Step 2: detection ---
    geoms = load_country_geometries()
    universe = _load_universe(rulebook)
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
    n_zero = n_flipped = n_inconclusive = n_frozen = 0
    non_triggered = [r for r in result.rows if not r.triggered]
    if no_sweep:
        LOG.info("[cli] --no-sweep: skipping ReliefWeb silence checks and zeros")
    else:
        delay = float(rulebook["cyclone.reliefweb_sweep.request_delay_sec"])
        ibtracs_summary = ibtracs_mod.store_summary(con)
        for idx, row in enumerate(non_triggered):
            sweep = sweep_mod.sweep_country_month(row.iso3, ym, rulebook)
            if idx < len(non_triggered) - 1 and delay > 0:
                time.sleep(delay)
            if sweep["inconclusive"]:
                n_inconclusive += 1
                if not dry_run:
                    detect_mod.record_sweep_on_trigger(
                        con,
                        hazard_code=result.hazard_code,
                        iso3=row.iso3,
                        ym=ym,
                        sweep_evidence=sweep,
                    )
                continue
            if not sweep["silent"]:
                # Cyclone reporting despite no qualifying track (remnant
                # systems, naming edge cases): trigger for the ladder.
                n_flipped += 1
                LOG.info(
                    "[cli] %s %s: ReliefWeb sweep found %d reports despite no "
                    "IBTrACS trigger — flipping to triggered (reliefweb_sweep)",
                    row.iso3,
                    ym,
                    sweep["total_hits"],
                )
                if not dry_run:
                    detect_mod.flip_trigger_from_sweep(
                        con,
                        hazard_code=result.hazard_code,
                        iso3=row.iso3,
                        ym=ym,
                        sweep_evidence=sweep,
                        rulebook=rulebook,
                    )
                continue
            # Silent sweep — a zero, if IBTrACS coverage allows one.
            if not dry_run:
                detect_mod.record_sweep_on_trigger(
                    con,
                    hazard_code=result.hazard_code,
                    iso3=row.iso3,
                    ym=ym,
                    sweep_evidence=sweep,
                )
            if not result.coverage_ok:
                continue
            evidence = {
                "ibtracs": {
                    **ibtracs_summary,
                    "query": {
                        "ym": ym,
                        "min_wind_kt": rulebook["cyclone.min_wind_kt"],
                        "buffer_km": rulebook["cyclone.buffer_km"],
                        "n_points_month_global": result.n_points_month,
                        "n_points_qualifying_global": result.n_points_qualifying,
                        "n_storms_in_buffer": 0,
                        "coverage_note": result.coverage_note,
                    },
                },
                "reliefweb": sweep,
                "retrieved_at": sweep["retrieved_at"],
            }
            if dry_run:
                n_zero += 1
                continue
            outcome = res_mod.write_zero_resolution(
                con,
                iso3=row.iso3,
                hazard_code=result.hazard_code,
                ym=ym,
                evidence_of_absence=evidence,
                rulebook=rulebook,
            )
            if outcome == res_mod.WRITE_FROZEN_SKIP:
                n_frozen += 1
            else:
                n_zero += 1
        if not result.coverage_ok and non_triggered:
            LOG.warning(
                "[cli] coverage gate suppressed zeros for %s: %s",
                ym,
                result.coverage_note,
            )

    # --- Summary ---
    n_triggered = sum(1 for r in result.rows if r.triggered)
    LOG.info("--- resolve-hazards summary (%s %s) ---", result.hazard_code, ym)
    LOG.info("  countries assessed:        %d", len(result.rows))
    LOG.info("  triggered (ibtracs):       %d", n_triggered)
    LOG.info("  flipped by sweep:          %d", n_flipped)
    LOG.info("  resolved zero:             %d%s", n_zero, " (dry-run)" if dry_run else "")
    LOG.info("  sweep inconclusive:        %d", n_inconclusive)
    LOG.info("  frozen (skipped, logged):  %d", n_frozen)
    LOG.info("  ibtracs coverage ok:       %s (%s)", result.coverage_ok, result.coverage_note)
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="resolve-hazards",
        description="People-affected resolution machine (Phase 1: cyclone detection + zeros)",
    )
    parser.add_argument(
        "--hazard",
        required=True,
        choices=sorted(_HAZARD_ALIASES),
        type=str.lower,
        help="Hazard to resolve (Phase 1: cyclone / tc)",
    )
    parser.add_argument(
        "--month",
        required=True,
        type=_parse_month,
        help="Target month, YYYY-MM",
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
        help="Reuse the existing haz_raw_ibtracs store (no download)",
    )
    parser.add_argument(
        "--no-sweep",
        action="store_true",
        help="Skip ReliefWeb silence checks (no zeros are written)",
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
    if hazard != "cyclone":  # pragma: no cover - argparse choices guard this
        LOG.error("Hazard '%s' is not implemented yet", args.hazard)
        sys.exit(2)

    started = datetime.now(timezone.utc)
    rc = run_cyclone_month(
        ym=args.month,
        db_url=args.db or None,
        countries_filter=args.countries,
        scope=args.scope,
        skip_fetch=args.skip_fetch,
        no_sweep=args.no_sweep,
        dry_run=args.dry_run,
    )
    LOG.info(
        "resolve-hazards finished in %.1fs (exit %d)",
        (datetime.now(timezone.utc) - started).total_seconds(),
        rc,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
