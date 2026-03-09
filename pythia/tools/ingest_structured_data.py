# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Ingest per-country structured data into DuckDB.

Populates the ACAPS (INFORM Severity, Risk Radar, Daily Monitoring,
Humanitarian Access), IPC phase, and ReliefWeb report tables so that the
Horizon Scanner pipeline can read them at runtime.

Modelled on ``resolver.tools.fetch_conflict_forecasts`` but calls the
per-country fetch+store helpers exposed by ``pythia.acaps``,
``pythia.ipc_phases``, and ``horizon_scanner.reliefweb``.

Usage:
    python -m pythia.tools.ingest_structured_data
    python -m pythia.tools.ingest_structured_data --sources acaps ipc
    python -m pythia.tools.ingest_structured_data --iso3 AFG,SYR,YEM
    python -m pythia.tools.ingest_structured_data --dry-run
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------
# Each entry maps a short source name to a list of (label, fetch_fn, store_fn)
# tuples.  The callables are imported lazily so the module stays importable
# without pulling in every dependency at parse time.

_SOURCE_GROUPS: dict[str, list[str]] = {
    "acaps": [
        "acaps_inform_severity",
        "acaps_risk_radar",
        "acaps_daily_monitoring",
        "acaps_humanitarian_access",
    ],
    "ipc": [
        "ipc_phases",
    ],
    "reliefweb": [
        "reliefweb_reports",
    ],
}

ALL_SOURCE_NAMES = sorted(_SOURCE_GROUPS.keys())


# ---------------------------------------------------------------------------
# Lazy fetch/store dispatch
# ---------------------------------------------------------------------------

def _fetch_and_store_one(
    label: str,
    iso3: str,
    *,
    dry_run: bool = False,
) -> bool:
    """Fetch + optionally store a single source for one country.

    Returns True if data was obtained (and stored unless *dry_run*).
    """
    if label == "acaps_inform_severity":
        from pythia.acaps import fetch_inform_severity, store_inform_severity
        data = fetch_inform_severity(iso3)
        if data:
            if not dry_run:
                store_inform_severity(iso3, data)
            return True

    elif label == "acaps_risk_radar":
        from pythia.acaps import fetch_risk_radar, store_risk_radar
        data = fetch_risk_radar(iso3)
        if data:
            if not dry_run:
                store_risk_radar(iso3, data)
            return True

    elif label == "acaps_daily_monitoring":
        from pythia.acaps import fetch_daily_monitoring, store_daily_monitoring
        data = fetch_daily_monitoring(iso3)
        if data:
            if not dry_run:
                store_daily_monitoring(iso3, data)
            return True

    elif label == "acaps_humanitarian_access":
        from pythia.acaps import fetch_humanitarian_access, store_humanitarian_access
        data = fetch_humanitarian_access(iso3)
        if data:
            if not dry_run:
                store_humanitarian_access(iso3, data)
            return True

    elif label == "ipc_phases":
        from pythia.ipc_phases import fetch_ipc_phases, store_ipc_phases
        data = fetch_ipc_phases(iso3)
        if data:
            if not dry_run:
                store_ipc_phases(iso3, data)
            return True

    elif label == "reliefweb_reports":
        from horizon_scanner.reliefweb import (
            fetch_reliefweb_reports,
            store_reliefweb_reports,
        )
        reports = fetch_reliefweb_reports(iso3)
        if reports:
            if not dry_run:
                store_reliefweb_reports(iso3, reports)
            return True

    else:
        LOG.error("[ingest] unknown source label: %s", label)

    return False


# ---------------------------------------------------------------------------
# Country list loader
# ---------------------------------------------------------------------------

_COUNTRIES_CSV = Path(__file__).resolve().parents[2] / "resolver" / "data" / "countries.csv"


def _load_iso3_list(override: str | None = None) -> list[str]:
    """Return a sorted list of ISO-3 country codes.

    If *override* is given it should be a comma-separated string of codes.
    Otherwise the canonical ``resolver/data/countries.csv`` is read.
    """
    if override:
        return sorted({c.strip().upper() for c in override.split(",") if c.strip()})

    if not _COUNTRIES_CSV.exists():
        LOG.error("Countries file not found: %s", _COUNTRIES_CSV)
        sys.exit(1)

    iso3s: list[str] = []
    with open(_COUNTRIES_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            code = row.get("iso3", "").strip().upper()
            if code:
                iso3s.append(code)
    return sorted(set(iso3s))


# ---------------------------------------------------------------------------
# Main ingestion loop
# ---------------------------------------------------------------------------

def ingest(
    *,
    iso3_override: str | None = None,
    sources: Sequence[str] | None = None,
    dry_run: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run the structured-data ingestion loop.

    Returns a nested dict:
        { source_label: { "success": int, "fail": int, "empty": int } }
    """
    countries = _load_iso3_list(iso3_override)
    LOG.info("[ingest] %d countries to process", len(countries))

    # Resolve which individual labels to run
    if sources is None:
        sources = ALL_SOURCE_NAMES
    unknown = [s for s in sources if s not in _SOURCE_GROUPS]
    if unknown:
        raise ValueError(f"Unknown source group(s): {unknown}")

    labels: list[str] = []
    for src in sources:
        labels.extend(_SOURCE_GROUPS[src])

    LOG.info("[ingest] sources: %s  (labels: %s)", sources, labels)
    if dry_run:
        LOG.info("[ingest] DRY RUN — data will be fetched but NOT stored")

    # Per-label counters
    stats: dict[str, dict[str, int]] = {
        lbl: {"success": 0, "fail": 0, "empty": 0} for lbl in labels
    }
    # Per-country success/failure
    country_ok: dict[str, int] = {}
    country_fail: dict[str, int] = {}

    for idx, iso3 in enumerate(countries):
        LOG.info(
            "[ingest] [%d/%d] %s",
            idx + 1, len(countries), iso3,
        )
        c_ok = 0
        c_fail = 0

        for lbl in labels:
            try:
                got_data = _fetch_and_store_one(lbl, iso3, dry_run=dry_run)
                if got_data:
                    stats[lbl]["success"] += 1
                    c_ok += 1
                else:
                    stats[lbl]["empty"] += 1
            except Exception as exc:
                LOG.error(
                    "[ingest] %s / %s failed: %s", iso3, lbl, exc,
                    exc_info=True,
                )
                stats[lbl]["fail"] += 1
                c_fail += 1

        country_ok[iso3] = c_ok
        country_fail[iso3] = c_fail

        # Rate-limit between countries (ACAPS in particular)
        if idx < len(countries) - 1:
            time.sleep(1)

    # ---- Summary ----
    print("\n===== Ingestion Summary =====")
    if dry_run:
        print("MODE: DRY RUN (no data written)")
    print(f"Countries processed: {len(countries)}")
    print()

    print("Per-source row counts:")
    for lbl in labels:
        s = stats[lbl]
        print(f"  {lbl:40s}  ok={s['success']:4d}  empty={s['empty']:4d}  fail={s['fail']:4d}")

    total_ok = sum(v for v in country_ok.values())
    total_fail = sum(v for v in country_fail.values())
    print(f"\nPer-country totals: success={total_ok}  failure={total_fail}")

    failed_countries = [c for c, n in country_fail.items() if n > 0]
    if failed_countries:
        print(f"Countries with failures ({len(failed_countries)}): {', '.join(failed_countries)}")

    return stats


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest per-country structured data (ACAPS, IPC, ReliefWeb) "
            "into DuckDB for the Horizon Scanner pipeline."
        ),
    )
    parser.add_argument(
        "--iso3",
        default=None,
        help="Comma-separated ISO-3 codes to process (default: all from countries.csv)",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=ALL_SOURCE_NAMES,
        default=None,
        help="Which source groups to ingest (default: all)",
    )
    parser.add_argument(
        "--db",
        dest="db_url",
        default=None,
        help="Override PYTHIA_DB_URL (DuckDB connection string)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch data but do not write to DB",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.db_url:
        os.environ["PYTHIA_DB_URL"] = args.db_url

    try:
        stats = ingest(
            iso3_override=args.iso3,
            sources=args.sources,
            dry_run=args.dry_run,
        )
        total_fail = sum(s["fail"] for s in stats.values())
        if total_fail:
            LOG.warning("[ingest] completed with %d failure(s)", total_fail)
        else:
            LOG.info("[ingest] completed successfully")
    except Exception as exc:
        LOG.error("Fatal: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
