# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Load national population denominators into ``haz_raw_population``.

The simplest reliable route: the repo already curates national population
figures in ``resolver/data/population.csv`` (World Bank / WorldPop-derived,
also consumed by ``resolver/tools/denominators.py`` for per-capita
normalisation). This loader parses that file with the same value-parsing
rules and persists it into the resolver DB with retrieval timestamps, so the
resolution machine's sanity checks (``sanity.population_cap``) read
denominators from the DB like every other input.

Refreshing the denominators means refreshing the CSV (World Bank API or
COD-PS via HDX — both keyless) and re-running this loader; the loader itself
performs no network I/O.

Usage::

    python -m resolver.hazard_resolution.population [--db PATH_OR_URL] [--csv PATH]
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from resolver.db.duckdb_io import close_db, get_db
from resolver.hazard_resolution.schema import ensure_haz_schema

# Reuse the repo's population value/year parsing (handles "1,234,567" strings,
# floats, blanks) rather than re-implementing it here.
from resolver.tools.denominators import _parse_population, _parse_year

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = ROOT / "data" / "population.csv"
COUNTRIES_CSV_PATH = ROOT / "data" / "countries.csv"

_ISO3_RE = re.compile(r"^[A-Z]{3}$")


def _known_iso3s() -> frozenset[str] | None:
    """ISO3 codes from the repo country registry, or None when unavailable.

    population.csv carries World Bank aggregate rows (AFE, ARB, ...) whose
    codes are ISO3-shaped; filtering against countries.csv keeps only real
    countries out of the denominators.
    """

    if not COUNTRIES_CSV_PATH.exists():
        return None
    try:
        frame = pd.read_csv(COUNTRIES_CSV_PATH, dtype=str, encoding="utf-8-sig")
    except Exception:
        LOG.warning("could not read %s; skipping country filter", COUNTRIES_CSV_PATH, exc_info=True)
        return None
    columns = {str(col).strip().lower(): col for col in frame.columns}
    iso_col = columns.get("iso3")
    if iso_col is None:
        return None
    return frozenset(frame[iso_col].astype(str).str.strip().str.upper())


def load_population_records(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Parse the population CSV into the ``haz_raw_population`` column shape.

    Case/space-insensitive on column names; the year comes from a ``year``
    column when present, else ``as_of`` (the curated CSV's convention).
    """

    path = Path(csv_path) if csv_path is not None else DEFAULT_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(f"population CSV not found at {path}")

    raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    columns = {str(col).strip().lower(): col for col in raw.columns}

    def _series(name: str) -> pd.Series:
        col = columns.get(name)
        if col is None:
            return pd.Series([""] * len(raw), dtype=str)
        return raw[col].fillna("").astype(str)

    year_source = "year" if "year" in columns else "as_of"
    records = pd.DataFrame(
        {
            "iso3": _series("iso3").str.strip().str.upper(),
            "year": _series(year_source).map(_parse_year),
            "population": _series("population").map(_parse_population),
            "source": _series("source").str.strip(),
            "product": _series("product").str.strip(),
            "download_date": _series("download_date").str.strip(),
            "source_url": _series("source_url").str.strip(),
            "notes": _series("notes").str.strip(),
        }
    )

    total = len(records)
    records = records[records["iso3"].str.match(_ISO3_RE)]
    records = records[records["year"].notna() & records["population"].notna()]
    records = records[records["population"] > 0]

    known = _known_iso3s()
    dropped_aggregates = 0
    if known is not None:
        before = len(records)
        records = records[records["iso3"].isin(known)]
        dropped_aggregates = before - len(records)

    records = records.drop_duplicates(subset=["iso3", "year", "source"], keep="last")
    records = records.reset_index(drop=True)
    records["year"] = records["year"].astype(int)
    records["population"] = records["population"].astype("int64")
    LOG.info(
        "population csv parsed | path=%s rows_in=%s rows_out=%s dropped_non_country=%s",
        path,
        total,
        len(records),
        dropped_aggregates,
    )
    return records


def load_population_into_db(
    conn: "duckdb.DuckDBPyConnection",
    csv_path: str | Path | None = None,
    retrieved_at: dt.datetime | None = None,
) -> int:
    """Upsert the population CSV into ``haz_raw_population``; return the row count.

    Idempotent per (iso3, year, source): re-running replaces matching rows
    with a fresh retrieval timestamp instead of duplicating them.
    """

    ensure_haz_schema(conn)
    records = load_population_records(csv_path)
    if records.empty:
        LOG.warning("population csv produced no loadable rows")
        return 0

    stamp = retrieved_at or dt.datetime.now(dt.timezone.utc)
    frame = records.copy()
    frame["retrieved_at"] = stamp.replace(tzinfo=None)

    conn.register("_haz_population_incoming", frame)
    try:
        conn.execute("BEGIN TRANSACTION")
        conn.execute(
            """
            DELETE FROM haz_raw_population
            WHERE EXISTS (
                SELECT 1 FROM _haz_population_incoming i
                WHERE i.iso3 = haz_raw_population.iso3
                  AND i.year = haz_raw_population.year
                  AND i.source = haz_raw_population.source
            )
            """
        )
        conn.execute(
            """
            INSERT INTO haz_raw_population
                (iso3, year, population, source, product, download_date,
                 source_url, notes, retrieved_at)
            SELECT iso3, year, population, source, product, download_date,
                   source_url, notes, retrieved_at
            FROM _haz_population_incoming
            """
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.unregister("_haz_population_incoming")

    LOG.info("haz_raw_population loaded | rows=%s retrieved_at=%s", len(frame), stamp.isoformat())
    return len(frame)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default=None,
        help="DuckDB path or duckdb:/// URL (default: standard resolver target)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help=f"Population CSV path (default: {DEFAULT_CSV_PATH})",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    conn = get_db(args.db)
    try:
        rows = load_population_into_db(conn, args.csv)
    finally:
        close_db(conn)
    print(f"[haz-population] loaded {rows} rows into haz_raw_population")
    return 0


if __name__ == "__main__":
    sys.exit(main())
