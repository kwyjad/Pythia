# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Apply the hazard-resolution ``haz_*`` schema to a resolver DuckDB.

Usage::

    python -m resolver.hazard_resolution.migrate [--db PATH_OR_URL]

Idempotent: safe to run repeatedly against a live DB. ``--db`` defaults to
the standard resolver target (``RESOLVER_DB_URL`` env var, else
``resolver/db/resolver.duckdb``), resolved by ``duckdb_io`` exactly as the
rest of the pipeline resolves it.
"""

from __future__ import annotations

import argparse
import logging
import sys

from resolver.db.duckdb_io import close_db, get_db
from resolver.hazard_resolution.rulebook import load_rulebook
from resolver.hazard_resolution.schema import ensure_haz_schema

LOG = logging.getLogger(__name__)


def migrate(db: str | None = None) -> dict[str, int]:
    """Ensure the ``haz_*`` schema on ``db``; return per-table row counts."""

    # Fail fast on a broken rulebook: a schema without a loadable rulebook
    # is a machine that cannot run.
    rulebook = load_rulebook()
    LOG.info("rulebook ok | path=%s", rulebook.path)

    conn = get_db(db)
    try:
        tables = ensure_haz_schema(conn)
        counts: dict[str, int] = {}
        for table in tables:
            counts[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        return counts
    finally:
        close_db(conn)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default=None,
        help="DuckDB path or duckdb:/// URL (default: standard resolver target)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    counts = migrate(args.db)
    for table, count in counts.items():
        print(f"[haz-migrate] {table}: {count} rows")
    print(f"[haz-migrate] ok — {len(counts)} tables ensured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
