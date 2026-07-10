# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Purge cross-contaminated acled_political_events rows.

The pre-July-2026 fetcher passed ``iso3`` as an (unsupported, silently
ignored) ACLED API filter param, so every country stored the SAME ~50 global
events stamped with its own iso3 — e.g. Iranian pro-regime rallies injected
into Somalia's ACE prompts. This script is the manual counterpart to the
self-heal in ``pythia.tools.ingest_structured_data._bulk_fetch_acled_political``.

Usage:
    python -m scripts.purge_acled_political_events           # purge only if contaminated
    python -m scripts.purge_acled_political_events --force   # wipe unconditionally

Idempotent: a clean (or empty/missing) table is a no-op. Repopulate with:
    python -m pythia.tools.ingest_structured_data --sources acled_political
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOG = logging.getLogger("purge_acled_political_events")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete all rows even if no contamination is detected.",
    )
    args = parser.parse_args()

    if args.force:
        from pythia.db.schema import connect

        con = connect(read_only=False)
        try:
            try:
                total = con.execute(
                    "SELECT COUNT(*) FROM acled_political_events"
                ).fetchone()
            except Exception:
                LOG.info("acled_political_events table missing — nothing to purge.")
                return 0
            n_rows = int(total[0] or 0) if total else 0
            con.execute("DELETE FROM acled_political_events")
            LOG.info("Force-purged %d acled_political_events rows.", n_rows)
        finally:
            con.close()
        return 0

    from pythia.acled_political import purge_contaminated_events

    purged = purge_contaminated_events()
    if purged:
        LOG.info("Purged %d contaminated acled_political_events rows.", purged)
    else:
        LOG.info("No cross-country contamination detected — nothing purged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
