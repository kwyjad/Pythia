# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""GDACS connector entry point for ``scripts/ci/run_connectors.py``.

Delegates to the Connector-protocol ``GdacsConnector`` via the standard
Resolver pipeline (``resolver.tools.run_pipeline``).

Usage:
    python -m resolver.ingestion.gdacs

ENV:
    GDACS_MONTHS          — months of history to fetch (default 3)
    GDACS_REQUEST_DELAY   — seconds between RSS requests (default 1.0)
"""

from __future__ import annotations

import logging
import sys

LOG = logging.getLogger(__name__)


def main() -> None:
    from resolver.tools.run_pipeline import run_pipeline

    LOG.info("[gdacs] starting GDACS ingestion via Resolver pipeline")

    try:
        result = run_pipeline(connectors=["gdacs"])
    except Exception as exc:
        LOG.error("[gdacs] pipeline failed: %s", exc, exc_info=True)
        print(f"[gdacs] ERROR: {exc}")
        sys.exit(1)

    total = result.total_facts
    LOG.info(
        "[gdacs] pipeline complete: %d facts, %d resolved, %d deltas",
        total,
        result.resolved_rows,
        result.delta_rows,
    )
    # Print rows= line so run_connectors.py can parse the row count.
    print(f"[gdacs] wrote rows={total}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    main()
