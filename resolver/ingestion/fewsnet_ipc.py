# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""FEWS NET IPC connector entry point for ``scripts/ci/run_connectors.py``.

Delegates to the Connector-protocol ``FewsnetIpcConnector`` via the standard
Resolver pipeline (``resolver.tools.run_pipeline``).

Usage:
    python -m resolver.ingestion.fewsnet_ipc

ENV:
    FEWSNET_MONTHS         — months of history to fetch (default 12; 120 for
                             backfill to 2016)
    FEWSNET_REQUEST_DELAY  — seconds between retries (default 1.0)
"""

from __future__ import annotations

import logging
import os
import sys

LOG = logging.getLogger(__name__)


def main() -> None:
    from resolver.tools.run_pipeline import run_pipeline

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    LOG.info("[fewsnet_ipc] starting FEWS NET IPC ingestion via Resolver pipeline")

    try:
        db_url = os.getenv("RESOLVER_DB_URL") or None
        result = run_pipeline(connectors=["fewsnet_ipc"], db_url=db_url)
    except Exception as exc:
        LOG.error("[fewsnet_ipc] pipeline failed: %s", exc, exc_info=True)
        print(f"[fewsnet_ipc] ERROR: {exc}")
        sys.exit(1)

    total = result.total_facts
    LOG.info(
        "[fewsnet_ipc] pipeline complete: %d facts, %d resolved, %d deltas",
        total,
        result.resolved_rows,
        result.delta_rows,
    )
    # Print rows= line so run_connectors.py can parse the row count.
    print(f"[fewsnet_ipc] wrote rows={total}")
    if not result.db_written:
        LOG.warning(
            "[fewsnet_ipc] DB not written (RESOLVER_DB_URL=%s)",
            os.getenv("RESOLVER_DB_URL", "(unset)"),
        )


if __name__ == "__main__":
    main()
