# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""IPC API connector entry point for ``scripts/ci/run_connectors.py``.

Delegates to the Connector-protocol ``IpcApiConnector`` via the standard
Resolver pipeline (``resolver.tools.run_pipeline``).

Usage:
    python -m resolver.ingestion.ipc_api

ENV:
    IPC_API_KEY            — API key for api.ipcinfo.org (required)
    IPC_API_MONTHS         — months of history to fetch (default 24)
    IPC_API_REQUEST_DELAY  — seconds between retries (default 1.0)
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

    LOG.info("[ipc_api] starting IPC API ingestion via Resolver pipeline")

    try:
        db_url = os.getenv("RESOLVER_DB_URL") or None
        result = run_pipeline(connectors=["ipc_api"], db_url=db_url)
    except Exception as exc:
        LOG.error("[ipc_api] pipeline failed: %s", exc, exc_info=True)
        print(f"[ipc_api] ERROR: {exc}")
        sys.exit(1)

    total = result.total_facts
    LOG.info(
        "[ipc_api] pipeline complete: %d facts, %d resolved, %d deltas",
        total,
        result.resolved_rows,
        result.delta_rows,
    )
    # Print rows= line so run_connectors.py can parse the row count.
    print(f"[ipc_api] wrote rows={total}")
    if not result.db_written:
        LOG.warning(
            "[ipc_api] DB not written (RESOLVER_DB_URL=%s)",
            os.getenv("RESOLVER_DB_URL", "(unset)"),
        )


if __name__ == "__main__":
    main()
