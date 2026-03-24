# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Emit a single diagnostics JSONL line for a Phase 4 context source.

Usage (from workflow YAML):
    python -m pythia.tools.emit_phase4_diag enso ok --rows 1
    python -m pythia.tools.emit_phase4_diag hdx_signals ok --rows 847
    python -m pythia.tools.emit_phase4_diag seasonal_tc error --reason "fetch-failed"
"""

from __future__ import annotations

import argparse
import os
import sys


def emit(
    connector_id: str,
    status: str,
    rows: int = 0,
    reason: str = "",
    extras: dict | None = None,
) -> None:
    path = os.environ.get("DIAGNOSTICS_REPORT_PATH")
    if not path:
        return
    from resolver.ingestion.diagnostics_emitter import (
        start_run,
        finalize_run,
        append_jsonl,
    )

    ctx = start_run(connector_id, "real")
    result = finalize_run(
        ctx,
        status,
        reason=reason or None,
        counts={"written": rows} if rows else {},
        extras=extras or {},
    )
    append_jsonl(path, result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit a Phase 4 diagnostics JSONL line",
    )
    parser.add_argument("connector_id", help="Source identifier (e.g., enso, hdx_signals)")
    parser.add_argument("status", choices=["ok", "error", "skipped"], help="Run status")
    parser.add_argument("--rows", type=int, default=0, help="Number of rows written")
    parser.add_argument("--reason", default="", help="Reason string for non-ok status")
    args = parser.parse_args()
    emit(args.connector_id, args.status, args.rows, args.reason)


if __name__ == "__main__":
    main()
