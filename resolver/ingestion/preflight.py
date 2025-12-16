# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Minimal dependency preflight for ingestion workflows."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from resolver.db._duckdb_available import (
    DUCKDB_AVAILABLE,
    duckdb_unavailable_reason,
    get_duckdb,
)
from resolver.db.runtime_flags import (
    FAST_FIXTURES_ENV,
    resolve_fast_fixtures_mode,
)
from resolver.ingestion.diagnostics_emitter import (
    append_jsonl as diagnostics_append_jsonl,
    finalize_run as diagnostics_finalize_run,
    start_run as diagnostics_start_run,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTICS_DIR = REPO_ROOT / "diagnostics" / "ingestion"
CONNECTORS_REPORT = DIAGNOSTICS_DIR / "connectors_report.jsonl"

LOG = logging.getLogger("resolver.ingestion.preflight")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolver ingestion preflight checks")
    parser.add_argument(
        "--skip-duckdb",
        action="store_true",
        help="Skip DuckDB dependency checks",
    )
    return parser.parse_args(argv)


def _duckdb_status() -> tuple[Dict[str, Any], Optional[str]]:
    """Return a DuckDB availability report without importing the module directly."""

    if not DUCKDB_AVAILABLE:
        reason = duckdb_unavailable_reason() or "duckdb unavailable"
        return {
            "dependency": "duckdb",
            "status": "missing",
            "error": reason,
        }, reason

    try:
        module = get_duckdb()
    except Exception as exc:  # pragma: no cover - defensive guard
        reason = str(exc)
        return {
            "dependency": "duckdb",
            "status": "error",
            "error": reason,
        }, reason

    version = getattr(module, "__version__", "unknown")
    return {
        "dependency": "duckdb",
        "status": "ok",
        "version": version,
    }, None


def _ensure_diagnostics_dir() -> None:
    try:
        DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:  # pragma: no cover - diagnostics helper
        LOG.debug("preflight: unable to create diagnostics dir", exc_info=True)


def _record_result(extras: Mapping[str, Any], status: str, reason: str) -> None:
    context = diagnostics_start_run("preflight", "real")
    result = diagnostics_finalize_run(
        context,
        status,
        reason=reason,
        extras=extras,
    )
    diagnostics_append_jsonl(CONNECTORS_REPORT, result)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    LOG.setLevel(logging.INFO)

    _ensure_diagnostics_dir()

    status = "ok"
    reason = "dependencies ok"
    extras: Dict[str, Any] = {"checks": []}

    mode, auto_fallback, fallback_reason = resolve_fast_fixtures_mode()
    extras["fast_fixtures_mode"] = mode
    extras["fast_fixtures_auto_fallback"] = auto_fallback
    if fallback_reason:
        extras["fast_fixtures_reason"] = fallback_reason

    if mode != "duckdb":
        if auto_fallback:
            reason = "duckdb unavailable; noop mode active"
            LOG.warning(
                "Fast fixtures running in noop mode because DuckDB is unavailable (%s)",
                fallback_reason,
            )
        else:
            reason = f"noop mode requested via {FAST_FIXTURES_ENV}"
            LOG.info(
                "Fast fixtures noop mode requested via %s", FAST_FIXTURES_ENV
            )
        status = "noop"

    if args.skip_duckdb:
        extras["checks"].append(
            {"dependency": "duckdb", "status": "skipped", "reason": "--skip-duckdb"}
        )
    elif mode != "duckdb":
        extras["checks"].append(
            {
                "dependency": "duckdb",
                "status": "noop",
                "reason": reason,
            }
        )
    else:
        details, error_reason = _duckdb_status()
        extras["checks"].append(details)
        if details.get("status") == "ok":
            extras["duckdb_version"] = details.get("version")
        else:
            extras["duckdb_error"] = error_reason
            status = "noop"
            reason = error_reason or "duckdb unavailable; noop mode active"
            LOG.warning("DuckDB dependency unavailable: %s", error_reason)

    exit_code = 0
    extras["exit_code"] = exit_code
    extras["status_raw"] = status

    _record_result(extras, status, reason)

    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
