"""Minimal dependency preflight for ingestion workflows."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

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


def _check_duckdb() -> tuple[bool, Dict[str, Any], Optional[str]]:
    try:
        import duckdb  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        return False, {"dependency": "duckdb", "status": "missing", "error": str(exc)}, (
            f"missing_dependency: duckdb ({exc})"
        )

    details: Dict[str, Any] = {
        "dependency": "duckdb",
        "status": "ok",
        "version": getattr(duckdb, "__version__", "unknown"),
    }
    try:
        connection = duckdb.connect(":memory:")
        connection.close()
    except Exception as exc:  # pragma: no cover - defensive guard
        details["status"] = "error"
        details["error"] = str(exc)
        return False, details, f"duckdb_connect_failed: {exc}"

    return True, details, None


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

    if not args.skip_duckdb:
        ok, details, error_reason = _check_duckdb()
        extras["checks"].append(details)
        if ok:
            extras["duckdb_version"] = details.get("version")
        else:
            status = "error"
            reason = error_reason or "missing_dependency: duckdb"

    exit_code = 0 if status == "ok" else 1
    extras["exit_code"] = exit_code
    extras["status_raw"] = status

    _record_result(extras, status, reason)

    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
