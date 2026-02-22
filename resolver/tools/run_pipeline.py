#!/usr/bin/env python3
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""
Resolver pipeline orchestrator.

Iterates registered connectors, validates their output, enriches the
combined facts, runs precedence resolution, computes deltas, and writes
to DuckDB.

Usage:
    python -m resolver.tools.run_pipeline
    python -m resolver.tools.run_pipeline --connectors acled idmc
    python -m resolver.tools.run_pipeline --db path/to/resolver.duckdb
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import yaml

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


@dataclass
class ConnectorResult:
    name: str
    rows: int
    status: str  # "ok" | "empty" | "error"
    error: str = ""


@dataclass
class PipelineResult:
    connector_results: List[ConnectorResult] = field(default_factory=list)
    total_facts: int = 0
    resolved_rows: int = 0
    delta_rows: int = 0
    db_written: bool = False


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    *,
    db_url: str | None = None,
    connectors: Sequence[str] | None = None,
    dry_run: bool = False,
) -> PipelineResult:
    """Run the full Resolver pipeline.

    1. Fetch + normalise data from each connector
    2. Validate canonical schema
    3. Enrich with registry lookups
    4. Derive ``ym`` (month key) from ``as_of_date``
    5. Run precedence resolution
    6. Compute deltas
    7. Write to DuckDB
    """
    from resolver.connectors import discover_connectors
    from resolver.connectors.validate import validate_canonical
    from resolver.tools.enrich import derive_ym, enrich

    result = PipelineResult()

    # --- Step 1-3: fetch, validate, collect ---
    all_facts: list[pd.DataFrame] = []
    active_connectors = discover_connectors(connectors)

    for connector in active_connectors:
        cr = ConnectorResult(name=connector.name, rows=0, status="ok")
        try:
            df = connector.fetch_and_normalize()
            validate_canonical(df, source=connector.name)
            cr.rows = len(df)
            if df.empty:
                cr.status = "empty"
                LOG.info("[%s] produced 0 rows", connector.name)
            else:
                all_facts.append(df)
                LOG.info("[%s] produced %d rows", connector.name, len(df))
        except Exception as exc:
            cr.status = "error"
            cr.error = str(exc)
            LOG.error("[%s] failed: %s", connector.name, exc)
        result.connector_results.append(cr)

    if not all_facts:
        LOG.warning("No facts collected from any connector")
        return result

    combined = pd.concat(all_facts, ignore_index=True)
    result.total_facts = len(combined)

    # --- Step 4-5: enrich + derive ym ---
    combined = enrich(combined)
    combined = derive_ym(combined)

    # --- Step 6: precedence resolution ---
    resolved = _run_precedence(combined)
    result.resolved_rows = len(resolved) if resolved is not None else 0

    # --- Step 7: compute deltas ---
    deltas = _run_deltas(resolved)
    result.delta_rows = len(deltas) if deltas is not None else 0

    # --- Step 8: write to DuckDB ---
    if not dry_run and db_url:
        _write_to_db(db_url, resolved, deltas)
        result.db_written = True

    _log_summary(result)
    return result


# ---------------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------------

_PRECEDENCE_CONFIG = ROOT / "tools" / "precedence_config.yml"


def _load_precedence_config() -> dict:
    """Load and normalise the precedence config.

    ``precedence_config.yml`` uses a list-of-lists shorthand for tiers::

        tiers:
          - ["acled", "idmc"]
          - ["emdat"]

    The precedence engine expects the explicit dict form::

        tiers:
          - name: "Tier 0"
            sources: ["acled", "idmc"]

    This function converts between the two formats.
    """
    if not _PRECEDENCE_CONFIG.exists():
        return {}
    with open(_PRECEDENCE_CONFIG, "r") as fp:
        cfg = yaml.safe_load(fp) or {}

    # Normalise tiers from list-of-lists to list-of-dicts if needed.
    raw_tiers = cfg.get("tiers", [])
    normalised: list[dict] = []
    for idx, tier in enumerate(raw_tiers):
        if isinstance(tier, list):
            normalised.append({"name": f"Tier {idx}", "sources": tier})
        elif isinstance(tier, dict):
            normalised.append(tier)
        else:
            normalised.append({"name": f"Tier {idx}", "sources": [str(tier)]})
    cfg["tiers"] = normalised
    return cfg


def _run_precedence(combined: pd.DataFrame) -> pd.DataFrame | None:
    """Map canonical columns to precedence-engine names and resolve."""
    from resolver.tools.precedence_engine import resolve_facts_frame

    if combined.empty:
        return None

    # The precedence engine uses different column names.
    work = combined.copy()
    work = work.rename(columns={
        "iso3": "country_iso3",
        "hazard_code": "hazard_type",
        "ym": "month",
        "as_of_date": "as_of",
    })

    # The engine needs a "source" column (connector name / publisher).
    if "source" not in work.columns:
        work["source"] = ""

    # The engine sorts by run_id as a final tiebreaker.
    if "run_id" not in work.columns:
        work["run_id"] = ""

    config = _load_precedence_config()
    resolved = resolve_facts_frame(work, config)

    # Map back to canonical names for DB write.
    if not resolved.empty:
        resolved = resolved.rename(columns={
            "country_iso3": "iso3",
            "hazard_type": "hazard_code",
            "month": "ym",
        })

    return resolved


# ---------------------------------------------------------------------------
# Deltas
# ---------------------------------------------------------------------------


def _run_deltas(resolved: pd.DataFrame | None) -> pd.DataFrame | None:
    """Compute monthly deltas from resolved facts.

    The precedence engine output has columns like ``selected_source``,
    ``selected_as_of``.  ``make_deltas.process_group`` expects ``ym``,
    ``iso3``, ``hazard_code``, ``metric``, ``value``, ``as_of``,
    ``source_name``, ``source_url``, ``series_semantics``.

    This function maps between the two schemas.
    """
    if resolved is None or resolved.empty:
        return None

    from resolver.tools.make_deltas import process_group

    # Map precedence-engine output columns → make_deltas input columns.
    work = resolved.copy()
    if "selected_as_of" in work.columns and "as_of" not in work.columns:
        work["as_of"] = work["selected_as_of"]
    if "selected_source" in work.columns and "source_name" not in work.columns:
        work["source_name"] = work["selected_source"]
    for col in ("source_url", "series_semantics"):
        if col not in work.columns:
            work[col] = ""

    # process_group needs sorted ym.
    if "ym" not in work.columns:
        LOG.warning("Resolved frame has no 'ym' column — cannot compute deltas")
        return None

    work = work.sort_values("ym")

    groups = work.groupby(["iso3", "hazard_code", "metric"], dropna=False)
    delta_parts: list[pd.DataFrame] = []
    for _, group in groups:
        try:
            part = process_group(group)
            if part:
                delta_parts.append(pd.DataFrame(part))
        except SystemExit:
            # process_group raises SystemExit on NaN; skip that group.
            LOG.warning("Delta computation failed for a group; skipping")

    if not delta_parts:
        return None

    deltas = pd.concat(delta_parts, ignore_index=True)
    return deltas


# ---------------------------------------------------------------------------
# DuckDB write
# ---------------------------------------------------------------------------


def _write_to_db(
    db_url: str,
    resolved: pd.DataFrame | None,
    deltas: pd.DataFrame | None,
) -> None:
    """Write resolved facts and deltas to DuckDB."""
    from resolver.db.duckdb_io import get_db, init_schema, write_facts_tables

    conn = get_db(db_url)
    init_schema(conn)

    results = write_facts_tables(
        conn,
        facts_resolved=resolved,
        facts_deltas=deltas,
    )
    for table, upsert_result in results.items():
        LOG.info(
            "DuckDB %s: %d rows written (%d before → %d after)",
            table,
            upsert_result.rows_written,
            upsert_result.rows_before,
            upsert_result.rows_after,
        )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _log_summary(result: PipelineResult) -> None:
    LOG.info("--- Pipeline Summary ---")
    for cr in result.connector_results:
        status = f"{cr.rows} rows" if cr.status == "ok" else cr.status
        if cr.error:
            status += f" ({cr.error})"
        LOG.info("  %-20s %s", cr.name, status)
    LOG.info("  Total facts:    %d", result.total_facts)
    LOG.info("  Resolved rows:  %d", result.resolved_rows)
    LOG.info("  Delta rows:     %d", result.delta_rows)
    LOG.info("  DB written:     %s", result.db_written)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Resolver pipeline orchestrator")
    parser.add_argument(
        "--connectors",
        nargs="*",
        default=None,
        help="Run only these connectors (default: all registered)",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("RESOLVER_DB_URL", ""),
        help="DuckDB URL or path (default: $RESOLVER_DB_URL)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and resolve but do not write to DuckDB",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("RESOLVER_LOG_LEVEL", "INFO"),
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    result = run_pipeline(
        db_url=args.db or None,
        connectors=args.connectors,
        dry_run=args.dry_run,
    )

    if any(cr.status == "error" for cr in result.connector_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
