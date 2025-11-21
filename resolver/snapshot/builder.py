"""DB-backed snapshot builder (skeleton).

This module provides a placeholder entry point for constructing monthly
snapshots directly from DuckDB without relying on connector staging
outputs. The implementation intentionally focuses on deterministic,
test-friendly messaging until the DB-first snapshot logic is
implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SnapshotPlan:
    """Structured placeholder describing the planned snapshot build."""

    month: Optional[str]
    db_url: Optional[str]
    dry_run: bool
    notes: List[str]


def plan_snapshot_from_db(
    *, month: str | None, db_url: str | None, dry_run: bool = False
) -> SnapshotPlan:
    """Return a deterministic placeholder plan for DB-based snapshots.

    The real snapshot builder will operate on DuckDB canonical tables to
    materialise monthly bundles. For now, we return a stable structure so
    callers and tests have a predictable contract.
    """

    notes: list[str] = [
        "Snapshot-from-DB builder is not implemented yet.",
        "This is a stub to decouple ingestion from freeze_snapshot.",
        "Implement DuckDB-driven snapshot assembly here.",
    ]
    return SnapshotPlan(month=month, db_url=db_url, dry_run=dry_run, notes=notes)


def print_stub_plan(plan: SnapshotPlan) -> None:
    """Emit a consistent, human-friendly stub description."""

    print("### Snapshot-from-DB (stub)")
    print(f"- target_month: {plan.month or 'not specified'}")
    print(f"- db_url: {plan.db_url or 'duckdb:///resolver/db/resolver.duckdb'}")
    print(f"- dry_run: {plan.dry_run}")
    print("- status: pending implementation")
    print("- notes:")
    for note in plan.notes:
        print(f"  - {note}")


def run_stub_snapshot(*, month: str | None, db_url: str | None, dry_run: bool) -> int:
    """Generate and print a stub snapshot plan.

    Returns zero to align with the final builder's success contract while
    making it clear that no work has been performed yet.
    """

    plan = plan_snapshot_from_db(month=month, db_url=db_url, dry_run=dry_run)
    print_stub_plan(plan)
    return 0

