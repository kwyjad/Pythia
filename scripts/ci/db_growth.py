# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""Disk room for the canonical DB, and how much each run adds to it.

The canonical DB was about 18 GB on 28 September 2026 and 30.6 GB on
7 October, 40% of it ``haz_revisions``. A runner that runs out of disk
mid-upload must not be how the next jump is discovered, so every workflow
that downloads the DB checks the room it needs straight after the download,
and every canonical upload records how many bytes the run added.

Subcommands (all print to stdout and, under Actions, to the step summary):

``downloaded --db PATH [--factor F]``
    After a download: file size and free disk. Fails (exit 2, ``::error``)
    when free space is below ``size x factor + 2 GiB``: the working room a
    writer needs (DuckDB rewrites blocks and keeps a WAL; a compaction
    copies the whole file). Stdlib only, so it runs before any pip install.
    Records the size in ``$GITHUB_ENV`` for the before-upload record.

``before-upload --db PATH``
    Before a canonical upload: file size, free disk and the bytes this run
    added since its download; appends a row to ``db_growth_log`` in the DB
    itself, so the history travels with the canonical artifact. Never fails.

``report --db PATH``
    Growth per workflow from ``db_growth_log``, and ``haz_revisions`` rows
    and provenance bytes per day (the per-run growth of the table before
    the October 2026 dedupe, measured from the rows themselves).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from typing import Any, Optional

GIB = 1024 ** 3
HEADROOM_BYTES = 2 * GIB
GROWTH_LOG_DDL = """
CREATE TABLE IF NOT EXISTS db_growth_log (
    recorded_at TIMESTAMP,
    workflow TEXT,
    run_id TEXT,
    bytes_at_download BIGINT,
    bytes_before_upload BIGINT,
    bytes_added BIGINT,
    free_bytes BIGINT
)
"""


def _db_path(db: str) -> str:
    raw = (db or "").strip()
    return raw[len("duckdb:///"):] if raw.startswith("duckdb:///") else raw


def _gb(n: Optional[int]) -> str:
    return "n/a" if n is None else f"{n / GIB:.2f} GB"


def _summary(lines: list[str]) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n\n")
    except OSError:
        pass


def _github_env(key: str, value: str) -> None:
    path = os.environ.get("GITHUB_ENV")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{key}={value}\n")
    except OSError:
        pass


def free_bytes(path: str) -> int:
    target = path if os.path.isdir(path) else (os.path.dirname(os.path.abspath(path)) or ".")
    return int(shutil.disk_usage(target).free)


def room_needed(db_bytes: int, factor: float) -> int:
    return int(db_bytes * max(factor, 0.0)) + HEADROOM_BYTES


def check_room(db_bytes: int, free: int, factor: float) -> tuple[bool, str]:
    need = room_needed(db_bytes, factor)
    ok = free >= need
    msg = (
        f"canonical DB {_gb(db_bytes)}; free disk {_gb(free)}; working room needed "
        f"{_gb(need)} (DB x {factor:g} + 2 GB headroom)"
    )
    return ok, msg


def cmd_downloaded(args: argparse.Namespace) -> int:
    path = _db_path(args.db)
    if not os.path.exists(path):
        print(f"[db_growth] no DB at {path}; nothing to check")
        return 0
    size = os.path.getsize(path)
    free = free_bytes(path)
    ok, msg = check_room(size, free, args.factor)
    print(f"[db_growth] {msg}")
    _github_env("CANONICAL_DB_BYTES_AT_DOWNLOAD", str(size))
    _summary([
        "#### Canonical DB disk check",
        f"- {msg}",
    ])
    if not ok:
        print(
            "::error title=Not enough disk for the canonical DB::"
            f"{msg}. The job stops here rather than die of a full disk mid-write or "
            "mid-upload. Compact the canonical DB (Compact Resolver DB, apply=true) "
            "or free runner space before re-running."
        )
        return 2
    return 0


def cmd_before_upload(args: argparse.Namespace) -> int:
    path = _db_path(args.db)
    if not os.path.exists(path):
        print(f"[db_growth] no DB at {path}; nothing to record")
        return 0
    size = os.path.getsize(path)
    free = free_bytes(path)
    raw = os.environ.get("CANONICAL_DB_BYTES_AT_DOWNLOAD") or ""
    at_download = int(raw) if raw.isdigit() else None
    added = (size - at_download) if at_download is not None else None
    workflow = os.environ.get("GITHUB_WORKFLOW", args.workflow or "")
    run_id = os.environ.get("GITHUB_RUN_ID", args.run_id or "")
    line = (
        f"canonical DB {_gb(size)} before upload; {_gb(at_download)} at download; "
        f"this run added {_gb(added) if added is not None else 'n/a'} "
        f"({added if added is not None else 'n/a'} bytes); free disk {_gb(free)}"
    )
    print(f"[db_growth] {line}")
    _summary(["#### Canonical DB growth", f"- {line}"])
    try:
        import duckdb  # noqa: PLC0415

        con = duckdb.connect(path)
        try:
            con.execute(GROWTH_LOG_DDL)
            con.execute(
                "INSERT INTO db_growth_log VALUES (?, ?, ?, ?, ?, ?, ?)",
                [datetime.now(timezone.utc).replace(tzinfo=None), workflow, run_id,
                 at_download, size, added, free],
            )
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001 - a record must never stop an upload
        print(f"[db_growth] growth not recorded in the DB: {exc}")
    return 0


def growth_report(con: Any, days: int = 45) -> dict[str, Any]:
    out: dict[str, Any] = {"by_workflow": [], "revisions_by_day": []}

    def _exists(table: str) -> bool:
        return bool(con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]
        ).fetchone()[0])

    if _exists("db_growth_log"):
        rows = con.execute(
            """
            SELECT workflow, COUNT(*), AVG(bytes_added), MAX(bytes_added), MAX(recorded_at)
            FROM db_growth_log WHERE bytes_added IS NOT NULL
            GROUP BY workflow ORDER BY 3 DESC NULLS LAST
            """
        ).fetchall()
        out["by_workflow"] = [
            {"workflow": r[0], "runs": int(r[1]), "mean_bytes_added": float(r[2] or 0),
             "max_bytes_added": int(r[3] or 0), "last": str(r[4])}
            for r in rows
        ]
    if _exists("haz_revisions"):
        rows = con.execute(
            f"""
            SELECT CAST(observed_at AS DATE) AS d, COUNT(*),
                   SUM(LENGTH(COALESCE(detail_json, '')))
            FROM haz_revisions
            WHERE observed_at >= CURRENT_DATE - INTERVAL {int(days)} DAY
            GROUP BY 1 ORDER BY 1
            """
        ).fetchall()
        out["revisions_by_day"] = [
            {"day": str(r[0]), "rows": int(r[1]), "detail_bytes": int(r[2] or 0)} for r in rows
        ]
        total = con.execute(
            "SELECT COUNT(*), SUM(LENGTH(COALESCE(detail_json, ''))) FROM haz_revisions"
        ).fetchone()
        out["revisions_total"] = {"rows": int(total[0]), "detail_bytes": int(total[1] or 0)}
    return out


def cmd_report(args: argparse.Namespace) -> int:
    try:
        import duckdb  # noqa: PLC0415

        con = duckdb.connect(_db_path(args.db), read_only=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[db_growth] cannot open DB: {exc}")
        return 0
    try:
        report = growth_report(con, args.days)
    finally:
        con.close()
    lines = ["#### Canonical DB growth report"]
    for r in report["by_workflow"]:
        lines.append(
            f"- {r['workflow']}: {r['runs']} run(s), mean {_gb(int(r['mean_bytes_added']))} added, "
            f"max {_gb(r['max_bytes_added'])}"
        )
    if not report["by_workflow"]:
        lines.append("- no growth records yet (db_growth_log is written before each canonical upload)")
    tot = report.get("revisions_total")
    if tot:
        lines.append(f"- haz_revisions: {tot['rows']:,} rows, {_gb(tot['detail_bytes'])} of provenance")
        for r in report["revisions_by_day"]:
            lines.append(f"  - {r['day']}: {r['rows']:,} rows, {_gb(r['detail_bytes'])}")
    print("\n".join(lines))
    _summary(lines)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("downloaded")
    d.add_argument("--db", default="data/resolver.duckdb")
    d.add_argument("--factor", type=float, default=float(os.environ.get("CANONICAL_DB_ROOM_FACTOR", "1.0")))
    d.set_defaults(fn=cmd_downloaded)
    b = sub.add_parser("before-upload")
    b.add_argument("--db", default="data/resolver.duckdb")
    b.add_argument("--workflow", default="")
    b.add_argument("--run-id", default="")
    b.set_defaults(fn=cmd_before_upload)
    r = sub.add_parser("report")
    r.add_argument("--db", default="data/resolver.duckdb")
    r.add_argument("--days", type=int, default=45)
    r.add_argument("--json-out", default="")
    r.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
