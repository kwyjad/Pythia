# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Build the DB copy that gets published to the release.

Why this exists: GitHub caps a release asset at 2 GiB. On 2026-08-06 the
canonical DB had grown to 3.0 GB and ``gh release upload --clobber`` DELETED
the working 453 MB asset before the oversized replacement was rejected,
leaving the release with no database at all. Two defects, one incident: the
published copy carried data nothing serves, and the upload was destructive
before it was validated.

What the release is FOR is the API and dashboard. Those read facts, forecasts,
scores, questions and interpretations — they never read one ``haz_*`` table
(the PA resolution machine's own raw caches and working tables). Those live in
the canonical ARTIFACT, which the machine resumes from and which this script
never touches. Dropping them from the published copy is what makes the asset
serveable again; compaction is a free bonus on top.

On 2026-09-01 it happened again and the narrow prefix was why: this stripped
only ``haz_raw_``, so the machine's WORKING tables still shipped — and
``haz_triggers`` alone carries one ``evidence_of_absence_json`` per assessed
cell across ~200 countries x 3 hazards x ~360 months. The publish aborted at
5.1 GB. ``grep haz_ pythia/api/ web/src/`` returns nothing, and the release
asset's only consumer is ``pythia/api/db_sync.py`` (every pipeline job takes
the canonical artifact via ``.github/actions/download-canonical-db``), so the
prefix is now the whole ``haz_`` family.

It also reports per-table sizes on the way through. The 2026-09-01 abort could
not be attributed to any table because nothing measured one, which meant the
only way to choose what to cut was to guess. Now the same log block that says
"over the limit" says what is over it.

    python -m scripts.ci.build_release_db \
        --src data/resolver.duckdb --out data_out/resolver.duckdb \
        [--max-bytes 2147483648] [--keep-all]

Exit codes: 0 on success, 1 when the result still exceeds --max-bytes, 2 when
credential-shaped text survives the scrub, 3 when the file is in a storage
format the API's DuckDB 1.x cannot read. Any failure stops the publish
workflow BEFORE it clobbers a working asset.

Since Oct 2026 it also drops licensed and internal tables (``EXCLUDED_TABLES``)
and scrubs every text column of credential-shaped strings
(``scrub_database``), because the release is public.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# GitHub's per-asset ceiling for a release.
GITHUB_ASSET_LIMIT_BYTES = 2 * 1024 * 1024 * 1024

# Prefixes of tables excluded from the PUBLISHED copy: the PA resolution
# machine's raw source caches (documents, tracks, provider payloads) AND its
# working, ledger and verdict tables. Nothing in pythia/api/** or web/src/**
# references any of them — verified by grep, and pinned by a test so a future
# consumer cannot quietly start depending on a table this strips.
#
# Widening this is a decision about what the dashboard can ever show, so it
# stays deliberate: if a future view wants the machine's verdicts, narrow this
# back to the tables that view needs and say so here. Until then the release
# is a serving copy and the machine's state belongs in the canonical artifact.
EXCLUDED_TABLE_PREFIXES = ("haz_",)

# Whole tables excluded from the PUBLISHED copy for reasons other than size
# (Oct 2026 security audit). The canonical artifact keeps every one of them,
# so the forecaster and the PA machine are untouched; only the public asset
# loses them.
#
# - Licensed third-party rows whose terms do not allow public redistribution:
#   ACLED event-level data, EM-DAT figures, ACAPS narrative products, and the
#   raw staging copy of every connector (facts_raw). Derived monthly
#   aggregates (facts_resolved, acled_monthly_fatalities) still ship.
# - Pipeline internals nothing serves: provider Batch-API state and request
#   bodies, and the in-process run ledger.
#
# acaps_inform_severity (published index scores) stays pending the owner's
# licence check; add it here if ACAPS says otherwise.
EXCLUDED_TABLES = frozenset(
    {
        "acled_political_events",
        "emdat_pa",
        "facts_raw",
        "acaps_risk_radar",
        "acaps_daily_monitoring",
        "acaps_humanitarian_access",
        "llm_batches",
        "llm_batch_requests",
        "ui_runs",
        # Sibyl's evidence record (Oct 2026): the text of third-party pages
        # and documents its trials read, kept for audit in the canonical DB
        # and never republished. No API route reads it.
        "sibyl_evidence",
    }
)


# Every file this module CREATES is written in the storage format DuckDB 1.x
# reads (Nov 2026). DuckDB 2.0.0 (PyPI, 2026-10-21) writes a fresh database
# in storage version 999, which no 1.x release can open, and the publish
# workflow installed duckdb unpinned: the 2026-11-14 release asset was
# written by 2.0.0, and the API (``duckdb>=1.1,<2``) answered 500 on every
# route that touched the DB. An EXISTING 1.x file keeps its format when a
# 2.0 process writes to it, so only the fresh files built here and by the
# compaction copy were exposed. 'v1.4.0' gives storage version 67 under 1.x
# and 2.0 alike, readable by every DuckDB from 1.4 on. Not 'v1.0.0': that
# format lacks the newer compression, and the 2026-11-14 release came out at
# 1,218 MB in it against 262 MB in v1.4.0 (measured on the release itself).
FRESH_FILE_STORAGE_VERSION = "v1.4.0"

# The newest storage version DuckDB 1.5 can read ("we can only read versions
# between 64 and 68"). A file above it is refused before anything is
# uploaded or swapped in.
MAX_READABLE_STORAGE_VERSION = 68


def _connect():
    import duckdb

    return duckdb.connect()


def _attach_fresh(con, path: Path, alias: str) -> None:
    """Attach a NEW database file in the 1.x-readable storage format."""
    con.execute(
        f"ATTACH '{path}' AS {alias} "
        f"(STORAGE_VERSION '{FRESH_FILE_STORAGE_VERSION}')"
    )


def storage_version_of(path: str | Path) -> tuple[int, str]:
    """(storage version number, writer version string) from a file header.

    Read straight from the bytes, so it answers without a DuckDB that can
    open the file, which is exactly the case it exists to catch.
    """
    import struct

    with open(path, "rb") as fh:
        head = fh.read(64)
    if len(head) < 60 or head[8:12] != b"DUCK":
        raise ValueError(f"{path} is not a DuckDB database file")
    number = struct.unpack("<Q", head[12:20])[0]
    writer = head[52:60].split(b"\x00", 1)[0].decode("ascii", "replace")
    return int(number), writer


def tables_in(con, alias: str) -> list[str]:
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_catalog = ? AND table_schema = 'main' ORDER BY table_name",
        [alias],
    ).fetchall()
    return [str(r[0]) for r in rows]


def is_excluded(table: str) -> bool:
    name = table.lower()
    return name in EXCLUDED_TABLES or any(name.startswith(p) for p in EXCLUDED_TABLE_PREFIXES)


def _text_columns(con, alias: str) -> list[tuple[str, str]]:
    rows = con.execute(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_catalog = ? AND table_schema = 'main' "
        "AND data_type IN ('VARCHAR', 'JSON') ORDER BY table_name, column_name",
        [alias],
    ).fetchall()
    return [(str(t), str(c)) for t, c in rows]


def _count_matches(con, alias: str, table: str, column: str, pattern: str) -> int:
    row = con.execute(
        f'SELECT COUNT(*) FROM {alias}."{table}" '
        f'WHERE regexp_matches(CAST("{column}" AS VARCHAR), ?)',
        [pattern],
    ).fetchone()
    return int(row[0]) if row else 0


def scrub_database(con, alias: str) -> dict[str, object]:
    """Replace every credential-shaped string in every text column.

    A request error can quote the URL it failed on, and until Oct 2026 the
    Gemini key travelled in that URL, so error and debug columns could hold a
    live key. The writers now scrub at write time; this pass is the net under
    them, run over the copy that is about to become public.

    One existence scan per column (the combined pattern), and a rewrite only
    for columns that matched, so the large prompt and response columns cost
    one read each. Returns ``scrubbed`` (rows changed per ``table.column``)
    and ``residual`` (matches left after the rewrite, which must be empty).
    """

    from pythia.secret_scrub import COMBINED_PATTERN, SQL_PATTERNS

    scrubbed: dict[str, int] = {}
    residual: dict[str, int] = {}
    for table, column in _text_columns(con, alias):
        hits = _count_matches(con, alias, table, column, COMBINED_PATTERN)
        if not hits:
            continue
        for pattern, replacement in SQL_PATTERNS:
            con.execute(
                f'UPDATE {alias}."{table}" SET "{column}" = '
                f'regexp_replace(CAST("{column}" AS VARCHAR), ?, ?, \'g\') '
                f'WHERE regexp_matches(CAST("{column}" AS VARCHAR), ?)',
                [pattern, replacement, pattern],
            )
        scrubbed[f"{table}.{column}"] = hits
        left = _count_matches(con, alias, table, column, COMBINED_PATTERN)
        if left:
            residual[f"{table}.{column}"] = left
    return {"scrubbed": scrubbed, "residual": residual}


def compact_database(src: str, out: str) -> dict[str, int]:
    """Copy src -> out to reclaim space. The ONLY thing that shrinks a file.

    DuckDB reuses freed blocks but never truncates, so deleting rows stops a
    file growing without giving any bytes back. Reclaiming means writing a
    fresh database, which is what ``COPY FROM DATABASE`` does here.

    Shared with the retention workflow rather than reimplemented there: a
    second copy of this would eventually diverge from the release path, and
    this is the operation with the ~40 GB disk peak if it is run in the wrong
    order (compact the ROWS first, then the file).
    """
    src_path, out_path = Path(src), Path(out)
    if not src_path.exists():
        raise FileNotFoundError(f"source DB not found: {src_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)
    Path(str(out_path) + ".wal").unlink(missing_ok=True)

    src_bytes = src_path.stat().st_size
    con = _connect()
    try:
        con.execute(f"ATTACH '{src_path}' AS csrc (READ_ONLY)")
        _attach_fresh(con, out_path, "cdst")
        con.execute("COPY FROM DATABASE csrc TO cdst")
        con.execute("CHECKPOINT cdst")
    finally:
        con.close()
    out_bytes = out_path.stat().st_size
    storage_version, writer = storage_version_of(out_path)
    if storage_version > MAX_READABLE_STORAGE_VERSION:
        raise RuntimeError(
            f"compacted DB has storage version {storage_version} (written by "
            f"DuckDB {writer}); DuckDB 1.x cannot read it, so it must not "
            "replace the canonical DB"
        )
    return {
        "src_bytes": src_bytes,
        "out_bytes": out_bytes,
        "saved_bytes": src_bytes - out_bytes,
    }


def _size_lines(con, alias: str, top: int) -> list[str]:
    """Per-table sizes, or a note saying why there are none.

    Never fatal: this is a diagnostic wrapped around the one job that must not
    fail for a diagnostic's sake. A publish that stops because the size REPORT
    broke would be a worse outcome than a publish with no size report.
    """
    if top <= 0:
        return []
    try:
        from scripts.ci.db_table_sizes import report_lines, size_report, whole_file

        return report_lines(
            size_report(con, catalog=alias, top=top), whole_file(con, alias)
        )
    except Exception as exc:  # pragma: no cover - diagnostics must never block
        return [f"(size report unavailable: {exc})"]


def build_release_db(
    src: str, out: str, *, keep_all: bool = False, report_top: int = 15
) -> dict[str, object]:
    """Copy src -> out, dropping excluded tables and compacting.

    Two passes on purpose: ``COPY FROM DATABASE`` reproduces the schema
    faithfully, but DROPping afterwards leaves the freed blocks in the file —
    so the second copy is what actually reclaims them.
    """
    src_path, out_path = Path(src), Path(out)
    if not src_path.exists():
        raise FileNotFoundError(f"source DB not found: {src_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_path = out_path.with_suffix(out_path.suffix + ".work")
    for stale in (out_path, work_path):
        stale.unlink(missing_ok=True)
        Path(str(stale) + ".wal").unlink(missing_ok=True)

    src_bytes = src_path.stat().st_size
    dropped: list[str] = []
    scrub: dict[str, object] = {"scrubbed": {}, "residual": {}}
    before: list[str] = []
    after: list[str] = []
    canonical_counts: dict[str, int] = {}
    canonical_table_count = 0

    con = _connect()
    try:
        con.execute(f"ATTACH '{src_path}' AS src (READ_ONLY)")
        _attach_fresh(con, work_path, "work")
        con.execute("COPY FROM DATABASE src TO work")
        # Blocks are not allocated until the write is flushed, so an
        # un-checkpointed database measures as zero bytes.
        con.execute("CHECKPOINT work")
        before = _size_lines(con, "work", report_top)
        all_tables = tables_in(con, "work")
        canonical_table_count = len(all_tables)
        if not keep_all:
            # Row counts of what the release drops, so a report on the release
            # copy can say "canonical: N rows" rather than "no table".
            for table in all_tables:
                if is_excluded(table):
                    try:
                        canonical_counts[table] = int(con.execute(
                            f'SELECT COUNT(*) FROM work."{table}"').fetchone()[0])
                    except Exception:
                        pass
        if not keep_all:
            for table in tables_in(con, "work"):
                if is_excluded(table):
                    con.execute(f"DROP TABLE IF EXISTS work.{table}")
                    dropped.append(table)
        # The scrub runs whatever --keep-all says: keeping a table is a
        # decision about size, never about publishing a credential.
        scrub = scrub_database(con, "work")
        con.execute("CHECKPOINT work")
        con.execute("DETACH src")
        _attach_fresh(con, out_path, "dst")
        con.execute("COPY FROM DATABASE work TO dst")
        con.execute("CHECKPOINT dst")
        after = _size_lines(con, "dst", report_top)
    finally:
        con.close()
    work_path.unlink(missing_ok=True)
    Path(str(work_path) + ".wal").unlink(missing_ok=True)

    out_bytes = out_path.stat().st_size
    storage_version, writer = storage_version_of(out_path)
    return {
        "src_bytes": src_bytes,
        "out_bytes": out_bytes,
        "storage_version": storage_version,
        "writer_version": writer,
        "dropped_tables": dropped,
        "canonical_counts": canonical_counts,
        "canonical_table_count": canonical_table_count,
        "scrubbed": scrub["scrubbed"],
        "residual": scrub["residual"],
        "saved_bytes": src_bytes - out_bytes,
        "sizes_before": before,
        "sizes_after": after,
    }


def _mb(n: float) -> str:
    return f"{n / (1024 * 1024):.1f} MB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-bytes", type=int, default=GITHUB_ASSET_LIMIT_BYTES)
    parser.add_argument("--keep-all", action="store_true",
                        help="Compact only; keep every table (diagnostics)")
    parser.add_argument("--report-top", type=int, default=15,
                        help="Per-table sizes to log before/after (0 disables)")
    parser.add_argument("--report-json", default=None,
                        help="Write the dropped tables, their canonical row counts "
                             "and the canonical table count here (read by "
                             "inspect_resolver_db --release-build-json)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[build_release_db] %(message)s")
    stats = build_release_db(
        args.src, args.out, keep_all=args.keep_all, report_top=args.report_top
    )
    if args.report_json:
        try:
            import json

            Path(args.report_json).write_text(json.dumps({
                k: stats.get(k) for k in (
                    "src_bytes", "out_bytes", "dropped_tables",
                    "canonical_counts", "canonical_table_count",
                )
            }, indent=2))
        except Exception as exc:  # a report never costs the publish
            LOGGER.warning("could not write --report-json: %s", exc)
    LOGGER.info(
        "source %s -> published %s (saved %s)",
        _mb(stats["src_bytes"]), _mb(stats["out_bytes"]), _mb(stats["saved_bytes"]),
    )
    if stats["dropped_tables"]:
        LOGGER.info(
            "dropped %d table(s) the release does not publish (machine caches, "
            "licensed sources, pipeline internals): %s",
            len(stats["dropped_tables"]), ", ".join(stats["dropped_tables"]),
        )
    if stats["scrubbed"]:
        # A non-zero count means a credential reached the canonical DB and
        # would have been published: rotate it, whatever else this run does.
        for col, n in sorted(stats["scrubbed"].items()):
            LOGGER.warning("scrubbed credential-shaped text from %d row(s) of %s", n, col)
        print("::warning title=Credentials scrubbed from release::"
              f"{sum(stats['scrubbed'].values())} row(s) across "
              f"{len(stats['scrubbed'])} column(s) carried credential-shaped text; "
              "the published copy is clean, but rotate the affected key.")
    else:
        LOGGER.info("credential scan: no credential-shaped text found")
    if stats["residual"]:
        LOGGER.error("credential-shaped text survived the scrub: %s",
                     ", ".join(sorted(stats["residual"])))
        print("::error::Release DB still carries credential-shaped text after "
              "the scrub; publish aborted before clobbering the existing asset.")
        return 2
    # Log the attribution in the SAME block as any size error, so whoever
    # reads the failure also reads what caused it.
    for label, key in (("canonical", "sizes_before"), ("published", "sizes_after")):
        for line in stats.get(key) or []:
            LOGGER.info("%s | %s", label, line)
    storage_version = int(stats["storage_version"])
    LOGGER.info("storage version %d (written by DuckDB %s)",
                storage_version, stats["writer_version"])
    if storage_version > MAX_READABLE_STORAGE_VERSION:
        # The 2026-11-14 release was storage version 999 and the API could
        # not open it. Refuse before the clobber, like the other guards.
        print(f"::error::Release DB has storage version {storage_version} "
              f"(written by DuckDB {stats['writer_version']}); the API's "
              "DuckDB 1.x cannot open it. Publish aborted before clobbering "
              "the existing asset.")
        return 3
    out_bytes = int(stats["out_bytes"])
    if out_bytes > args.max_bytes:
        # Loud and non-destructive: the caller must NOT proceed to a clobbering
        # upload. This is the guard whose absence cost the release its asset.
        LOGGER.error(
            "published DB is %s, over the %s release-asset limit — refusing to "
            "publish rather than delete the working asset and fail the upload",
            _mb(out_bytes), _mb(args.max_bytes),
        )
        print(f"::error::Release DB {_mb(out_bytes)} exceeds the "
              f"{_mb(args.max_bytes)} GitHub asset limit; publish aborted "
              "before clobbering the existing asset.")
        return 1
    print(f"RELEASE_DB_BYTES={out_bytes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
