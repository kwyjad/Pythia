# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""What a zero was checked against, stored once (Oct 2026).

A flood zero rests on GDACS listing no qualifying event for the country in
the month. Until October 2026 the zero's provenance proved that by carrying
the URL of EVERY GDACS flood event in the cache, twice: once under
``evidence_of_absence.gdacs.source_urls`` (from ``raw_store_summary``) and
again in the top-level ``source_urls`` that ``resolutions._collect_urls``
builds from it. On the canonical DB of 8 Oct 2026 that was 20,048 rows and
4.86 GB of a 9.2 GB file, with lists of up to 5,099 URLs, and the list grows
with the cache, so every month's new rows were larger than the last.

The list is the same in every zero written against the same cache. So:

* a zero now cites the GDACS events FOR ITS COUNTRY AND WINDOW (usually
  none: a zero means none qualified, and most months list nothing at all
  for the country), plus a snapshot identifier for the cache it was checked
  in — ``<source>:<hazard>:<date>:<count>:<sha256 prefix>`` — and
* the full sorted URL list of each snapshot is stored ONCE, in
  ``haz_evidence_snapshots``, keyed by that identifier.

Nothing a zero could cite before becomes unrecoverable: the old list is
exactly the snapshot's list, and :func:`snapshot_urls` returns it.

:func:`rewrite_zero_rows` brings the rows already written into the same
shape, in committed batches (a single transaction over 5 GB of JSON is what
ran DuckDB out of memory on ``haz_revisions``), each row updated only if its
provenance still hashes to what was read (compare-and-swap), and refusing a
row whose old URLs would not all be recoverable. :func:`integrity_report`
is what the workflow compares before and after: row counts by hazard and
status, a checksum over every column but ``provenance_json``, and sizes.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
from typing import TYPE_CHECKING, Any, Iterable

from resolver.hazard_resolution.schema import ensure_haz_schema

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)

TABLE = "haz_evidence_snapshots"

#: The detector blocks whose ``source_urls`` is a whole-cache listing. GDACS
#: is the only one that grows without bound (IBTrACS cites one or two file
#: URLs; drought cites the analyses and indicator feeds actually read).
CACHE_LISTING_DETECTORS = ("gdacs",)

#: Rows per committed batch in the rewrite. A flood zero averaged 254 KB of
#: provenance on 8 Oct 2026, so a batch reads ~130 MB of JSON.
DEFAULT_BATCH_ROWS = 500


def urls_digest(urls: Iterable[Any]) -> tuple[list[str], str]:
    """The sorted distinct URL list and the SHA-256 of its JSON form."""

    ordered = sorted({str(u) for u in urls if u})
    digest = hashlib.sha256(
        json.dumps(ordered, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ordered, digest


def make_snapshot_id(
    source: str, hazard: str | None, snapshot_date: str | None, n_urls: int, sha256: str
) -> str:
    """``gdacs:FL:2026-10-08:5102:1a2b3c4d5e6f7a8b`` — readable and unique.

    The date and the count are there for the reader; the hash is what makes
    the identifier unique, so two snapshots with the same date and count but
    different contents never share a row.
    """

    return f"{source}:{hazard or '*'}:{snapshot_date or 'undated'}:{int(n_urls)}:{sha256[:16]}"


def _date_part(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:10] if len(text) >= 10 else (text or None)


def store_snapshot(
    con: "duckdb.DuckDBPyConnection",
    *,
    source: str,
    hazard: str | None,
    urls: Iterable[Any],
    snapshot_date: Any,
) -> dict[str, Any]:
    """Store one snapshot's URL list (once) and return the citation a row carries."""

    ensure_haz_schema(con)
    ordered, digest = urls_digest(urls)
    date = _date_part(snapshot_date)
    snapshot_id = make_snapshot_id(source, hazard, date, len(ordered), digest)
    con.execute(
        f"""
        INSERT OR IGNORE INTO {TABLE}
            (snapshot_id, source, hazard, snapshot_date, n_urls, urls_sha256, urls_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [snapshot_id, source, hazard, date, len(ordered), digest,
         json.dumps(ordered, separators=(",", ":"))],
    )
    return {
        "snapshot_id": snapshot_id,
        "snapshot_date": date,
        "n_urls": len(ordered),
        "urls_sha256": digest,
        "table": TABLE,
    }


def snapshot_urls(con: "duckdb.DuckDBPyConnection", snapshot_id: str) -> list[str] | None:
    """The full URL list a snapshot stands for, or None when it is not stored."""

    ensure_haz_schema(con)
    row = con.execute(
        f"SELECT urls_json, urls_sha256 FROM {TABLE} WHERE snapshot_id = ?", [snapshot_id]
    ).fetchone()
    if row is None:
        return None
    urls = json.loads(row[0])
    _, digest = urls_digest(urls)
    if digest != row[1]:
        raise ValueError(f"snapshot {snapshot_id} does not match its own hash")
    return urls


def summary_with_snapshot(
    con: "duckdb.DuckDBPyConnection", source: str, hazard: str | None, *, store: bool = True
) -> dict[str, Any]:
    """``raw_store_summary`` with the URL list replaced by a snapshot citation.

    ``store=False`` (a dry run) computes the citation and writes nothing.
    """

    from resolver.hazard_resolution.sources import raw_store_summary

    summary = raw_store_summary(con, source, hazard)
    urls = summary.pop("source_urls", []) or []
    if store:
        summary["snapshot"] = store_snapshot(
            con, source=source, hazard=hazard, urls=urls,
            snapshot_date=summary.get("last_retrieved_at"),
        )
    else:
        ordered, digest = urls_digest(urls)
        date = _date_part(summary.get("last_retrieved_at"))
        summary["snapshot"] = {
            "snapshot_id": make_snapshot_id(source, hazard, date, len(ordered), digest),
            "snapshot_date": date, "n_urls": len(ordered), "urls_sha256": digest,
            "table": TABLE,
        }
    return summary


def events_in_window(detail: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The detector's own list of this country's events in this month.

    For a zero the qualifying list is empty by definition; the events below
    the trigger level are what the zero was weighed against.
    """

    detail = detail or {}
    out: list[dict[str, Any]] = []
    for key in ("events_qualifying", "events_below_threshold"):
        for event in detail.get(key) or []:
            if isinstance(event, dict):
                out.append(dict(event))
    return out


# --------------------------------------------------------------------------
# Rewrite of the rows already written
# --------------------------------------------------------------------------


def _window_index(con: "duckdb.DuckDBPyConnection", hazard: str) -> dict[tuple[str, str], list[dict]]:
    """(iso3, ym) -> the cached events naming that country and overlapping it."""

    from resolver.hazard_resolution.sources import load_raw_records

    index: dict[tuple[str, str], list[dict]] = {}
    for event in load_raw_records(con, "gdacs", hazard=hazard):
        summary = {
            "event_id": event.get("event_id"),
            "alert_level": event.get("alert_level"),
            "exposed_population": event.get("exposed_population"),
            "start_date": event.get("start_date"),
            "end_date": event.get("end_date"),
            "source_url": event.get("_source_url"),
        }
        for iso3 in event.get("iso3_list") or []:
            for ym in event.get("months_overlapped") or []:
                index.setdefault((str(iso3).upper(), str(ym)), []).append(summary)
    return index


def slim_provenance(
    con: "duckdb.DuckDBPyConnection",
    provenance: dict[str, Any],
    *,
    iso3: str,
    ym: str,
    hazard: str,
    window_index: dict[tuple[str, str], list[dict]] | None,
) -> tuple[dict[str, Any], bool]:
    """Replace a whole-cache listing with a snapshot citation. Pure apart from the store.

    Returns ``(provenance, changed)``. Raises ``ValueError`` when the old
    top-level URLs would not all be recoverable from the new row plus its
    snapshots — the one thing this rewrite must never do.
    """

    from resolver.hazard_resolution.resolutions import _collect_urls

    evidence = provenance.get("evidence_of_absence")
    if not isinstance(evidence, dict):
        return provenance, False
    changed = False
    recoverable: set[str] = set()
    for detector in CACHE_LISTING_DETECTORS:
        block = evidence.get(detector)
        if not isinstance(block, dict) or not isinstance(block.get("source_urls"), list):
            continue
        old_urls = block.pop("source_urls")
        citation = store_snapshot(
            con, source=detector, hazard=hazard, urls=old_urls,
            snapshot_date=block.get("last_retrieved_at"),
        )
        block["snapshot"] = citation
        listed = set(str(u) for u in old_urls if u)
        recoverable |= listed
        if "events_in_window" not in block:
            candidates = (window_index or {}).get((iso3.upper(), ym), [])
            # Only events the zero's own listing named: the cache may have
            # gained events since, and a zero must not cite what it never saw.
            block["events_in_window"] = [
                dict(e) for e in candidates if str(e.get("source_url")) in listed
            ]
            block["events_in_window_basis"] = "reconstructed_at_rewrite"
        changed = True
    if not changed:
        return provenance, False

    old_top = [str(u) for u in provenance.get("source_urls") or [] if u]
    new_top = _collect_urls(evidence)
    provenance["source_urls"] = new_top
    missing = set(old_top) - set(new_top) - recoverable
    if missing:
        raise ValueError(
            f"{len(missing)} URL(s) cited by {iso3}/{hazard}/{ym} would not be "
            f"recoverable, e.g. {sorted(missing)[:2]}"
        )
    return provenance, True


def _target_keys(con: "duckdb.DuckDBPyConnection") -> list[tuple]:
    clauses = " OR ".join(
        f"json_type(provenance_json, '$.evidence_of_absence.{d}.source_urls') = 'ARRAY'"
        for d in CACHE_LISTING_DETECTORS
    )
    return con.execute(
        f"""
        SELECT iso3, year, month, hazard FROM haz_resolutions
        WHERE {clauses}
        ORDER BY hazard, iso3, year, month
        """
    ).fetchall()


def rewrite_zero_rows(
    con: "duckdb.DuckDBPyConnection",
    *,
    apply: bool = False,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    max_seconds: float | None = None,
) -> dict[str, Any]:
    """Rewrite every row carrying a whole-cache listing; committed batches, CAS per row.

    Dry run by default: counts the rows and the bytes they would lose and
    writes nothing (no snapshot rows either). Idempotent: a rewritten row no
    longer matches the selection. ``max_seconds`` stops between batches and
    leaves the rest for the next run, which picks them up by the same query.
    """

    import time

    ensure_haz_schema(con)
    started = time.monotonic()
    keys = _target_keys(con)
    out: dict[str, Any] = {
        "applied": bool(apply),
        "rows_targeted": len(keys),
        "rows_rewritten": 0,
        "rows_cas_skipped": 0,
        "rows_refused": 0,
        "bytes_before": 0,
        "bytes_after": 0,
        "batches": 0,
        "stopped_early": False,
        "refusals": [],
    }
    if not keys:
        return out

    indexes: dict[str, dict] = {}
    for start in range(0, len(keys), max(1, int(batch_rows))):
        if max_seconds is not None and time.monotonic() - started > max_seconds:
            out["stopped_early"] = True
            break
        batch = keys[start:start + batch_rows]
        con.execute("CREATE OR REPLACE TEMP TABLE _snap_keys (iso3 TEXT, year INTEGER, month INTEGER, hazard TEXT)")
        con.executemany("INSERT INTO _snap_keys VALUES (?, ?, ?, ?)", [list(k) for k in batch])
        rows = con.execute(
            """
            SELECT r.iso3, r.year, r.month, r.hazard, r.provenance_json
            FROM haz_resolutions r JOIN _snap_keys k USING (iso3, year, month, hazard)
            """
        ).fetchall()
        updates: list[list[Any]] = []
        if apply:
            con.execute("BEGIN TRANSACTION")
        try:
            for iso3, year, month, hazard, prov_json in rows:
                ym = f"{int(year):04d}-{int(month):02d}"
                if hazard not in indexes:
                    indexes[hazard] = _window_index(con, hazard) if apply else {}
                try:
                    prov = json.loads(prov_json)
                    if not apply:
                        # Measure without writing snapshot rows.
                        trial = json.loads(prov_json)
                        for d in CACHE_LISTING_DETECTORS:
                            block = (trial.get("evidence_of_absence") or {}).get(d)
                            if isinstance(block, dict):
                                block.pop("source_urls", None)
                        trial["source_urls"] = []
                        out["bytes_before"] += len(prov_json)
                        out["bytes_after"] += len(json.dumps(trial, default=str))
                        continue
                    new_prov, changed = slim_provenance(
                        con, prov, iso3=iso3, ym=ym, hazard=hazard,
                        window_index=indexes[hazard],
                    )
                except ValueError as exc:
                    out["rows_refused"] += 1
                    if len(out["refusals"]) < 20:
                        out["refusals"].append(str(exc))
                    continue
                if not changed:
                    continue
                new_json = json.dumps(new_prov, default=str)
                out["bytes_before"] += len(prov_json)
                out["bytes_after"] += len(new_json)
                old_md5 = hashlib.md5(prov_json.encode("utf-8")).hexdigest()
                updates.append([new_json, iso3, year, month, hazard, old_md5])
            if apply and updates:
                before = con.execute("SELECT COUNT(*) FROM haz_resolutions").fetchone()[0]
                for params in updates:
                    cur = con.execute(
                        """
                        UPDATE haz_resolutions SET provenance_json = ?
                        WHERE iso3 = ? AND year = ? AND month = ? AND hazard = ?
                          AND md5(provenance_json) = ?
                        """,
                        params,
                    )
                    n = cur.fetchone()[0] if cur.description else 0
                    if n == 1:
                        out["rows_rewritten"] += 1
                    else:
                        out["rows_cas_skipped"] += 1
                after = con.execute("SELECT COUNT(*) FROM haz_resolutions").fetchone()[0]
                if after != before:
                    raise RuntimeError(f"row count moved {before} -> {after} inside a batch")
            if apply:
                con.execute("COMMIT")
        except Exception:
            if apply:
                con.execute("ROLLBACK")
            raise
        out["batches"] += 1
        LOG.info(
            "[snapshots] batch %d: %d rows read, %d rewritten so far",
            out["batches"], len(rows), out["rows_rewritten"],
        )
    con.execute("DROP TABLE IF EXISTS _snap_keys")
    out["duration_sec"] = round(time.monotonic() - started, 1)
    return out


# --------------------------------------------------------------------------
# Integrity report
# --------------------------------------------------------------------------


def integrity_report(con: "duckdb.DuckDBPyConnection", db_path: str | None = None) -> dict[str, Any]:
    """Counts, a checksum over every column but provenance_json, and sizes."""

    cols = [
        str(r[1])  # PRAGMA table_info: (cid, NAME, ...)
        for r in con.execute("PRAGMA table_info('haz_resolutions')").fetchall()
        if str(r[1]) != "provenance_json"
    ]
    row_expr = "concat_ws('|', " + ", ".join(
        f"coalesce(CAST(\"{c}\" AS VARCHAR), '<null>')" for c in cols
    ) + ")"
    checksum = con.execute(
        f"""
        SELECT md5(coalesce(string_agg(md5({row_expr}), ',' ORDER BY iso3, year, month, hazard), ''))
        FROM haz_resolutions
        """
    ).fetchone()[0]
    counts = [
        {"hazard": h, "status": s, "rows": int(n)}
        for h, s, n in con.execute(
            "SELECT hazard, status, COUNT(*) FROM haz_resolutions GROUP BY 1, 2 ORDER BY 1, 2"
        ).fetchall()
    ]
    prov_bytes = int(
        con.execute("SELECT COALESCE(SUM(length(provenance_json)), 0) FROM haz_resolutions").fetchone()[0]
    )
    report: dict[str, Any] = {
        "columns_checksummed": cols,
        "checksum_excluding_provenance": checksum,
        "counts": counts,
        "rows_total": sum(c["rows"] for c in counts),
        "provenance_json_bytes": prov_bytes,
    }
    try:
        from scripts.ci.db_table_sizes import size_report, whole_file

        info = whole_file(con)
        sizes = {t.table: t.est_bytes for t in size_report(con)}
        report["table_bytes"] = int(sizes.get("haz_resolutions", 0))
        report["snapshot_table_bytes"] = int(sizes.get(TABLE, 0))
        report["db_used_bytes"] = int(info.get("used_bytes", 0))
        report["db_block_bytes"] = int(info.get("file_bytes", 0))
    except Exception as exc:  # noqa: BLE001 - sizes are reported, never required
        report["size_error"] = str(exc)
    if db_path and os.path.exists(db_path):
        report["db_file_bytes"] = os.path.getsize(db_path)
    return report


def compare_reports(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Every difference that must not exist; an empty list means they match."""

    problems = []
    if before.get("checksum_excluding_provenance") != after.get("checksum_excluding_provenance"):
        problems.append("checksum over every column but provenance_json moved")
    if before.get("counts") != after.get("counts"):
        problems.append(f"counts moved: {before.get('counts')} -> {after.get('counts')}")
    if before.get("columns_checksummed") != after.get("columns_checksummed"):
        problems.append("the checksummed column set moved")
    return problems


def format_report(before: dict[str, Any], after: dict[str, Any] | None, rewrite: dict[str, Any] | None) -> list[str]:
    mb = 1024 * 1024
    lines = ["| hazard | status | rows before | rows after |", "|---|---|---:|---:|"]
    after_counts = {(c["hazard"], c["status"]): c["rows"] for c in (after or {}).get("counts", [])}
    for c in before.get("counts", []):
        lines.append(
            f"| {c['hazard']} | {c['status']} | {c['rows']:,} | "
            f"{after_counts.get((c['hazard'], c['status']), '—')} |"
        )
    lines.append("")

    def _size(r: dict, key: str) -> str:
        v = r.get(key)
        return f"{v / mb:,.1f} MB" if isinstance(v, (int, float)) else "—"

    lines.append("| measure | before | after |")
    lines.append("|---|---:|---:|")
    for label, key in (
        ("checksum (all columns but provenance_json)", "checksum_excluding_provenance"),
    ):
        lines.append(f"| {label} | `{before.get(key)}` | `{(after or {}).get(key, '—')}` |")
    for label, key in (
        ("provenance_json bytes", "provenance_json_bytes"),
        ("haz_resolutions on disk", "table_bytes"),
        ("haz_evidence_snapshots on disk", "snapshot_table_bytes"),
        ("database used blocks", "db_used_bytes"),
        ("database file", "db_file_bytes"),
    ):
        lines.append(f"| {label} | {_size(before, key)} | {_size(after or {}, key)} |")
    if rewrite:
        lines.append("")
        lines.append(
            f"Rewrite: {rewrite.get('rows_rewritten', 0):,} of "
            f"{rewrite.get('rows_targeted', 0):,} rows in "
            f"{rewrite.get('duration_sec', '?')} s; CAS skipped "
            f"{rewrite.get('rows_cas_skipped', 0)}, refused {rewrite.get('rows_refused', 0)}"
            f"{', stopped early' if rewrite.get('stopped_early') else ''}."
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    """``python -m resolver.hazard_resolution.evidence_snapshots --db PATH [--apply]``.

    Exit 1 when the after-report differs from the before-report in counts or
    checksum, so the workflow stops before the canonical upload.
    """

    import argparse

    import duckdb

    parser = argparse.ArgumentParser(description="Rewrite whole-cache listings in zero rows")
    parser.add_argument("--db", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--batch-rows", type=int, default=DEFAULT_BATCH_ROWS)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    path = args.db.replace("duckdb:///", "")
    con = duckdb.connect(path)
    try:
        ensure_haz_schema(con)
        before = integrity_report(con, path)
        rewrite = rewrite_zero_rows(
            con, apply=args.apply, batch_rows=args.batch_rows, max_seconds=args.max_seconds
        )
        con.execute("CHECKPOINT")
        after = integrity_report(con, path) if args.apply else None
    finally:
        con.close()

    problems = compare_reports(before, after) if after else []
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "before": before, "after": after, "rewrite": rewrite, "problems": problems,
    }
    lines = format_report(before, after, rewrite)
    if problems:
        lines += ["", "**MISMATCH:** " + "; ".join(problems)]
    text = "\n".join(lines)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
    if args.md_out:
        with open(args.md_out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    if rewrite.get("rows_refused"):
        # A refused row is left exactly as it was, so nothing is lost; it is
        # named for a person to read, and the rewrite still goes ahead.
        print(f"::warning::{rewrite['rows_refused']} zero row(s) kept their listing: "
              f"{'; '.join(rewrite.get('refusals', [])[:3])}")
    if problems:
        print("::error::haz_resolutions moved under the snapshot rewrite: " + "; ".join(problems))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
