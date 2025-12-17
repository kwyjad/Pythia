# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""ACLED monthly fatalities → DuckDB writer CLI."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from resolver.db import duckdb_io
from resolver.db.schema_keys import ACLED_MONTHLY_FATALITIES_KEY_COLUMNS
from resolver.ingestion.acled_client import ACLEDClient
from resolver.ingestion.utils.iso_normalize import to_iso3
from scripts.ci import append_error_to_summary

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)

DEFAULT_DB_PATH = Path("./resolver_data/resolver.duckdb")
ROOT = Path(__file__).resolve().parents[2]
ACLED_DIAGNOSTICS_DIR = ROOT / "diagnostics" / "acled"
ACLED_DUCKDB_SUMMARY_PATH = ACLED_DIAGNOSTICS_DIR / "duckdb_summary.md"
ACLED_INGESTION_DIAGNOSTICS_DIR = ROOT / "diagnostics" / "ingestion" / "acled"
ACLED_CLI_FETCH_META_PATH = ACLED_INGESTION_DIAGNOSTICS_DIR / "cli_fetch_meta.json"
ACLED_HTTP_DIAG_PATH = ACLED_INGESTION_DIAGNOSTICS_DIR / "http_diag.json"
ACLED_CLI_FRAME_DIAG_PATH = ACLED_DIAGNOSTICS_DIR / "acled_cli_frame_diag.json"
SUMMARY_HEADER = "### ACLED HTTP summary"


def _relpath(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:  # pragma: no cover - defensive
        return path.as_posix()


def _normalize_db_url_arg(raw: str | None) -> tuple[str, str]:
    """Return (DuckDB URL, filesystem path) for ``raw`` input."""

    candidate = (raw or "").strip() or str(DEFAULT_DB_PATH)
    if "://" in candidate and not candidate.lower().startswith("duckdb://"):
        return candidate, candidate
    if candidate.lower().startswith("duckdb://"):
        if candidate.lower().startswith("duckdb:///"):
            fs_part = candidate[len("duckdb:///") :]
            fs_path = Path(fs_part).expanduser().resolve()
            return f"duckdb:///{fs_path.as_posix()}", str(fs_path)
        return candidate, candidate
    fs_path = Path(candidate).expanduser().resolve()
    return f"duckdb:///{fs_path.as_posix()}", str(fs_path)


def _parse_countries(raw: str | None) -> list[str]:
    if not raw:
        return []
    values: list[str] = []
    for item in raw.split(","):
        value = item.strip().upper()
        if value:
            values.append(value)
    return values


def _normalize_iso3(series: pd.Series) -> pd.Series:
    if series.empty:
        return series.astype("string")
    normalized = []
    for value in series:
        if pd.isna(value):
            normalized.append(pd.NA)
            continue
        iso = to_iso3(str(value))
        if iso:
            normalized.append(iso)
            continue
        text = str(value).strip().upper()
        normalized.append(text or pd.NA)
    result = pd.Series(normalized, index=series.index, dtype="string")
    return result


def _count_missing_iso3(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    if "iso3" not in frame.columns:
        return len(frame)
    series = frame["iso3"]
    normalized = series.astype("string")
    mask = normalized.isna() | (normalized.str.strip() == "")
    return int(mask.sum())


def _ensure_iso3(work: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``iso3`` exists, normalizes, and falls back to ``country`` when missing."""

    if "iso3" in work.columns:
        iso_series = _normalize_iso3(work["iso3"])
    else:
        iso_series = pd.Series(pd.NA, index=work.index, dtype="string")
    iso_series = iso_series.astype("string")
    missing_mask = iso_series.isna() | (iso_series.str.strip() == "")
    if "country" in work.columns:
        country_iso = _normalize_iso3(work["country"])
        country_iso = country_iso.astype("string")
        iso_series = iso_series.mask(missing_mask, country_iso)
    iso_series = iso_series.astype("string").str.strip().str.upper()
    iso_series = iso_series.replace("", pd.NA)
    work = work.copy()
    work["iso3"] = iso_series
    return work


def _format_preview_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if pd.isna(value):
        return ""
    return str(value)


def _build_acled_summary_lines(
    *,
    frame: pd.DataFrame,
    start: str,
    end: str,
    page_size: int,
    fields_mode: str,
    fields_value: str = "",
    diagnostics_path: str | None = None,
) -> list[str]:
    rows = [
        SUMMARY_HEADER,
        "",
        f"- Window: {start} → {end}",
        f"- Page size: {page_size}",
        f"- Fields parameter: {'pipe (`|`)' if fields_mode == 'pipe' else 'unset'}",
        f"- Diagnostics meta: {diagnostics_path or _relpath(ACLED_CLI_FETCH_META_PATH)}",
        f"- Rows: {len(frame)}",
    ]
    if fields_value:
        rows.insert(5, f"- Fields value: {fields_value}")
    preview_cols = ["iso3", "month", "fatalities"]
    if frame.empty or not all(col in frame.columns for col in preview_cols):
        rows.append("- Preview: (no rows)")
        return rows
    preview = frame.loc[:, preview_cols].head(3)
    if preview.empty:
        rows.append("- Preview: (no rows)")
        return rows
    rows.append("")
    rows.append("| iso3 | month | fatalities |")
    rows.append("| --- | --- | --- |")
    for record in preview.itertuples(index=False):
        iso3, month, fatalities = record
        rows.append(
            f"| {_format_preview_value(iso3)} | {_format_preview_value(month)} | {_format_preview_value(fatalities)} |"
        )
    return rows


def _append_summary_to_step(lines: list[str]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    text = "\n".join([""] + lines) + "\n"
    with open(summary_path, "a", encoding="utf-8") as handle:
        handle.write(text)


def _write_acled_duckdb_summary(lines: list[str]) -> None:
    ACLED_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines).rstrip() + "\n"
    ACLED_DUCKDB_SUMMARY_PATH.write_text(text, encoding="utf-8")


def _load_cli_fetch_meta() -> dict:
    try:
        payload = json.loads(ACLED_CLI_FETCH_META_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except (OSError, json.JSONDecodeError):  # pragma: no cover - diagnostics best effort
        return {}
    return {}


def _print_cli_fetch_summary(*, start: str, end: str, rows: int, meta: dict) -> None:
    fields_param = bool(meta.get("fields_param"))
    print("ACLED CLI — fetch summary")
    print(f" start: {start}")
    print(f" end: {end}")
    print(f" rows: {rows}")
    print(f" fields_param: {'true' if fields_param else 'false'}")


def _write_cli_frame_diag(payload: dict) -> None:
    try:
        ACLED_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, indent=2, sort_keys=True)
        ACLED_CLI_FRAME_DIAG_PATH.write_text(text + "\n", encoding="utf-8")
    except Exception:  # pragma: no cover - diagnostics best effort
        LOGGER.debug("acled_to_duckdb.frame_diag_write_failed", exc_info=True)


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="Inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Inclusive end date (YYYY-MM-DD)")
    parser.add_argument(
        "--db",
        dest="db",
        default=str(DEFAULT_DB_PATH),
        help="DuckDB URL or filesystem path (default: %(default)s)",
    )
    parser.add_argument(
        "--countries",
        dest="countries",
        default=None,
        help="Optional comma-separated ISO3 country codes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch data but skip DuckDB writes",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("RESOLVER_LOG_LEVEL", "INFO"),
        help="Set logging level (default: %(default)s)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    LOGGER.setLevel(level)

    duckdb_url, duckdb_path = _normalize_db_url_arg(args.db)
    if duckdb_path and "://" not in duckdb_path:
        Path(duckdb_path).parent.mkdir(parents=True, exist_ok=True)
    elif duckdb_url.lower().startswith("duckdb:///"):
        fs_part = duckdb_url[len("duckdb:///") :]
        Path(fs_part).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    if not duckdb_io.DUCKDB_AVAILABLE:
        reason = duckdb_io.duckdb_unavailable_reason()
        print(f"DuckDB is unavailable: {reason}", file=sys.stderr)
        return 2

    countries = _parse_countries(args.countries)
    LOGGER.info(
        "acled_to_duckdb.start | start=%s end=%s countries=%s db_url=%s dry_run=%s",
        args.start,
        args.end,
        countries or "<all>",
        duckdb_url,
        bool(args.dry_run),
    )

    client = ACLEDClient()
    try:
        frame = client.monthly_fatalities(args.start, args.end, countries=countries or None)
    except Exception as exc:  # pragma: no cover - error path
        context = {
            "start": args.start,
            "end": args.end,
            "countries": countries or "<all>",
            "db_url": duckdb_url,
            "dry_run": bool(args.dry_run),
        }
        LOGGER.error(
            "acled_to_duckdb.monthly_fatalities_error | start=%s end=%s countries=%s exc=%s",  # noqa: G004
            args.start,
            args.end,
            countries or "<all>",
            exc,
            exc_info=True,
        )
        try:
            append_error_to_summary.append_error(
                "ACLED CLI — monthly_fatalities error",
                type(exc).__name__,
                str(exc),
                context,
            )
        except Exception:  # pragma: no cover - diagnostics best effort
            pass
        raise
    LOGGER.info("acled_to_duckdb.fetch_done | rows=%s", len(frame))

    fetch_meta = _load_cli_fetch_meta()
    _print_cli_fetch_summary(
        start=args.start,
        end=args.end,
        rows=len(frame),
        meta=fetch_meta,
    )

    if frame is None:
        frame = pd.DataFrame()

    input_rows = len(frame)
    missing_iso3_before = _count_missing_iso3(frame)
    missing_iso3_after = missing_iso3_before

    if not frame.empty:
        work = _ensure_iso3(frame.copy())
        missing_iso3_after = _count_missing_iso3(work)
        work = work.dropna(subset=["iso3"]).copy()
        work["iso3"] = work["iso3"].astype(str).str.upper()
        work["month"] = (
            pd.to_datetime(work["month"], errors="coerce")
            .dt.to_period("M")
            .dt.to_timestamp(how="start")
        )
        work = work.dropna(subset=["month"]).copy()
        work["fatalities"] = (
            pd.to_numeric(work.get("fatalities"), errors="coerce")
            .fillna(0)
            .astype("int64")
        )
        if "source" not in work.columns:
            work["source"] = "ACLED"
        else:
            work["source"] = work["source"].astype(str).fillna("ACLED")
        if "updated_at" not in work.columns:
            work["updated_at"] = pd.Timestamp.now(tz="UTC")
        else:
            work["updated_at"] = pd.to_datetime(work["updated_at"], errors="coerce", utc=True)
            work["updated_at"] = work["updated_at"].fillna(pd.Timestamp.now(tz="UTC"))
        if isinstance(work["updated_at"].dtype, pd.DatetimeTZDtype):
            work["updated_at"] = work["updated_at"].dt.tz_convert(None)
        work = work.sort_values(["iso3", "month"]).reset_index(drop=True)
        frame = work
    else:
        LOGGER.warning(
            "acled_to_duckdb.fetch_empty | start=%s end=%s countries=%s",  # noqa: G004
            args.start,
            args.end,
            countries or "<all>",
        )
        frame = frame.copy()

    LOGGER.debug(
        "acled_to_duckdb.normalized | rows=%s preview=%s",
        len(frame),
        frame.head(3).to_dict("records") if not frame.empty else [],
    )

    # ``ACLEDClient`` exposes ``page_size``/``fields`` attributes during real runs, but
    # test stubs may omit them. Guard access so diagnostics never crash the CLI.
    _page_size = getattr(client, "page_size", None)
    try:
        fields_mode = "pipe" if fetch_meta.get("fields_param") else "unset"
        diagnostics_path = _relpath(ACLED_CLI_FETCH_META_PATH)
        summary_lines = _build_acled_summary_lines(
            frame=frame,
            start=args.start,
            end=args.end,
            page_size=int(_page_size) if (_page_size is not None) else 0,
            fields_mode=fields_mode,
            fields_value=str(fetch_meta.get("fields_value") or ""),
            diagnostics_path=diagnostics_path,
        )
    except Exception as exc:  # pragma: no cover - diagnostics best effort
        LOGGER.debug(
            "acled_to_duckdb.summary_build_skipped | reason=%s", exc, exc_info=True
        )
        summary_lines = []

    preview_records: list[dict] = []
    preview_cols = ["iso3", "month", "fatalities"]
    if not frame.empty and all(col in frame.columns for col in preview_cols):
        preview_df = frame.loc[:, preview_cols].head(3)
        for record in preview_df.to_dict("records"):
            fatalities_value = record.get("fatalities", 0)
            if pd.isna(fatalities_value):
                fatalities_value = 0
            try:
                fatalities_value = int(fatalities_value)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                fatalities_value = 0
            preview_records.append(
                {
                    "iso3": _format_preview_value(record["iso3"]),
                    "month": _format_preview_value(record["month"]),
                    "fatalities": fatalities_value,
                }
            )

    diag_payload = {
        "input_rows": input_rows,
        "output_rows": len(frame),
        "missing_iso3_before": missing_iso3_before,
        "missing_iso3_after": missing_iso3_after,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "preview_rows": preview_records,
    }
    _write_cli_frame_diag(diag_payload)

    iso_counts_df = pd.DataFrame(columns=["iso3", "count"])
    cod_present = False
    if not frame.empty and "iso3" in frame.columns:
        iso_series = (
            frame["iso3"]
            .astype("string")
            .str.strip()
            .str.upper()
        )
        iso_series = iso_series[~iso_series.isna() & (iso_series != "")]
        cod_present = "COD" in set(iso_series)
        if not iso_series.empty:
            iso_counts_df = (
                iso_series.to_frame(name="iso3")
                .groupby("iso3", as_index=False)
                .size()
                .rename(columns={"size": "count"})
                .sort_values(["count", "iso3"], ascending=[False, True])
                .head(10)
            )

    print("ACLED CLI — top ISO3 counts (first 10):")
    if iso_counts_df.empty:
        print(" - (no iso3 values)")
    else:
        for record in iso_counts_df.itertuples(index=False):
            print(f" - {record.iso3}: {int(record.count)}")
    print(f" - COD present: {'yes' if cod_present else 'no'}")

    diagnostics_block = [
        "",
        "### ACLED CLI diagnostics",
        f"- Input rows: {input_rows}",
        f"- iso3 missing before fallback: {missing_iso3_before}",
        f"- iso3 missing after fallback: {missing_iso3_after}",
    ]
    if preview_records:
        diagnostics_block.append("- Preview (first 3 grouped rows):")
        diagnostics_block.append("")
        diagnostics_block.append("| iso3 | month | fatalities |")
        diagnostics_block.append("| --- | --- | --- |")
        for record in preview_records:
            diagnostics_block.append(
                f"| {record['iso3']} | {record['month']} | {record['fatalities']} |"
            )
    else:
        diagnostics_block.append("- Preview: (no grouped rows)")
    summary_lines.extend(diagnostics_block)

    iso_counts_block = [
        "",
        "#### ISO3 counts (top 10)",
        f"- COD present: {'yes' if cod_present else 'no'}",
    ]
    if iso_counts_df.empty:
        iso_counts_block.append("- ISO3 counts: (no iso3 values)")
    else:
        iso_counts_block.append("")
        iso_counts_block.append("| iso3 | count |")
        iso_counts_block.append("| --- | --- |")
        for record in iso_counts_df.itertuples(index=False):
            iso_counts_block.append(f"| {record.iso3} | {int(record.count)} |")
    summary_lines.extend(iso_counts_block)

    if frame.empty:
        meta_rel = _relpath(ACLED_CLI_FETCH_META_PATH)
        http_rel = _relpath(ACLED_HTTP_DIAG_PATH)
        zero_rows_block = [
            "",
            "#### ACLED CLI zero-row diagnostics",
            "- Rows fetched: 0",
            f"- Inspect {meta_rel} for request parameters",
            f"- Inspect {http_rel} for HTTP diagnostics",
        ]
        summary_lines.extend(zero_rows_block)
    _write_acled_duckdb_summary(summary_lines)
    _append_summary_to_step(summary_lines)

    if args.dry_run:
        LOGGER.info(
            "acled_to_duckdb.dry_run | rows=%s keys=%s", len(frame), ACLED_MONTHLY_FATALITIES_KEY_COLUMNS
        )
        print("✅ Wrote 0 rows to DuckDB (dry-run)")
        print(" - acled_monthly_fatalities Δ=0 total=(skipped)")
        return 0

    conn = duckdb_io.get_db(duckdb_url)
    try:
        result = duckdb_io.upsert_dataframe(
            conn,
            "acled_monthly_fatalities",
            frame,
            keys=ACLED_MONTHLY_FATALITIES_KEY_COLUMNS,
        )
    finally:
        duckdb_io.close_db(conn)

    LOGGER.info(
        "acled_to_duckdb.write_done | rows_in=%s rows_written=%s rows_before=%s rows_after=%s rows_delta=%s keys=%s db=%s",
        len(frame),
        result.rows_written,
        result.rows_before,
        result.rows_after,
        result.rows_delta,
        ACLED_MONTHLY_FATALITIES_KEY_COLUMNS,
        duckdb_path or duckdb_url,
    )

    print(f"✅ Wrote {result.rows_written} rows to DuckDB")
    print(f" - acled_monthly_fatalities Δ={result.rows_delta} total={result.rows_after}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(run())
