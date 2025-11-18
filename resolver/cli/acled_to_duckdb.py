"""ACLED monthly fatalities → DuckDB writer CLI."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
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
SUMMARY_HEADER = "### ACLED HTTP summary"


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
) -> list[str]:
    rows = [
        SUMMARY_HEADER,
        "",
        f"- Window: {start} → {end}",
        f"- Page size: {page_size}",
        f"- Fields parameter: {'pipe (`|`)' if fields_mode == 'pipe' else 'unset'}",
        f"- Rows: {len(frame)}",
    ]
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

    if frame is None:
        frame = pd.DataFrame()

    if not frame.empty:
        work = frame.copy()
        if "iso3" in work.columns:
            work["iso3"] = _normalize_iso3(work["iso3"])
        else:
            work["iso3"] = pd.Series(pd.NA, index=work.index, dtype="string")
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
        if pd.api.types.is_datetime64tz_dtype(work["updated_at"]):
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

    summary_lines = _build_acled_summary_lines(
        frame=frame,
        start=args.start,
        end=args.end,
        page_size=client.page_size,
        fields_mode="pipe" if client.fields else "unset",
    )
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
