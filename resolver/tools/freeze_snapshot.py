#!/usr/bin/env python3
"""
freeze_snapshot.py — validate and freeze a monthly snapshot for the resolver.

Example:
  python resolver/tools/freeze_snapshot.py \
      --facts resolver/samples/facts_sample.csv \
      --month 2025-09

What it does:
  1) Validates the input "facts" table using validate_facts.py and your registries.
  2) Writes resolver/snapshots/YYYY-MM/{facts_resolved,facts_deltas}.{csv,parquet}
  3) Writes a manifest.json with created_at_utc and source_commit_sha (if available)

Notes:
  - Accepts CSV or Parquet as input.
  - If --month is omitted, uses current UTC year-month.
  - Never mutates existing snapshots; you may overwrite only by passing --overwrite.
  - Also mirrors resolved outputs to legacy facts.{csv,parquet} for backward compatibility.
"""

import argparse
import os
import sys
import json
import subprocess
import datetime as dt
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
except ImportError:
    print("Please 'pip install pandas pyarrow' to run the freezer.", file=sys.stderr)
    sys.exit(2)

try:
    from resolver.db import duckdb_io
except Exception:  # pragma: no cover - optional dependency for db dual-write
    duckdb_io = None

from resolver.common import get_logger
from resolver.helpers.series_semantics import normalize_series_semantics
from resolver.tools.schema_validate import load_schema

try:
    from resolver.tools.export_facts import (
        _prepare_resolved_for_db as exporter_prepare_resolved_for_db,
        _prepare_deltas_for_db as exporter_prepare_deltas_for_db,
    )
except Exception:  # pragma: no cover - defensive: fall back to local implementations
    exporter_prepare_resolved_for_db = None  # type: ignore
    exporter_prepare_deltas_for_db = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]      # .../resolver
REPO_ROOT = ROOT.parent
TOOLS = ROOT / "tools"
SNAPSHOTS = ROOT / "snapshots"
VALIDATOR = TOOLS / "validate_facts.py"
SCHEMA_PATH = TOOLS / "schema.yml"
SHOCKS_PATH = ROOT / "data" / "shocks.csv"
COUNTRIES_PATH = ROOT / "data" / "countries.csv"
SUMMARY_PATH = Path("diagnostics") / "summary.md"
REPO_SUMMARY_PATH = REPO_ROOT / "diagnostics" / "summary.md"

LOGGER = get_logger(__name__)


def _append_to_summary(section_title: str, body_markdown: str) -> None:
    """Best-effort append to diagnostics/summary.md without raising on failure."""

    try:
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SUMMARY_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"\n\n### {section_title}\n\n")
            handle.write(body_markdown)
            if not body_markdown.endswith("\n"):
                handle.write("\n")
    except Exception:
        LOGGER.error("Failed to append to summary.md:\n%s", traceback.format_exc())


def _append_to_repo_summary(section_title: str, body_markdown: str) -> None:
    """Append diagnostics to the repository-level summary for CI collection."""

    try:
        REPO_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REPO_SUMMARY_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"\n\n### {section_title}\n\n")
            handle.write(body_markdown)
            if not body_markdown.endswith("\n"):
                handle.write("\n")
    except Exception:
        LOGGER.error(
            "Failed to append to repo-level summary.md:\n%s", traceback.format_exc()
        )


def _isoformat_date_strings(series: "pd.Series") -> "pd.Series":
    parsed = pd.to_datetime(series, errors="coerce")
    iso = parsed.dt.strftime("%Y-%m-%d")
    fallback = series.fillna("").astype(str)
    fallback = fallback.replace({"NaT": "", "<NA>": "", "nan": "", "NaN": ""})
    return iso.where(parsed.notna(), fallback).fillna("").astype(str)


def _fallback_prepare_resolved_for_db(df: "pd.DataFrame | None") -> "pd.DataFrame | None":
    if df is None or df.empty:
        return None

    frame = df.copy()
    for column in ("as_of_date", "publication_date"):
        if column in frame.columns:
            frame[column] = _isoformat_date_strings(frame[column])

    if "ym" not in frame.columns:
        frame["ym"] = ""
    frame["ym"] = frame["ym"].fillna("").astype(str)
    mask = frame["ym"].str.strip() == ""
    if mask.any() and "as_of_date" in frame.columns:
        frame.loc[mask, "ym"] = frame.loc[mask, "as_of_date"].astype(str).str.slice(0, 7)
        mask = frame["ym"].str.strip() == ""
    if mask.any() and "publication_date" in frame.columns:
        frame.loc[mask, "ym"] = frame.loc[mask, "publication_date"].astype(str).str.slice(0, 7)

    if "series_semantics" not in frame.columns:
        frame["series_semantics"] = ""
    frame["series_semantics"] = frame["series_semantics"].fillna("").astype(str)
    frame = normalize_series_semantics(frame)

    if "value" in frame.columns:
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")

    return frame


def _fallback_prepare_deltas_for_db(df: "pd.DataFrame | None") -> "pd.DataFrame | None":
    if df is None or df.empty:
        return None

    frame = df.copy()

    if "series_semantics_out" in frame.columns:
        semantics_out = frame["series_semantics_out"].where(
            frame["series_semantics_out"].notna(), ""
        ).astype(str)
        if "series_semantics" in frame.columns:
            semantics_current_raw = frame["series_semantics"]
            semantics_current = semantics_current_raw.astype(str)
            prefer_out = semantics_current_raw.isna() | semantics_current.str.strip().eq("")
            frame.loc[prefer_out, "series_semantics"] = semantics_out.loc[prefer_out]
        else:
            frame["series_semantics"] = semantics_out
        frame = frame.drop(columns=["series_semantics_out"])

    if "as_of" in frame.columns:
        frame["as_of"] = _isoformat_date_strings(frame["as_of"])

    if "ym" not in frame.columns:
        frame["ym"] = ""
    frame["ym"] = frame["ym"].fillna("").astype(str)
    mask = frame["ym"].str.strip() == ""
    if mask.any() and "as_of" in frame.columns:
        frame.loc[mask, "ym"] = frame.loc[mask, "as_of"].astype(str).str.slice(0, 7)

    if "series_semantics" not in frame.columns:
        frame["series_semantics"] = "new"
    semantics = frame["series_semantics"].fillna("").astype(str)
    frame["series_semantics"] = semantics.mask(
        semantics.str.strip() == "",
        "new",
    )
    frame = normalize_series_semantics(frame)

    for column in [
        "value_new",
        "value_stock",
        "rebase_flag",
        "first_observation",
        "delta_negative_clamped",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def _prepare_resolved_frame_for_db(df: "pd.DataFrame | None") -> "pd.DataFrame | None":
    if callable(exporter_prepare_resolved_for_db):  # type: ignore[arg-type]
        return exporter_prepare_resolved_for_db(df)
    return _fallback_prepare_resolved_for_db(df)


def _prepare_deltas_frame_for_db(df: "pd.DataFrame | None") -> "pd.DataFrame | None":
    if callable(exporter_prepare_deltas_for_db):  # type: ignore[arg-type]
        return exporter_prepare_deltas_for_db(df)
    return _fallback_prepare_deltas_for_db(df)

def run_validator(facts_path: Path) -> None:
    """Invoke the validator script as a subprocess for simplicity."""
    if not VALIDATOR.exists():
        print(f"Validator not found at {VALIDATOR}", file=sys.stderr)
        sys.exit(2)

    preview_dir = facts_path.parent
    try:
        preview_dir.mkdir(parents=True, exist_ok=True)
    except Exception:  # pragma: no cover - defensive guard
        pass

    stdout_path = preview_dir / "validator_stdout.txt"
    stderr_path = preview_dir / "validator_stderr.txt"
    diag_dir = Path("diagnostics") / "ingestion"
    try:
        diag_dir.mkdir(parents=True, exist_ok=True)
    except Exception:  # pragma: no cover - diagnostics best effort
        pass
    diag_stdout = diag_dir / "preview_validator.stdout.txt"
    diag_stderr = diag_dir / "preview_validator.stderr.txt"
    export_preview_dir = diag_dir / "export_preview"
    try:
        export_preview_dir.mkdir(parents=True, exist_ok=True)
    except Exception:  # pragma: no cover - diagnostics best effort
        pass
    export_preview_stderr = export_preview_dir / "validator_stderr.txt"
    export_preview_stdout = export_preview_dir / "validator_stdout.txt"

    cmd = [sys.executable, str(VALIDATOR), "--facts", str(facts_path)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    stdout_text = res.stdout or ""
    stderr_text = res.stderr or ""

    def _tail(text: str, lines: int = 200) -> str:
        return "\n".join(text.splitlines()[-lines:]).strip()

    stdout_tail = _tail(stdout_text)
    stderr_tail = _tail(stderr_text)

    stdout_path.write_text(stdout_text, encoding="utf-8")
    stderr_path.write_text(stderr_text, encoding="utf-8")
    try:
        diag_stdout.write_text(stdout_text, encoding="utf-8")
        diag_stderr.write_text(stderr_text, encoding="utf-8")
        export_preview_stdout.write_text(stdout_text, encoding="utf-8")
        export_preview_stderr.write_text(stderr_text, encoding="utf-8")
    except Exception:
        LOGGER.error("Failed to persist validator diagnostics:\n%s", traceback.format_exc())

    if res.returncode != 0:

        snapshot_md_parts: List[str] = []
        try:
            df = load_table(facts_path)
        except Exception:
            snapshot_md_parts.append(
                "(Unable to render validated CSV sample: "
                f"{traceback.format_exc().strip()})"
            )
            df = None
        else:
            try:
                head_df = df.head(5)
                try:
                    head_md = head_df.to_markdown(index=False)
                except Exception:
                    head_md = head_df.to_string(index=False)
            except Exception:
                head_md = (
                    "(Unable to render CSV sample: "
                    f"{traceback.format_exc().strip()})"
                )

            snapshot_md_parts.append(
                f"**Validated file:** `{facts_path.resolve()}`"
            )
            snapshot_md_parts.append(f"**Rows:** {len(df)}")
            snapshot_md_parts.append("")
            snapshot_md_parts.append("**facts_for_month.csv (first 5 rows):**")
            snapshot_md_parts.append("")
            snapshot_md_parts.append(head_md)
            snapshot_md_parts.append("")
            columns_text = ", ".join(str(col) for col in df.columns)
            snapshot_md_parts.append("**Columns:**")
            snapshot_md_parts.append("")
            snapshot_md_parts.append(f"```\n{columns_text}\n```")

            try:
                schema_payload = load_schema(SCHEMA_PATH)
                required_cols = list(schema_payload.get("required", []))
            except Exception:
                snapshot_md_parts.append("")
                snapshot_md_parts.append(
                    "**Required-field coverage:**\n"
                    f"(Unable to load schema: {traceback.format_exc().strip()})"
                )
            else:
                snapshot_md_parts.append("")
                snapshot_md_parts.append("**Required-field coverage:**")
                coverage_lines = []
                total_rows = len(df)
                for col in required_cols:
                    if col in df.columns:
                        series = df[col].fillna("").astype(str).str.strip()
                        non_empty = int(series.ne("").sum())
                    else:
                        non_empty = 0
                    coverage_lines.append(
                        f"- `{col}`: {non_empty} non-empty of {total_rows}"
                    )
                if coverage_lines:
                    snapshot_md_parts.append("\n".join(coverage_lines))
                else:
                    snapshot_md_parts.append("(No required columns configured)")

        stderr_block = f"```\n{stderr_tail or '(no stderr)'}\n```"
        stdout_block = f"```\n{stdout_tail or '(no stdout)'}\n```"
        snapshot_md = "\n".join(snapshot_md_parts) or "(No snapshot data captured)"

        _append_to_summary("Preview validator stdout (tail)", stdout_block)
        _append_to_summary("Preview validator stderr (tail)", stderr_block)
        _append_to_summary("facts_for_month.csv (validated) snapshot", snapshot_md)

        try:
            _append_to_repo_summary("Preview validator stdout (tail)", stdout_block)
            _append_to_repo_summary("Preview validator stderr (tail)", stderr_block)
            _append_to_repo_summary(
                "facts_for_month.csv (validated) snapshot", snapshot_md
            )
        except Exception:
            LOGGER.warning(
                "Could not append validator diagnostics to repo summary", exc_info=True
            )

        if stderr_tail:
            LOGGER.error("Preview validator stderr tail:\n%s", stderr_tail)
        if stdout_tail:
            LOGGER.error("Preview validator stdout tail:\n%s", stdout_tail)
        if not stderr_tail and not stdout_tail:
            LOGGER.error("Preview validator failed without stdout/stderr output")

        exc_msg_lines: List[str] = ["validate_facts failed"]
        if stderr_tail:
            exc_msg_lines.append("---- validator stderr (tail) ----")
            exc_msg_lines.append(stderr_tail)
        if stdout_tail:
            exc_msg_lines.append("---- validator stdout (tail) ----")
            exc_msg_lines.append(stdout_tail)
        if snapshot_md_parts:
            exc_msg_lines.append("---- facts_for_month.csv snapshot ----")
            for line in snapshot_md_parts[:8]:
                if line:
                    exc_msg_lines.append(line)
        raise SystemExit("\n".join(exc_msg_lines))


def _normalize_facts_for_validation(facts_path: Path) -> None:
    """Normalize semantic fields prior to validation (best effort)."""

    try:
        frame = pd.read_csv(facts_path, dtype=str).fillna("")
    except Exception:
        return

    if frame.empty:
        return

    if "metric" in frame.columns:
        frame["metric"] = frame["metric"].replace({
            "total_affected": "affected",
            "people_affected": "affected",
        })
        frame["metric"] = frame["metric"].where(
            frame["metric"].astype(str).str.strip().ne(""), "affected"
        )
    else:
        frame["metric"] = "affected"
    frame["metric"] = frame["metric"].astype(str).str.strip().str.lower()

    if "source_type" in frame.columns:
        frame["source_type"] = frame["source_type"].replace({"api": "agency"})
        frame["source_type"] = frame["source_type"].where(
            frame["source_type"].str.strip().ne(""),
            "agency",
        )
    else:
        frame["source_type"] = "agency"

    if "hazard_code" in frame.columns:
        frame["hazard_code"] = frame["hazard_code"].astype(str).str.strip().str.upper()
        try:
            shocks = pd.read_csv(SHOCKS_PATH, dtype=str).fillna("")
        except Exception:
            shocks = None
        if shocks is not None and not shocks.empty and "hazard_code" in shocks.columns:
            desired_cols = [
                col
                for col in ("hazard_code", "hazard_label", "hazard_class")
                if col in shocks.columns
            ]
            shocks = shocks[desired_cols].drop_duplicates("hazard_code")
            frame = frame.drop(
                columns=[c for c in ("hazard_label", "hazard_class") if c in frame.columns]
            )
            frame = frame.merge(shocks, on="hazard_code", how="left")

    # Country name enrichment
    if "iso3" in frame.columns:
        try:
            countries = pd.read_csv(COUNTRIES_PATH, dtype=str).fillna("")
            countries.columns = [c.strip().lstrip("\ufeff") for c in countries.columns]
            iso_to_country = dict(zip(countries.get("iso3", []), countries.get("country_name", [])))
        except Exception:
            iso_to_country = {}
        if "country_name" not in frame.columns:
            frame["country_name"] = frame["iso3"].map(iso_to_country).fillna("")
        else:
            mask = frame["country_name"].astype(str).str.strip().eq("")
            if mask.any():
                frame.loc[mask, "country_name"] = (
                    frame.loc[mask, "iso3"].map(iso_to_country).fillna("")
                )
    elif "country_name" not in frame.columns:
        frame["country_name"] = ""

    # Unit derived from metric (fallback to persons)
    metric_to_unit = {
        "affected": "persons",
        "in_need": "persons",
        "displaced": "persons",
        "cases": "persons_cases",
        "fatalities": "persons",
        "events": "events",
        "participants": "persons",
    }
    if "unit" not in frame.columns:
        frame["unit"] = ""
    frame["unit"] = frame.apply(
        lambda row: (
            metric_to_unit.get(str(row.get("metric", "")).strip().lower(), "persons")
            if str(row.get("unit", "")).strip() == ""
            else str(row.get("unit", "")).strip()
        ),
        axis=1,
    )

    # Publication date clamped to [as_of_date, today]
    today = dt.date.today()

    def _parse_iso_date(value: str) -> Optional[dt.date]:
        text = str(value).strip()
        if not text:
            return None
        try:
            return dt.date.fromisoformat(text)
        except ValueError:
            return None

    if "publication_date" not in frame.columns:
        frame["publication_date"] = ""

    def _clamp_publication(row: "pd.Series") -> str:
        as_of = _parse_iso_date(row.get("as_of_date", ""))
        publication = _parse_iso_date(row.get("publication_date", ""))
        if publication is None:
            publication = as_of or today
        if as_of and publication < as_of:
            publication = as_of
        if publication > today:
            publication = today
        return publication.isoformat()

    frame["publication_date"] = frame.apply(_clamp_publication, axis=1)

    defaults = {
        "publisher": "CRED / UCLouvain (EM-DAT)",
        "source_url": "https://public.emdat.be/",
        "doc_title": "EM-DAT People Affected (automated import)",
        "definition_text": (
            "EM-DAT 'Total Affected' canonicalised to monthly country-hazard affected counts."
        ),
        "method": "api",
        "confidence": "med",
        "revision": "1",
        "ingested_at": today.isoformat(),
    }

    for column, value in defaults.items():
        if column not in frame.columns:
            frame[column] = value
        else:
            frame[column] = frame[column].where(
                frame[column].astype(str).str.strip().ne(""), value
            )

    if "event_id" not in frame.columns:
        frame["event_id"] = ""

    def _fallback_event_id(row: "pd.Series") -> str:
        if str(row.get("event_id", "")).strip():
            return str(row["event_id"])
        disno = str(row.get("disno_first", "")).strip()
        if disno:
            return disno
        iso3 = str(row.get("iso3", "")).strip()
        hazard = str(row.get("hazard_code", "")).strip()
        as_of = str(row.get("as_of_date", "")).strip()
        if iso3 or hazard or as_of:
            return "-".join(filter(None, [iso3, hazard, as_of]))
        return "unknown"

    frame["event_id"] = frame.apply(_fallback_event_id, axis=1)

    try:
        frame.to_csv(facts_path, index=False)
    except Exception:
        pass

def load_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in [".csv", ".tsv"]:
        return pd.read_csv(path, dtype=str).fillna("")
    elif ext == ".parquet":
        return pd.read_parquet(path)
    else:
        raise SystemExit(f"Unsupported input extension: {ext}. Use .csv or .parquet")


def _filter_preview_to_month(preview_path: Path, month: str) -> "PreviewFilterResult":
    filtered_path = preview_path.with_name("facts_for_month.csv")
    try:
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:  # pragma: no cover - best effort directory creation
        pass

    if not preview_path.exists():
        empty = pd.DataFrame()
        empty.to_csv(filtered_path, index=False)
        return PreviewFilterResult(
            filtered_path=filtered_path,
            filtered_df=empty,
            original_rows=0,
            filtered_rows=0,
            had_ym_column=False,
        )

    df = load_table(preview_path)
    original_rows = len(df)
    had_ym_column = "ym" in df.columns

    if not had_ym_column:
        filtered_df = df.head(0).copy()
    else:
        mask = df["ym"].astype(str) == str(month)
        filtered_df = df.loc[mask].copy()

    filtered_rows = len(filtered_df)
    filtered_df.to_csv(filtered_path, index=False)

    meta_path = filtered_path.with_name("facts_for_month.meta.json")
    meta_payload = {
        "month": str(month),
        "total_rows": int(original_rows),
        "filtered_rows": int(filtered_rows),
        "ym_column_present": bool(had_ym_column),
    }
    try:
        meta_path.write_text(json.dumps(meta_payload, indent=2) + "\n", encoding="utf-8")
    except Exception:  # pragma: no cover - diagnostics only
        LOGGER.debug("Failed to write month filter metadata", exc_info=True)

    return PreviewFilterResult(
        filtered_path=filtered_path,
        filtered_df=filtered_df,
        original_rows=original_rows,
        filtered_rows=filtered_rows,
        had_ym_column=had_ym_column,
    )


def _filter_dataframe_by_month(df: "pd.DataFrame | None", month: str) -> "pd.DataFrame | None":
    if df is None:
        return None
    if "ym" not in df.columns:
        return df.copy()
    mask = df["ym"].astype(str) == str(month)
    return df.loc[mask].copy()


def write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure string columns are strings (avoid mixed dtypes)
    for c in df.columns:
        if df[c].dtype.name not in ("float64","int64","bool"):
            df[c] = df[c].astype(str)
    df.to_parquet(out_path, index=False)


@dataclass
class SnapshotResult:
    ym: str
    out_dir: Path
    resolved_parquet: Optional[Path] = None
    resolved_csv: Optional[Path] = None
    deltas_parquet: Optional[Path] = None
    deltas_csv: Optional[Path] = None
    manifest: Optional[Path] = None
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class PreviewFilterResult:
    filtered_path: Path
    filtered_df: "pd.DataFrame"
    original_rows: int
    filtered_rows: int
    had_ym_column: bool


class SnapshotError(RuntimeError):
    """Error raised when snapshot freezing fails."""


def _parse_month(value: Optional[str]) -> str:
    if value:
        try:
            year, month = map(int, value.split("-"))
            dt.date(year, month, 1)
            return f"{year:04d}-{month:02d}"
        except Exception as exc:  # pragma: no cover - defensive
            raise SnapshotError("--month must be YYYY-MM (e.g., 2025-09)") from exc
    now = dt.datetime.utcnow()
    return f"{now.year:04d}-{now.month:02d}"

def freeze_snapshot(
    *,
    facts: Path,
    month: Optional[str] = None,
    outdir: Path = SNAPSHOTS,
    overwrite: bool = False,
    deltas: Optional[Path] = None,
    resolved_csv: Optional[Path] = None,
    write_db: Optional[bool] = None,
    db_url: Optional[str] = None,
) -> SnapshotResult:
    facts_path = Path(facts)
    if not facts_path.exists():
        raise SnapshotError(f"Facts not found: {facts_path}")

    ym = _parse_month(month)
    base_out_dir = Path(outdir)
    out_dir = base_out_dir / ym

    filter_result = _filter_preview_to_month(facts_path, ym)
    filtered_facts_path = filter_result.filtered_path
    facts_df = filter_result.filtered_df

    if filter_result.original_rows and filter_result.original_rows != filter_result.filtered_rows:
        LOGGER.info(
            "Snapshot month filter applied: month=%s total_rows=%s filtered_rows=%s",
            ym,
            filter_result.original_rows,
            filter_result.filtered_rows,
        )

    metrics_summary = "(none)"
    try:
        if "metric" in facts_df.columns:
            metrics = sorted(
                {
                    str(value).strip()
                    for value in facts_df["metric"].fillna("")
                    if str(value).strip()
                }
            )
            if metrics:
                metrics_summary = ", ".join(metrics)
    except Exception:
        metrics_summary = f"(unable to summarise metrics: {traceback.format_exc().strip()})"

    context_lines = [
        f"**facts_for_month path:** `{filtered_facts_path.resolve()}`",
        f"**rows:** {filter_result.filtered_rows}",
        f"**month:** `{ym}`",
        f"**metric(s):** {metrics_summary}",
    ]
    _append_to_summary("Freeze snapshot — context", "\n".join(context_lines))

    validated_facts_df: Optional[pd.DataFrame] = None

    if facts_df.empty:
        msg = (
            f"No rows for month {ym} after filtering preview; "
            "skipping validator and snapshot"
        )
        LOGGER.info("freeze: %s", msg)
        _append_to_summary("Freeze snapshot", msg)
        _append_to_repo_summary(
            "Freeze snapshot",
            f"**{msg}**\n\n**facts_for_month path:** `{filtered_facts_path.resolve()}`",
        )
        return SnapshotResult(
            ym=ym,
            out_dir=out_dir,
            skipped=True,
            skip_reason=f"No rows for month {ym} after filtering preview",
        )

    if filter_result.filtered_rows > 0:
        _normalize_facts_for_validation(filtered_facts_path)
        run_validator(filtered_facts_path)
        try:
            validated_facts_df = load_table(filtered_facts_path)
        except Exception:
            validated_facts_df = None

    if deltas:
        deltas_path = Path(deltas)
        if not deltas_path.exists():
            raise SnapshotError(f"Deltas file not found: {deltas_path}")
    else:
        default_deltas = facts_path.with_name("deltas.csv")
        deltas_path = default_deltas if default_deltas.exists() else None

    if resolved_csv:
        resolved_path = Path(resolved_csv)
        if not resolved_path.exists():
            raise SnapshotError(f"Resolved file not found: {resolved_path}")
    else:
        default_resolved = facts_path.with_name("resolved.csv")
        resolved_path = default_resolved if default_resolved.exists() else None

    resolved_source = resolved_path if resolved_path else filtered_facts_path
    if resolved_path:
        resolved_df = load_table(resolved_source)
        resolved_df = _filter_dataframe_by_month(resolved_df, ym)
    else:
        resolved_df = facts_df.copy()
    deltas_df = load_table(deltas_path) if deltas_path else None
    deltas_df = _filter_dataframe_by_month(deltas_df, ym)

    resolved_parquet = out_dir / "facts_resolved.parquet"
    resolved_csv_out = out_dir / "facts_resolved.csv"
    manifest_out = out_dir / "manifest.json"
    deltas_csv_out = out_dir / "facts_deltas.csv" if deltas_df is not None else None
    deltas_parquet_out = out_dir / "facts_deltas.parquet" if deltas_df is not None else None

    legacy_parquet = out_dir / "facts.parquet"
    legacy_csv = out_dir / "facts.csv"

    existing_targets = [resolved_parquet, resolved_csv_out, manifest_out, legacy_parquet, legacy_csv]
    if deltas_csv_out:
        existing_targets.extend([deltas_csv_out, deltas_parquet_out])
    already = [p for p in existing_targets if p.exists()]
    if already and not overwrite:
        raise SnapshotError(
            f"Snapshot already exists for {ym}: {out_dir}. Use --overwrite to replace."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_df.to_csv(resolved_csv_out, index=False)
    write_parquet(resolved_df.copy(), resolved_parquet)

    resolved_df.to_csv(legacy_csv, index=False)
    write_parquet(resolved_df.copy(), legacy_parquet)

    if deltas_df is not None:
        deltas_df.to_csv(deltas_csv_out, index=False)
        write_parquet(deltas_df.copy(), deltas_parquet_out)

    manifest = {
        "created_at_utc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_file": str(resolved_source),
        "target_month": ym,
        "source_commit_sha": os.environ.get("GITHUB_SHA", ""),
        "resolved_rows": int(len(resolved_df)),
    }
    if deltas_df is not None:
        manifest["deltas_rows"] = int(len(deltas_df))
    manifest["artifacts"] = {
        "facts_resolved_csv": str(resolved_csv_out),
        "facts_resolved_parquet": str(resolved_parquet),
    }
    if deltas_df is not None:
        manifest["artifacts"].update(
            {
                "facts_deltas_csv": str(deltas_csv_out),
                "facts_deltas_parquet": str(deltas_parquet_out),
            }
        )

    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    _maybe_write_db(
        ym=ym,
        facts_df=facts_df,
        validated_facts_df=validated_facts_df,
        resolved_df=resolved_df,
        deltas_df=deltas_df,
        manifest=manifest,
        facts_out=resolved_parquet,
        deltas_out=deltas_parquet_out,
        write_db=write_db,
        db_url=db_url,
    )

    return SnapshotResult(
        ym=ym,
        out_dir=out_dir,
        resolved_parquet=resolved_parquet,
        resolved_csv=resolved_csv_out,
        deltas_parquet=deltas_parquet_out,
        deltas_csv=deltas_csv_out,
        manifest=manifest_out,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", required=True, help="Path to the facts CSV/Parquet to freeze")
    ap.add_argument("--month", help="Target month YYYY-MM; defaults to current UTC year-month")
    ap.add_argument("--outdir", default=str(SNAPSHOTS), help="Base output directory for snapshots")
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting existing snapshot files")
    ap.add_argument(
        "--deltas",
        help="Optional path to deltas.csv to include in the snapshot (defaults to sibling of --facts)",
    )
    ap.add_argument(
        "--resolved",
        help="Optional path to resolved.csv for DuckDB dual write (defaults to sibling of --facts)",
    )
    ap.add_argument(
        "--write-db",
        default=None,
        choices=["0", "1"],
        help="Set to 1 or 0 to force-enable or disable DuckDB dual-write (defaults to auto)",
    )
    ap.add_argument(
        "--db-url",
        default=None,
        help="Optional DuckDB URL override (defaults to RESOLVER_DB_URL)",
    )
    ap.add_argument(
        "--db",
        default=None,
        help="Alias for --db-url; DuckDB URL or path",
    )
    args = ap.parse_args()

    try:
        result = freeze_snapshot(
            facts=Path(args.facts),
            month=args.month,
            outdir=Path(args.outdir),
            overwrite=args.overwrite,
            deltas=Path(args.deltas) if args.deltas else None,
            resolved_csv=Path(args.resolved) if args.resolved else None,
            write_db=None if args.write_db is None else args.write_db == "1",
            db_url=args.db or args.db_url,
        )
    except SnapshotError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    if result.skipped:
        reason = result.skip_reason or "No rows to freeze"
        print(f"ℹ️ Snapshot skipped: {reason}")
        return

    print("✅ Snapshot written:")
    print(f" - {result.resolved_parquet}")
    print(f" - {result.resolved_csv}")
    if result.deltas_parquet:
        print(f" - {result.deltas_parquet}")
    if result.deltas_csv:
        print(f" - {result.deltas_csv}")
    print(f" - {result.manifest}")

if __name__ == "__main__":
    main()


def _maybe_write_db(
    *,
    ym: str,
    facts_df: "pd.DataFrame",
    validated_facts_df: "pd.DataFrame | None",
    resolved_df: "pd.DataFrame | None",
    deltas_df: "pd.DataFrame | None",
    manifest: Dict[str, Any],
    facts_out: Path,
    deltas_out: Path | None,
    write_db: Optional[bool] = None,
    db_url: Optional[str] = None,
) -> None:
    env_url = os.environ.get("RESOLVER_DB_URL", "").strip()
    if db_url is not None:
        db_url = db_url.strip()
    else:
        db_url = env_url
    summary_title = "Freeze snapshot — DB write"

    if write_db is None:
        allow_write = bool(db_url)
    else:
        allow_write = bool(write_db)

    if not allow_write:
        msg = "DuckDB snapshot write skipped: disabled via flag"
        LOGGER.debug(msg)
        _append_to_summary(summary_title, msg)
        _append_to_repo_summary(summary_title, msg)
        return
    if not db_url:
        msg = "DuckDB snapshot write skipped: no RESOLVER_DB_URL provided"
        LOGGER.debug(msg)
        _append_to_summary(summary_title, msg)
        _append_to_repo_summary(summary_title, msg)
        return
    if duckdb_io is None:
        msg = "DuckDB snapshot write skipped: duckdb_io unavailable"
        LOGGER.debug(msg)
        _append_to_summary(summary_title, msg)
        _append_to_repo_summary(summary_title, msg)
        return

    conn = None
    deltas_result = None
    deltas_input_rows = 0
    try:
        conn = duckdb_io.get_db(db_url)
        duckdb_io.init_schema(conn)
        resolved_payload = (
            resolved_df
            if resolved_df is not None
            else (
                validated_facts_df
                if validated_facts_df is not None and not validated_facts_df.empty
                else facts_df
            )
        )
        prepared_resolved = _prepare_resolved_frame_for_db(resolved_payload)

        if deltas_df is not None and not deltas_df.empty:
            deltas_source = deltas_df.copy()
        else:
            base_source = (
                validated_facts_df
                if validated_facts_df is not None and not validated_facts_df.empty
                else facts_df
            )
            deltas_source = base_source.copy() if base_source is not None and not base_source.empty else None

        deltas_input_rows = int(len(deltas_source)) if deltas_source is not None else 0

        if deltas_source is not None and not deltas_source.empty:
            for column in ("iso3", "ym", "hazard_code", "metric", "value"):
                if column not in deltas_source.columns:
                    deltas_source[column] = 0 if column == "value" else ""
            deltas_source["value"] = pd.to_numeric(
                deltas_source["value"], errors="coerce"
            )
            semantics_col = deltas_source.get("series_semantics")
            if semantics_col is None:
                deltas_source["series_semantics"] = "new"
            else:
                semantics = semantics_col.astype(str).str.strip()
                deltas_source.loc[semantics.eq(""), "series_semantics"] = "new"

        prepared_deltas = _prepare_deltas_frame_for_db(deltas_source)

        facts_result = None

        if prepared_resolved is not None and not prepared_resolved.empty:
            LOGGER.debug(
                "DuckDB snapshot write | table=facts_resolved rows=%s", len(prepared_resolved)
            )
            facts_result = duckdb_io.upsert_dataframe(
                conn,
                "facts_resolved",
                prepared_resolved,
                keys=duckdb_io.FACTS_RESOLVED_KEY_COLUMNS,
            )
            LOGGER.info(
                "DuckDB snapshot facts_resolved rows written: %s",
                facts_result.rows_delta,
            )
        else:
            LOGGER.debug("DuckDB snapshot facts_resolved skipped: no rows prepared")

        if prepared_deltas is not None and not prepared_deltas.empty:
            for column in ["as_of", "value_new", "value_stock", "series_semantics"]:
                if column not in prepared_deltas.columns:
                    if column == "series_semantics":
                        prepared_deltas[column] = "new"
                    elif column == "as_of":
                        prepared_deltas[column] = ""
                    else:
                        prepared_deltas[column] = pd.Series([pd.NA] * len(prepared_deltas))
            LOGGER.debug(
                "DuckDB snapshot write | table=facts_deltas rows=%s", len(prepared_deltas)
            )
            deltas_result = duckdb_io.upsert_dataframe(
                conn,
                "facts_deltas",
                prepared_deltas,
                keys=duckdb_io.FACTS_DELTAS_KEY_COLUMNS,
            )
            LOGGER.info(
                "DuckDB snapshot facts_deltas rows written: %s",
                deltas_result.rows_delta,
            )
        elif deltas_input_rows:
            LOGGER.debug("DuckDB snapshot facts_deltas skipped after preparation")
        else:
            LOGGER.debug("DuckDB snapshot facts_deltas skipped: no source rows")

        created_at = str(manifest.get("created_at_utc") or dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
        git_sha = str(manifest.get("source_commit_sha") or os.environ.get("GITHUB_SHA", ""))
        export_version = str(manifest.get("export_version") or manifest.get("schema_version", ""))
        try:
            conn.execute("DELETE FROM snapshots WHERE ym = ?", [ym])
            conn.execute(
                "INSERT INTO snapshots (ym, created_at, git_sha, export_version, facts_rows, deltas_rows) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    ym,
                    created_at,
                    git_sha,
                    export_version,
                    int(facts_result.rows_written if facts_result else 0),
                    int(deltas_result.rows_written if deltas_result else 0),
                ],
            )
            LOGGER.debug("DuckDB snapshot metadata recorded for ym=%s", ym)
        except Exception:
            LOGGER.debug("DuckDB snapshot metadata skipped", exc_info=True)
        if deltas_result is not None:
            written = int(deltas_result.rows_delta)
            msg = (
                f"Wrote {written} facts_deltas rows (input={int(deltas_result.rows_in)}) for month {ym}"
            )
        elif deltas_input_rows:
            msg = (
                f"DuckDB facts_deltas write skipped after preparation; input_rows={deltas_input_rows} for month {ym}"
            )
        else:
            msg = f"No facts_deltas rows available for month {ym}; skipped DB write"

        _append_to_summary(summary_title, msg)
        _append_to_repo_summary(summary_title, msg)

    except Exception as exc:  # pragma: no cover - dual-write should not block snapshots
        print(f"Warning: DuckDB snapshot write skipped ({exc}).", file=sys.stderr)
        _append_to_summary(summary_title, f"DuckDB snapshot write failed: {exc}")
        _append_to_repo_summary(summary_title, f"DuckDB snapshot write failed: {exc}")
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
