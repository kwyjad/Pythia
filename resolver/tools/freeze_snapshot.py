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
from typing import Any, Dict, List, Mapping, Optional

try:
    import pandas as pd
except ImportError:
    print("Please 'pip install pandas pyarrow' to run the freezer.", file=sys.stderr)
    sys.exit(2)

try:
    from resolver.db import duckdb_io
except Exception:  # pragma: no cover - optional dependency for db dual-write
    duckdb_io = None

try:
    from resolver.db.conn_shared import canonicalize_duckdb_target
except Exception:  # pragma: no cover - optional dependency
    canonicalize_duckdb_target = None  # type: ignore[assignment]

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

INGESTION_DIAGNOSTICS_DIR = Path("diagnostics") / "ingestion"
FREEZE_DB_DIAGNOSTICS_PATH = INGESTION_DIAGNOSTICS_DIR / "freeze_db.json"
INGESTION_SUMMARY_PATH = INGESTION_DIAGNOSTICS_DIR / "summary.md"

EM_DAT_METRICS = {
    "affected",
    "total_affected",
    "people_affected",
    "in_need",
    "pin",
    "pa",
}
EM_DAT_METRICS_LOWER = {metric.lower() for metric in EM_DAT_METRICS}
EM_DAT_SIGNATURE_COLUMNS = {
    "hazard_code",
    "hazard_label",
    "hazard_class",
    "publisher",
    "source_type",
}
EM_DAT_PUBLISHER_KEYWORDS = ("EM-DAT", "CRED", "UCLouvain")

TRUTHY_FLAGS = {"1", "true", "yes", "on"}
FALSY_FLAGS = {"0", "false", "no", "off"}


def _parse_bool_flag(value: Any) -> Optional[bool]:
    """Return True/False for common CLI/env flag strings."""

    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in TRUTHY_FLAGS:
        return True
    if text in FALSY_FLAGS:
        return False
    return None


def _env_write_db_flag() -> Optional[bool]:
    """Return the RESOLVER_WRITE_DB flag value when explicitly set."""

    return _parse_bool_flag(os.environ.get("RESOLVER_WRITE_DB"))


def _freeze_validation_enabled() -> bool:
    """Return True when the FREEZE_RUN_VALIDATOR flag requests validation."""

    raw = os.environ.get("FREEZE_RUN_VALIDATOR", "")
    return raw.strip().lower() in TRUTHY_FLAGS


def _legacy_emdat_override_enabled() -> bool:
    """Return True when the opt-in EM-DAT override flag is enabled."""

    return os.environ.get("FREEZE_ENABLE_EMDAT_OVERRIDE", "0").strip() == "1"


def _is_emdat_pa_facts(facts_path: Path, sample_rows: int = 50) -> bool:
    """Return True if the facts file appears to contain EM-DAT PA metrics."""

    try:
        frame = pd.read_csv(facts_path, dtype=str, nrows=sample_rows).fillna("")
    except Exception:
        return False

    if "metric" not in frame.columns:
        return False

    metrics = {m.strip() for m in frame["metric"].astype(str).tolist()}
    return bool(metrics & EM_DAT_METRICS)


def _is_emdat_flow_frame(frame: "pd.DataFrame | None") -> bool:
    if frame is None or frame.empty or "metric" not in frame.columns:
        return False
    metrics = (
        frame["metric"].fillna("").astype(str).str.strip().str.lower()
    )
    if metrics.empty:
        return False
    return metrics.isin(EM_DAT_METRICS_LOWER).all()


def _looks_like_emdat_pa_facts(frame: "pd.DataFrame | None") -> bool:
    if frame is None or frame.empty:
        return False

    cols = set(frame.columns)
    if not EM_DAT_SIGNATURE_COLUMNS.issubset(cols):
        return False

    def _string_series(column: str) -> "pd.Series":
        if column in frame.columns:
            return frame[column].fillna("").astype(str)
        return pd.Series([""] * len(frame), index=frame.index)

    publisher = _string_series("publisher")
    source_type = _string_series("source_type")
    combined = publisher.str.cat(source_type, sep=" ", na_rep="")
    if combined.empty:
        return False

    keywords = "|".join(keyword.upper() for keyword in EM_DAT_PUBLISHER_KEYWORDS)
    return combined.str.upper().str.contains(keywords, na=False).any()


def _is_emdat_preview(frame: "pd.DataFrame | None") -> bool:
    """Return True when the preview facts look like EM-DAT rows."""

    if frame is None or frame.empty:
        return False

    def _normalize_series(name: str) -> "pd.Series | None":
        if name not in frame.columns:
            return None
        return frame[name].fillna("").astype(str)

    publisher = _normalize_series("publisher")
    if publisher is not None and not publisher.empty:
        publisher_lc = publisher.str.lower()
        if publisher_lc.str.contains("cred / uclouvain (em-dat)").any():
            return True

    loose_fields = [
        _normalize_series("publisher"),
        _normalize_series("source"),
        _normalize_series("doc_title"),
    ]
    for series in loose_fields:
        if series is None or series.empty:
            continue
        if series.str.lower().str.contains("em-dat").any():
            return True

    return False


def _count_semantics_rows(
    frame: "pd.DataFrame | None", semantics: str = "new"
) -> int:
    if frame is None or frame.empty:
        return 0
    if "series_semantics" not in frame.columns:
        return 0
    series = frame["series_semantics"].fillna("").astype(str).str.lower()
    return int(series.eq(str(semantics).strip().lower()).sum())


def _passthrough_emdat_deltas(frame: "pd.DataFrame | None") -> "pd.DataFrame | None":
    if frame is None or frame.empty:
        return None
    if "series_semantics" not in frame.columns:
        return None
    semantics = frame["series_semantics"].fillna("").astype(str)
    mask = semantics.str.lower().eq("new")
    if not mask.any():
        return None
    subset = frame.loc[mask].copy()
    return _prepare_deltas_frame_for_db(subset)


def _normalize_preview_for_deltas(
    frame: "pd.DataFrame | None", month: str
) -> "pd.DataFrame | None":
    """Normalize a preview frame so it can be written to facts_deltas 1:1."""

    if frame is None or frame.empty:
        return frame

    working = frame.copy()

    if "ym" not in working.columns:
        working["ym"] = ""
    working["ym"] = working["ym"].fillna("").astype(str).str.strip()
    mask_missing = working["ym"].eq("")
    if mask_missing.any() and "as_of_date" in working.columns:
        derived = pd.to_datetime(
            working.loc[mask_missing, "as_of_date"], errors="coerce"
        ).dt.strftime("%Y-%m")
        working.loc[mask_missing, "ym"] = derived.fillna("")
        mask_missing = working["ym"].fillna("").astype(str).str.strip().eq("")
    if mask_missing.any() and "publication_date" in working.columns:
        fallback = pd.to_datetime(
            working.loc[mask_missing, "publication_date"], errors="coerce"
        ).dt.strftime("%Y-%m")
        working.loc[mask_missing, "ym"] = fallback.fillna("")
        mask_missing = working["ym"].fillna("").astype(str).str.strip().eq("")
    if mask_missing.any():
        working.loc[mask_missing, "ym"] = month

    if "hazard_code" not in working.columns:
        working["hazard_code"] = ""
    else:
        working["hazard_code"] = working["hazard_code"].fillna("").astype(str)

    if "series_semantics" not in working.columns:
        working["series_semantics"] = "new"
    else:
        semantics = working["series_semantics"].fillna("").astype(str).str.strip()
        working["series_semantics"] = semantics.mask(semantics.eq(""), "new")

    if "value" in working.columns:
        working["value"] = pd.to_numeric(working["value"], errors="coerce")

    return working


def _ensure_emdat_flow_semantics(
    frame: "pd.DataFrame | None",
) -> tuple[pd.DataFrame | None, bool]:
    """Ensure EM-DAT flow rows are marked as series_semantics="new"."""

    if frame is None or frame.empty:
        return frame, False
    if not _is_emdat_flow_frame(frame):
        return frame, False

    working = frame.copy()
    changed = False

    for column in ("series_semantics", "semantics"):
        if column not in working.columns:
            working[column] = ""
            changed = True
        series = working[column].fillna("").astype(str)
        blank_mask = series.str.strip().eq("")
        if blank_mask.any():
            working.loc[blank_mask, column] = "new"
            changed = True
        working[column] = working[column].fillna("").astype(str)

    return working, changed


def _column_histogram(
    frame: "pd.DataFrame | None", column: str
) -> Dict[str, int]:
    if frame is None or frame.empty or column not in frame.columns:
        return {}
    series = frame[column]
    try:
        normalized = series.fillna("").astype(str).str.strip()
    except Exception:
        normalized = pd.Series(series).fillna("").astype(str).str.strip()
    counts = normalized.value_counts(dropna=False).to_dict()
    histogram: Dict[str, int] = {}
    for key, value in counts.items():
        histogram[str(key)] = int(value)
    return dict(sorted(histogram.items(), key=lambda item: item[0]))


def _series_semantics_histogram(frame: "pd.DataFrame | None") -> Dict[str, int]:
    hist = _column_histogram(frame, "series_semantics")
    if hist:
        return hist
    return _column_histogram(frame, "semantics")


def _format_histogram(histogram: Mapping[str, int] | Dict[str, int]) -> str:
    if not histogram:
        return "(none)"
    parts: List[str] = []
    for key in sorted(histogram):
        label = "(blank)" if key == "" else str(key)
        parts.append(f"{label}: {int(histogram[key])}")
    return ", ".join(parts)


def _append_ingestion_summary(block: str) -> None:
    try:
        INGESTION_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    try:
        needs_leading_newline = False
        try:
            needs_leading_newline = (
                INGESTION_SUMMARY_PATH.exists()
                and INGESTION_SUMMARY_PATH.stat().st_size > 0
            )
        except OSError:
            needs_leading_newline = False
        with INGESTION_SUMMARY_PATH.open("a", encoding="utf-8") as handle:
            if needs_leading_newline:
                handle.write("\n\n")
            handle.write(block)
            if not block.endswith("\n"):
                handle.write("\n")
    except OSError:
        LOGGER.debug("Could not append freeze ingestion summary", exc_info=True)


def _render_freeze_db_success_markdown(payload: Mapping[str, Any]) -> str:
    db_url = str(payload.get("db_url") or "")
    db_path = str(payload.get("db_path") or "")
    lines = ["## Freeze Snapshot — DB diagnostics", ""]
    if db_url:
        lines.append(f"- **DB URL:** `{db_url}`")
    else:
        lines.append("- **DB URL:** (empty)")
    if db_path:
        lines.append(f"- **DB path:** `{db_path}`")
    lines.append(f"- **Month:** `{payload.get('month', '')}`")
    routing_mode = str(payload.get("routing_mode") or "").strip()
    if routing_mode:
        lines.append(f"- **Routing mode:** `{routing_mode}`")
    lines.append(
        f"- **facts_resolved rows:** {int(payload.get('facts_resolved_rows', 0) or 0)}"
    )
    lines.append(
        "- **facts_resolved semantics:** "
        + _format_histogram(payload.get("facts_resolved_semantics", {}))
    )
    lines.append(
        "- **facts_resolved metrics:** "
        + _format_histogram(payload.get("facts_resolved_metrics", {}))
    )
    lines.append(
        f"- **facts_deltas rows:** {int(payload.get('facts_deltas_rows', 0) or 0)}"
    )
    lines.append(
        "- **facts_deltas semantics:** "
        + _format_histogram(payload.get("facts_deltas_semantics", {}))
    )
    lines.append(
        "- **facts_deltas metrics:** "
        + _format_histogram(payload.get("facts_deltas_metrics", {}))
    )
    return "\n".join(lines)


def _render_db_error_markdown(
    step: str, payload: Mapping[str, Any], exc: Exception
) -> str:
    db_url = str(payload.get("db_url") or "")
    db_path = str(payload.get("db_path") or "")
    error_type = type(exc).__name__
    error_message = " ".join(str(exc).split()) or "(empty)"
    lines = ["## DB Write Diagnostics", "", f"- **Step:** {step}"]
    if db_url:
        lines.append(f"- **DB URL:** `{db_url}`")
    if db_path:
        lines.append(f"- **DB path:** `{db_path}`")
    lines.append(f"- **Error type:** `{error_type}`")
    lines.append(f"- **Error message:** {error_message}")
    lines.append(
        f"- **facts_resolved rows:** {int(payload.get('facts_resolved_rows', 0) or 0)}"
    )
    lines.append(
        "- **facts_resolved semantics:** "
        + _format_histogram(payload.get("facts_resolved_semantics", {}))
    )
    lines.append(
        "- **facts_resolved metrics:** "
        + _format_histogram(payload.get("facts_resolved_metrics", {}))
    )
    lines.append(
        f"- **facts_deltas rows:** {int(payload.get('facts_deltas_rows', 0) or 0)}"
    )
    lines.append(
        "- **facts_deltas semantics:** "
        + _format_histogram(payload.get("facts_deltas_semantics", {}))
    )
    lines.append(
        "- **facts_deltas metrics:** "
        + _format_histogram(payload.get("facts_deltas_metrics", {}))
    )
    return "\n".join(lines)


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


def _count_rows_for_month(conn: Any | None, table: str, month: str) -> Optional[int]:
    if conn is None:
        return None
    try:
        row = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE ym = ?", [month]
        ).fetchone()
    except Exception:
        LOGGER.debug("Failed to count rows for %s", table, exc_info=True)
        return None
    if not row:
        return 0
    return int(row[0])


def _format_optional_count(value: Optional[int]) -> str:
    return str(value) if value is not None else "(n/a)"


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

    if not _is_emdat_pa_facts(facts_path):
        try:
            _append_to_summary(
                "Preview validator skipped",
                (
                    "Skipped EM-DAT validate_facts for non-EM-DAT metrics in: "
                    f"`{facts_path}`"
                ),
            )
        except Exception:
            LOGGER.debug("Failed to append 'validator skipped' note", exc_info=True)
        return

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

    if not _is_emdat_pa_facts(facts_path):
        return

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

    try:
        preview_df = load_table(facts_path)
    except Exception:
        preview_df = None

    filter_result = _filter_preview_to_month(facts_path, ym)
    filtered_facts_path = filter_result.filtered_path
    facts_df = filter_result.filtered_df

    facts_df, emdat_semantics_applied = _ensure_emdat_flow_semantics(facts_df)
    emdat_signature_preview = _looks_like_emdat_pa_facts(facts_df)
    emdat_metric_preview = _is_emdat_flow_frame(facts_df)
    emdat_preview = emdat_signature_preview or emdat_metric_preview
    emdat_reason_bits: List[str] = []
    if emdat_signature_preview:
        emdat_reason_bits.append("publisher")
    if emdat_metric_preview:
        emdat_reason_bits.append("metric")
    emdat_preview_reason = ", ".join(emdat_reason_bits) if emdat_reason_bits else "none"
    if emdat_semantics_applied:
        try:
            facts_df.to_csv(filtered_facts_path, index=False)
        except Exception:
            LOGGER.debug(
                "Failed to persist EM-DAT semantics normalization", exc_info=True
            )

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

    normalizer_applied = False
    normalizer_reason = "non-emdat facts"
    validation_enabled = _freeze_validation_enabled()

    should_normalize = False
    if filter_result.filtered_rows > 0:
        should_normalize = _is_emdat_pa_facts(filtered_facts_path)
        if should_normalize:
            if validation_enabled:
                normalizer_reason = "emdat facts"
            else:
                normalizer_reason = "emdat facts (validator disabled)"
        else:
            normalizer_reason = "non-emdat facts"

    if should_normalize and validation_enabled:
        _normalize_facts_for_validation(filtered_facts_path)
        normalizer_applied = True

    normalizer_line = (
        f"applied={str(normalizer_applied).lower()} reason={normalizer_reason}"
    )
    _append_to_summary("Freeze snapshot — normalizer", normalizer_line)
    _append_to_repo_summary("Freeze snapshot — normalizer", normalizer_line)

    validator_section_title = "Freeze Snapshot — validator"
    validator_block: str

    if filter_result.filtered_rows > 0 and validation_enabled:
        run_validator(filtered_facts_path)
        validator_lines = [
            "status: ran",
            f"rows: {filter_result.filtered_rows}",
            f"path: `{filtered_facts_path.resolve()}`",
        ]
        validator_block = "\n".join(validator_lines)
        try:
            validated_facts_df = load_table(filtered_facts_path)
        except Exception:
            validated_facts_df = None
    elif filter_result.filtered_rows <= 0:
        validator_block = f"skipped: no rows for `{ym}`"
    else:
        validator_block = "skipped (set FREEZE_RUN_VALIDATOR=1 to enable)"

    _append_to_summary(validator_section_title, validator_block)
    _append_to_repo_summary(validator_section_title, validator_block)

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

    resolved_source = filtered_facts_path
    resolved_df = _prepare_resolved_frame_for_db(facts_df.copy())
    if resolved_df is None or resolved_df.empty:
        if resolved_path:
            resolved_source = resolved_path
            resolved_df = load_table(resolved_source)
            resolved_df = _filter_dataframe_by_month(resolved_df, ym)
        else:
            resolved_df = facts_df.copy()
    resolved_df, _ = _ensure_emdat_flow_semantics(resolved_df)

    deltas_df = _prepare_deltas_frame_for_db(facts_df.copy())
    if (deltas_df is None or deltas_df.empty) and deltas_path:
        deltas_df = load_table(deltas_path)
        deltas_df = _filter_dataframe_by_month(deltas_df, ym)
        deltas_df = _prepare_deltas_frame_for_db(deltas_df)

    preview_flow_rows = _count_semantics_rows(facts_df)
    emdat_passthrough_applied = False
    deltas_prepared_rows = int(len(deltas_df)) if deltas_df is not None else 0
    if emdat_preview and preview_flow_rows:
        if deltas_prepared_rows != preview_flow_rows:
            LOGGER.warning(
                "EM-DAT preview flow mismatch; forcing passthrough",
                extra={
                    "ym": ym,
                    "preview_flow_rows": preview_flow_rows,
                    "prepared_deltas_rows": deltas_prepared_rows,
                },
            )
            fallback_deltas = _passthrough_emdat_deltas(facts_df)
            if fallback_deltas is not None and not fallback_deltas.empty:
                deltas_df = fallback_deltas
                deltas_prepared_rows = int(len(deltas_df))
                emdat_passthrough_applied = True

    parity_lines = [
        f"- ym: {ym}",
        f"- facts_for_month rows: {len(facts_df)}",
        f"- resolved rows prepared: {len(resolved_df)}",
    ]
    parity_block = "\n".join(parity_lines)
    _append_to_summary("Freeze snapshot — parity inputs", parity_block)
    _append_to_repo_summary("Freeze snapshot — parity inputs", parity_block)

    flow_lines = [
        f"- ym: {ym}",
        f"- emdat_preview={str(emdat_preview).lower()} (reason: {emdat_preview_reason})",
        f"- preview flow rows: {preview_flow_rows}",
        f"- prepared deltas rows: {deltas_prepared_rows}",
        f"- passthrough_applied={str(emdat_passthrough_applied).lower()}",
    ]
    flow_lines.append(
        f"- wrote facts_deltas rows: {deltas_prepared_rows if deltas_df is not None else 0}"
    )
    flow_block = "\n".join(flow_lines)
    _append_to_summary("Freeze snapshot — flow passthrough", flow_block)
    _append_to_repo_summary("Freeze snapshot — flow passthrough", flow_block)

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
        facts_path=legacy_csv,
        resolved_path=resolved_csv_out,
        deltas_path=deltas_csv_out,
        manifest_path=manifest_out,
        month=ym,
        db_url=db_url,
        write_db=write_db,
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
    env_cli_default = _env_write_db_flag()
    default_write_db = None
    if env_cli_default is not None:
        default_write_db = "1" if env_cli_default else "0"
    ap.add_argument(
        "--write-db",
        default=default_write_db,
        choices=["0", "1"],
        help=(
            "Set to 1 or 0 to force-enable or disable DuckDB dual-write "
            "(defaults to 0 unless RESOLVER_WRITE_DB=1)"
        ),
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
        write_db_flag = _parse_bool_flag(args.write_db)

        result = freeze_snapshot(
            facts=Path(args.facts),
            month=args.month,
            outdir=Path(args.outdir),
            overwrite=args.overwrite,
            deltas=Path(args.deltas) if args.deltas else None,
            resolved_csv=Path(args.resolved) if args.resolved else None,
            write_db=write_db_flag,
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



def _load_frame_for_db(path: Path | None) -> "pd.DataFrame | None":
    if path is None:
        return None
    try:
        return load_table(path)
    except FileNotFoundError:
        LOGGER.warning(
            "freeze_snapshot: CSV not found for DuckDB write",
            extra={"path": str(path)},
        )
        return None


def _maybe_write_db(
    *,
    facts_path: Path,
    resolved_path: Path | None,
    deltas_path: Path | None,
    manifest_path: Path | None,
    month: str,
    db_url: Optional[str] = None,
    write_db: Optional[bool] = None,
) -> None:
    env_url = os.environ.get("RESOLVER_DB_URL", "").strip()
    if db_url is not None:
        db_url = db_url.strip()
    else:
        db_url = env_url

    summary_title = "Freeze snapshot — DB write"
    routing_summary_title = "Freeze snapshot — DB routing (contract)"
    counts_summary_title = "Snapshot DB write — pre/post counts"
    upsert_keys_title = "DuckDB — upsert keys"

    if write_db is None:
        env_flag = _env_write_db_flag()
        allow_write = bool(env_flag) if env_flag is not None else False
    else:
        parsed_flag = _parse_bool_flag(write_db)
        allow_write = bool(parsed_flag) if parsed_flag is not None else bool(write_db)

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

    canonical_path = ""
    canonical_url = db_url
    if canonicalize_duckdb_target is not None and db_url:
        try:
            canonical_path, canonical_url = canonicalize_duckdb_target(db_url)
        except Exception:
            canonical_path = ""
            canonical_url = db_url

    facts_df = _load_frame_for_db(facts_path)
    if facts_df is None:
        facts_df = pd.DataFrame()
    resolved_df = _load_frame_for_db(resolved_path)
    deltas_source = _load_frame_for_db(deltas_path)

    preview_rows = int(len(facts_df)) if facts_df is not None else 0
    prepared_resolved = None
    prepared_deltas = None
    routing_mode = "unrouted"
    routing_notes: List[str] = []

    if resolved_df is not None and not resolved_df.empty:
        routing_mode = "resolved_passthrough"
        LOGGER.debug(
            "freeze_snapshot.db_routing: resolved passthrough",
            extra={
                "month": month,
                "resolved_rows": len(resolved_df),
                "deltas_rows": len(deltas_source) if deltas_source is not None else 0,
            },
        )
        prepared_resolved = _prepare_resolved_frame_for_db(resolved_df)
        if deltas_source is not None and not deltas_source.empty:
            prepared_deltas = _prepare_deltas_frame_for_db(deltas_source)
    elif deltas_source is not None and not deltas_source.empty:
        routing_mode = "deltas_passthrough"
        LOGGER.debug(
            "freeze_snapshot.db_routing: deltas passthrough",
            extra={
                "month": month,
                "deltas_rows": len(deltas_source),
            },
        )
        prepared_deltas = _prepare_deltas_frame_for_db(deltas_source)
    elif preview_rows:
        routing_mode = "preview_to_deltas"
        routing_notes.append(
            "routed preview rows to facts_deltas because no resolved/deltas inputs were provided"
        )
        LOGGER.debug(
            "freeze_snapshot.db_routing: preview to deltas",
            extra={
                "month": month,
                "preview_rows": preview_rows,
            },
        )
        if _legacy_emdat_override_enabled() and _is_emdat_preview(facts_df):
            routing_mode = "legacy_emdat_override"
            routing_notes.append(
                "legacy EM-DAT override enabled via FREEZE_ENABLE_EMDAT_OVERRIDE"
            )
        normalized = _normalize_preview_for_deltas(facts_df.copy(), month)
        prepared_deltas = _prepare_deltas_frame_for_db(normalized)
    else:
        routing_notes.append("no frames available for DuckDB write")
        LOGGER.debug(
            "freeze_snapshot.db_routing: no frames available",
            extra={"month": month},
        )

    diagnostics_payload: Dict[str, Any] = {
        "db_url": canonical_url or db_url or "",
        "db_path": canonical_path or "",
        "month": month,
        "facts_resolved_rows": int(len(prepared_resolved)) if prepared_resolved is not None else 0,
        "facts_deltas_rows": int(len(prepared_deltas)) if prepared_deltas is not None else 0,
        "facts_resolved_semantics": _series_semantics_histogram(prepared_resolved),
        "facts_deltas_semantics": _series_semantics_histogram(prepared_deltas),
        "facts_resolved_metrics": _column_histogram(prepared_resolved, "metric"),
        "facts_deltas_metrics": _column_histogram(prepared_deltas, "metric"),
        "routing_mode": routing_mode,
        "preview_rows": preview_rows,
        "routing_notes": routing_notes,
    }

    manifest_payload: Dict[str, Any] | None = None
    manifest_entries: List[Dict[str, Any]] | None = None
    if manifest_path and manifest_path.exists():
        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(manifest_payload, dict):
                artifacts = manifest_payload.get("artifacts")
                if isinstance(artifacts, Mapping):
                    manifest_entries = []
                    for name, artifact in artifacts.items():
                        entry: Dict[str, Any] = {"name": str(name)}
                        if isinstance(artifact, Mapping):
                            entry.update({k: v for k, v in artifact.items()})
                            path_value = str(artifact.get("path") or artifact.get("uri") or "").strip()
                        else:
                            path_value = str(artifact)
                        if path_value:
                            entry["path"] = path_value
                        if not entry.get("path"):
                            entry["path"] = f"{month}/{name}"
                        resolved_rows = manifest_payload.get("resolved_rows")
                        deltas_rows = manifest_payload.get("deltas_rows")
                        if "rows" not in entry:
                            if "resolved" in name and resolved_rows is not None:
                                entry["rows"] = resolved_rows
                            elif "deltas" in name and deltas_rows is not None:
                                entry["rows"] = deltas_rows
                        manifest_entries.append(entry)
                elif isinstance(artifacts, list):
                    manifest_entries = [
                        dict(item)
                        for item in artifacts
                        if isinstance(item, Mapping)
                    ]
        except Exception:
            LOGGER.debug("Could not parse manifest for DuckDB diagnostics", exc_info=True)

    success_block = _render_freeze_db_success_markdown(diagnostics_payload)
    try:
        INGESTION_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
        with FREEZE_DB_DIAGNOSTICS_PATH.open("w", encoding="utf-8") as handle:
            json.dump(diagnostics_payload, handle, indent=2, sort_keys=True)
    except OSError:
        LOGGER.debug("Could not write freeze DB diagnostics", exc_info=True)

    conn = None
    pre_counts = {"resolved": None, "deltas": None}
    post_counts = {"resolved": None, "deltas": None}
    try:
        conn = duckdb_io.get_db(db_url)
        LOGGER.info(
            "freeze_snapshot.db_routing",
            extra={
                "month": month,
                "routing_mode": routing_mode,
                "facts_resolved_rows": diagnostics_payload["facts_resolved_rows"],
                "facts_deltas_rows": diagnostics_payload["facts_deltas_rows"],
                "preview_rows": preview_rows,
                "db_url": canonical_url or db_url or "",
            },
        )
        write_result = duckdb_io.write_snapshot(
            conn,
            ym=month,
            facts_resolved=prepared_resolved,
            facts_deltas=prepared_deltas,
            manifests=manifest_entries or ([manifest_payload] if manifest_payload else None),
            meta={
                **(manifest_payload if isinstance(manifest_payload, Mapping) else {}),
                "facts_path": str(facts_path),
                "resolved_path": str(resolved_path or ""),
                "deltas_path": str(deltas_path or ""),
            },
        )

        facts_rows = diagnostics_payload["facts_resolved_rows"]
        deltas_rows = diagnostics_payload["facts_deltas_rows"]
        LOGGER.info(
            "DuckDB snapshot written",
            extra={
                "month": month,
                "facts_resolved_rows": facts_rows,
                "facts_deltas_rows": deltas_rows,
                "db_url": canonical_url or db_url,
            },
        )
        upsert_keys_payload: Dict[str, Any] = {}
        if isinstance(write_result, Mapping):
            pre_counts = (
                write_result.get("pre_counts")
                if isinstance(write_result.get("pre_counts"), Mapping)
                else pre_counts
            )
            post_counts = (
                write_result.get("post_counts")
                if isinstance(write_result.get("post_counts"), Mapping)
                else post_counts
            )
            raw_upsert = write_result.get("upsert_keys")
            if isinstance(raw_upsert, Mapping):
                upsert_keys_payload = {
                    str(table): value
                    for table, value in raw_upsert.items()
                    if isinstance(value, Mapping)
                }
        if pre_counts["resolved"] is None:
            pre_counts["resolved"] = _count_rows_for_month(
                conn, "facts_resolved", month
            )
        if pre_counts["deltas"] is None:
            pre_counts["deltas"] = _count_rows_for_month(conn, "facts_deltas", month)
        if post_counts["resolved"] is None:
            post_counts["resolved"] = _count_rows_for_month(
                conn, "facts_resolved", month
            )
        if post_counts["deltas"] is None:
            post_counts["deltas"] = _count_rows_for_month(conn, "facts_deltas", month)
        notes_values = diagnostics_payload.get("routing_notes") or []
        notes_text = "; ".join(
            str(note).strip() for note in notes_values if str(note).strip()
        )
        if not notes_text:
            notes_text = "none"
        db_url_display = canonical_url or db_url or ""
        db_path_display = canonical_path or ""
        routing_lines = [
            f"- DB URL: `{db_url_display}`" if db_url_display else "- DB URL: (empty)"
        ]
        if db_path_display:
            routing_lines.append(f"- DB path: `{db_path_display}`")
        else:
            routing_lines.append("- DB path: (empty)")
        routing_lines.extend(
            [
                f"- Month: `{month}`",
                f"- Routing mode: `{routing_mode}`",
                f"- Preview rows: {preview_rows}",
                f"- Prepared facts_resolved rows: {facts_rows}",
                f"  - semantics: {_format_histogram(diagnostics_payload['facts_resolved_semantics'])}",
                f"- Prepared facts_deltas rows: {deltas_rows}",
                f"  - semantics: {_format_histogram(diagnostics_payload['facts_deltas_semantics'])}",
                f"- Notes: {notes_text}",
            ]
        )
        routing_block = "\n".join(routing_lines)
        _append_to_summary(routing_summary_title, routing_block)
        _append_to_repo_summary(routing_summary_title, routing_block)
        if upsert_keys_payload:
            table_lines = ["| table | rows_written | keys |", "| --- | --- | --- |"]
            for table in sorted(upsert_keys_payload):
                entry = upsert_keys_payload.get(table) or {}
                keys = entry.get("keys")
                if isinstance(keys, (list, tuple)):
                    keys_text = ", ".join(str(k) for k in keys)
                else:
                    keys_text = str(keys or "")
                rows_value = entry.get("rows")
                try:
                    rows_text = str(int(rows_value))
                except Exception:
                    rows_text = str(rows_value or 0)
                table_lines.append(
                    f"| {table} | {rows_text} | {keys_text or '(none)'} |"
                )
            upsert_block = "\n".join(table_lines)
            _append_to_summary(upsert_keys_title, upsert_block)
            _append_to_repo_summary(upsert_keys_title, upsert_block)
        counts_lines = [
            f"ym: `{month}`",
            (
                "pre:   resolved="
                f"{_format_optional_count(pre_counts['resolved'])}, "
                f"deltas={_format_optional_count(pre_counts['deltas'])}"
            ),
            (
                f"wrote: resolved={facts_rows}, "
                f"deltas={deltas_rows}"
            ),
            (
                "post:  resolved="
                f"{_format_optional_count(post_counts['resolved'])}, "
                f"deltas={_format_optional_count(post_counts['deltas'])}"
            ),
            f"routing: `{routing_mode}`",
        ]
        counts_block = "\n".join(counts_lines)
        _append_to_summary(counts_summary_title, counts_block)
        _append_to_repo_summary(counts_summary_title, counts_block)
        db_write_lines = [
            f"- DB URL: `{db_url_display}`" if db_url_display else "- DB URL: (empty)"
        ]
        if db_path_display:
            db_write_lines.append(f"- DB path: `{db_path_display}`")
        else:
            db_write_lines.append("- DB path: (empty)")
        db_write_lines.extend(
            [
                f"- Month: `{month}`",
                f"- facts_resolved rows written: {facts_rows}",
                f"- facts_deltas rows written: {deltas_rows}",
                f"- Routing mode: `{routing_mode}`",
                f"- Notes: {notes_text}",
            ]
        )
        db_write_block = "\n".join(db_write_lines)
        _append_ingestion_summary(success_block)
        _append_to_summary(summary_title, db_write_block)
        _append_to_repo_summary(summary_title, db_write_block)
    except Exception as exc:  # pragma: no cover - dual-write should not block snapshots
        print(f"Warning: DuckDB snapshot write skipped ({exc}).", file=sys.stderr)
        _append_db_error_to_summary(
            section=f"Freeze Snapshot — DB write ({month})",
            exc=exc,
            db_url=canonical_url or db_url,
            facts_path=facts_path,
            resolved_path=resolved_path,
            deltas_path=deltas_path,
            month=month,
            routing_mode=routing_mode,
            resolved_rows=diagnostics_payload["facts_resolved_rows"],
            deltas_rows=diagnostics_payload["facts_deltas_rows"],
        )
        error_block = _render_db_error_markdown("Freeze Snapshot", diagnostics_payload, exc)
        _append_ingestion_summary(error_block)
        _append_to_summary(summary_title, f"DuckDB snapshot write failed: {exc}")
        _append_to_repo_summary(summary_title, f"DuckDB snapshot write failed: {exc}")
    finally:
        duckdb_io.close_db(conn)


def _append_db_error_to_summary(
    *,
    section: str,
    exc: Exception,
    db_url: str | None,
    facts_path: Path,
    resolved_path: Path | None,
    deltas_path: Path | None,
    month: str,
    routing_mode: str,
    resolved_rows: int,
    deltas_rows: int,
) -> None:
    """Append a freeze snapshot DuckDB error block to diagnostics summary."""

    if not sys.executable:
        return

    context = {
        "db_url": db_url or "",
        "exception_class": type(exc).__name__,
        "facts_path": str(facts_path),
        "resolved_path": str(resolved_path or ""),
        "deltas_path": str(deltas_path or ""),
        "month": month,
        "routing_mode": routing_mode,
        "facts_resolved_rows": int(resolved_rows or 0),
        "facts_deltas_rows": int(deltas_rows or 0),
    }

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.ci.append_error_to_summary",
                "--section",
                section,
                "--error-type",
                type(exc).__name__,
                "--message",
                str(exc),
                "--context",
                json.dumps(context, sort_keys=True),
            ],
            check=False,
        )
    except Exception:
        LOGGER.debug("Failed to append freeze snapshot DB error", exc_info=True)


if __name__ == "__main__":
    main()
