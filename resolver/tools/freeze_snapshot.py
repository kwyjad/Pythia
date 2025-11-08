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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

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

try:
    from resolver.tools.export_facts import (
        _prepare_resolved_for_db as exporter_prepare_resolved_for_db,
        _prepare_deltas_for_db as exporter_prepare_deltas_for_db,
    )
except Exception:  # pragma: no cover - defensive: fall back to local implementations
    exporter_prepare_resolved_for_db = None  # type: ignore
    exporter_prepare_deltas_for_db = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]      # .../resolver
TOOLS = ROOT / "tools"
SNAPSHOTS = ROOT / "snapshots"
VALIDATOR = TOOLS / "validate_facts.py"

LOGGER = get_logger(__name__)


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
    cmd = [sys.executable, str(VALIDATOR), "--facts", str(facts_path)]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print("Validation failed; aborting snapshot.", file=sys.stderr)
        sys.exit(res.returncode)

def load_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in [".csv", ".tsv"]:
        return pd.read_csv(path, dtype=str).fillna("")
    elif ext == ".parquet":
        return pd.read_parquet(path)
    else:
        raise SystemExit(f"Unsupported input extension: {ext}. Use .csv or .parquet")

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
    resolved_parquet: Path
    resolved_csv: Path
    deltas_parquet: Optional[Path]
    deltas_csv: Optional[Path]
    manifest: Path


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

    run_validator(facts_path)

    ym = _parse_month(month)
    base_out_dir = Path(outdir)
    out_dir = base_out_dir / ym

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

    resolved_source = resolved_path if resolved_path else facts_path
    resolved_df = load_table(resolved_source)
    deltas_df = load_table(deltas_path) if deltas_path else None

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
        facts_df=resolved_df,
        resolved_df=load_table(resolved_path) if resolved_path else resolved_df,
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
            db_url=args.db_url,
        )
    except SnapshotError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

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
    if write_db is None:
        write_db = bool(db_url)
    if not write_db:
        LOGGER.debug("DuckDB snapshot write skipped: disabled via flag")
        return
    if not db_url:
        LOGGER.debug("DuckDB snapshot write skipped: no RESOLVER_DB_URL provided")
        return
    if duckdb_io is None:
        LOGGER.debug("DuckDB snapshot write skipped: duckdb_io unavailable")
        return

    conn = None
    try:
        conn = duckdb_io.get_db(db_url)
        duckdb_io.init_schema(conn)
        resolved_payload = resolved_df if resolved_df is not None else facts_df
        prepared_resolved = _prepare_resolved_frame_for_db(resolved_payload)
        prepared_deltas = _prepare_deltas_frame_for_db(deltas_df)

        facts_result = None
        deltas_result = None

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
        elif deltas_df is not None:
            LOGGER.debug("DuckDB snapshot facts_deltas skipped: no rows prepared")

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
    except Exception as exc:  # pragma: no cover - dual-write should not block snapshots
        print(f"Warning: DuckDB snapshot write skipped ({exc}).", file=sys.stderr)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
