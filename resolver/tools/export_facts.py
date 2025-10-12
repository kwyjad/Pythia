#!/usr/bin/env python3
"""
export_facts.py — normalize arbitrary staging inputs into resolver 'facts' outputs.

Examples:
  # Export from a folder of CSVs using the default mapping config:
  python resolver/tools/export_facts.py --in resolver/staging --out resolver/exports

  # Export from a single file with explicit config:
  python resolver/tools/export_facts.py \
      --in resolver/staging/sample_source.csv \
      --config resolver/tools/export_config.yml \
      --out resolver/exports

What it does:
  - Loads one or many staging files (.csv, .parquet, .json lines)
  - Applies a column mapping + constant fills from export_config.yml
  - Coerces datatypes and enums to match the Data Dictionary
  - Writes 'facts.csv' and 'facts.parquet' to the /resolver/exports/ folder
  - Prints how many rows written and where

After exporting:
  - Validate:   python resolver/tools/validate_facts.py --facts resolver/exports/facts.csv
  - Freeze:     python resolver/tools/freeze_snapshot.py --facts resolver/exports/facts.csv --month YYYY-MM
"""

import argparse
import os
import sys
import json
import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import pandas as pd
except ImportError:
    print("Please 'pip install pandas pyarrow pyyaml' to run the exporter.", file=sys.stderr)
    sys.exit(2)

try:  # Optional dependency for DB dual-write
    from resolver.db import duckdb_io
except Exception:  # pragma: no cover - allow exporter without duckdb installed
    duckdb_io = None

from resolver.common import (
    compute_series_semantics,
    get_logger,
    dict_counts,
    df_schema,
)
from resolver.helpers.series_semantics import normalize_series_semantics

try:
    import yaml
except ImportError:
    print("Please 'pip install pyyaml' to run the exporter.", file=sys.stderr)
    sys.exit(2)

ROOT = Path(__file__).resolve().parents[1]     # .../resolver
TOOLS = ROOT / "tools"
EXPORTS = ROOT / "exports"
DEFAULT_CONFIG = TOOLS / "export_config.yml"

REQUIRED = [
    "event_id","country_name","iso3",
    "hazard_code","hazard_label","hazard_class",
    "metric","value","unit",
    "as_of_date","publication_date",
    "publisher","source_type","source_url","doc_title",
    "definition_text","method","confidence",
    "revision","ingested_at"
]

def _load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _read_one(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in [".csv", ".tsv"]:
        return pd.read_csv(path, dtype=str).fillna("")
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in [".json", ".jsonl"]:
        return pd.read_json(path, lines=True, dtype_backend="pyarrow").astype(str).fillna("")
    raise SystemExit(f"Unsupported input extension: {ext}")

def _collect_inputs(inp: Path) -> List[Path]:
    if inp.is_dir():
        files: List[Path] = []
        for p in inp.rglob("*"):
            if p.suffix.lower() in (".csv",".tsv",".parquet",".json",".jsonl"):
                files.append(p)
        return files
    return [inp]

def _apply_mapping(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    cfg structure:
      mapping:
        event_id: [ "event_id", "evt_id" ]     # take first existing
        country_name: [ "country", "country_name" ]
        ...
      constants:
        unit: "persons"
        method: "api"
        revision: 1
      transforms:
        metric:
          # map source values to canonical
          affected: ["affected","people_affected","pa"]
          in_need: ["pin","people_in_need","in_need"]
      dates:
        as_of_date: ["as_of","date_as_of"]
        publication_date: ["pub_date","published"]
    """
    mapping = cfg.get("mapping", {})
    constants = cfg.get("constants", {})
    transforms = cfg.get("transforms", {})
    dates = cfg.get("dates", {})

    out = pd.DataFrame()

    # Column mapping: take the first available source column
    for target, sources in mapping.items():
        val = None
        for s in sources:
            if s in df.columns:
                val = df[s]
                break
        if val is None:
            out[target] = ""  # will fill from constants/defaults later
        else:
            out[target] = val.fillna("").astype(str)

    # Date mapping (same approach)
    for target, sources in dates.items():
        if target not in out.columns:
            out[target] = ""
        if not out[target].any():
            for s in sources:
                if s in df.columns:
                    out[target] = df[s].astype(str).fillna("")
                    break

    # Constants/defaults
    for k, v in constants.items():
        if k not in out.columns or (out[k] == "").all():
            out[k] = str(v)

    # Metric transforms
    if "metric" in transforms:
        look = {}
        for canonical, alts in transforms["metric"].items():
            for a in alts:
                look[str(a).lower()] = canonical
        if "metric" in out.columns:
            out["metric"] = out["metric"].astype(str).str.lower().map(lambda x: look.get(x, x))

    # Fill missing required columns with empty string (we’ll validate later)
    for col in REQUIRED:
        if col not in out.columns:
            out[col] = ""

    # Coerce some types
    # value should be numeric string; keep as string for consistency; validator enforces numeric
    # ensure revision and other ints are stringified
    for c in out.columns:
        out[c] = out[c].astype(str).fillna("")

    # Add ingested_at if empty
    if "ingested_at" in out.columns:
        mask = out["ingested_at"].str.len() == 0
        if mask.any():
            ts = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            out.loc[mask, "ingested_at"] = ts

    return out

RESOLVED_DB_NUMERIC = {"value"}

DELTAS_DB_NUMERIC = {
    "value_new",
    "value_stock",
    "rebase_flag",
    "first_observation",
    "delta_negative_clamped",
}

DELTAS_DB_DEFAULTS = {"series_semantics": "new"}

LOGGER = get_logger(__name__)
def _to_month(series: "pd.Series") -> "pd.Series":
    dates = pd.to_datetime(series, errors="coerce")
    formatted = dates.dt.strftime("%Y-%m")
    return formatted.fillna("")


def _warn_on_non_canonical_semantics(label: str, frame: "pd.DataFrame | None") -> None:
    if frame is None or frame.empty or "series_semantics" not in frame.columns:
        return
    values = (
        frame["series_semantics"]
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )
    invalid = sorted(
        value for value in values if value not in {"", "new", "stock"}
    )
    if invalid:
        LOGGER.warning(
            "DuckDB write: %s frame has non-canonical series_semantics values: %s",
            label,
            invalid,
        )


def _apply_series_semantics(frame: "pd.DataFrame") -> "pd.DataFrame":
    if "series_semantics" not in frame.columns:
        frame["series_semantics"] = ""
    frame["series_semantics"] = frame.apply(
        lambda row: compute_series_semantics(
            metric=row.get("metric"), existing=row.get("series_semantics")
        ),
        axis=1,
    )
    return frame


def _isoformat_date_strings(series: "pd.Series") -> "pd.Series":
    parsed = pd.to_datetime(series, errors="coerce")
    iso = parsed.dt.strftime("%Y-%m-%d")
    fallback = series.fillna("").astype(str)
    fallback = fallback.replace({"NaT": "", "<NA>": "", "nan": "", "NaN": ""})
    return iso.where(parsed.notna(), fallback).fillna("").astype(str)


def _prepare_resolved_for_db(df: "pd.DataFrame | None") -> "pd.DataFrame | None":
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

    frame = _apply_series_semantics(frame)
    frame = normalize_series_semantics(frame)
    frame["series_semantics"] = frame["series_semantics"].fillna("").astype(str)

    for column in RESOLVED_DB_NUMERIC:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def _prepare_deltas_for_db(df: "pd.DataFrame | None") -> "pd.DataFrame | None":
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
    mask = frame["ym"].str.len() == 0
    if mask.any() and "as_of" in frame.columns:
        frame.loc[mask, "ym"] = _to_month(frame.loc[mask, "as_of"])

    if "series_semantics" not in frame.columns:
        frame["series_semantics"] = DELTAS_DB_DEFAULTS["series_semantics"]
    semantics = frame["series_semantics"].fillna("").astype(str)
    frame["series_semantics"] = semantics.mask(
        semantics.str.strip() == "",
        DELTAS_DB_DEFAULTS["series_semantics"],
    )
    frame = normalize_series_semantics(frame)

    for column in DELTAS_DB_NUMERIC:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def _parse_write_db_flag(value: str | bool | None) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        if stripped in {"1", "true", "True"}:
            return True
        if stripped in {"0", "false", "False"}:
            return False
    try:
        return bool(int(value))
    except Exception:
        return None


def _maybe_write_to_db(
    *,
    facts_resolved: "Optional[pd.DataFrame]" = None,
    facts_deltas: "Optional[pd.DataFrame]" = None,
    db_url: Optional[str] = None,
    write_db: Optional[bool] = None,
) -> None:
    """Write exported facts into DuckDB when enabled via environment or flag."""

    if duckdb_io is None:
        return
    env_url = os.environ.get("RESOLVER_DB_URL", "").strip()
    if db_url is not None:
        db_url = db_url.strip()
    else:
        db_url = env_url
    if write_db is None:
        write_db = bool(db_url)
    if not write_db or not db_url:
        LOGGER.debug("DuckDB write skipped: disabled or missing RESOLVER_DB_URL")
        return

    resolved_prepared = _prepare_resolved_for_db(facts_resolved)
    deltas_prepared = _prepare_deltas_for_db(facts_deltas)
    if resolved_prepared is None and deltas_prepared is None:
        LOGGER.debug("DuckDB write skipped: no prepared frames to persist")
        return

    conn = None
    try:
        LOGGER.info("Writing exports to DuckDB at %s", db_url)
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "Resolved frame schema: %s",
                df_schema(resolved_prepared),
            )
            LOGGER.debug(
                "Resolved series_semantics distribution: %s",
                dict_counts(
                    resolved_prepared["series_semantics"]
                    if resolved_prepared is not None
                    else []
                ),
            )
            LOGGER.debug(
                "Deltas frame schema: %s",
                df_schema(deltas_prepared),
            )
        _warn_on_non_canonical_semantics("facts_resolved", resolved_prepared)
        _warn_on_non_canonical_semantics("facts_deltas", deltas_prepared)
        conn = duckdb_io.get_db(db_url)
        duckdb_io.init_schema(conn)
        if resolved_prepared is not None and not resolved_prepared.empty:
            written_resolved = duckdb_io.upsert_dataframe(
                conn,
                "facts_resolved",
                resolved_prepared,
                keys=duckdb_io.FACTS_RESOLVED_KEY_COLUMNS,
            )
            LOGGER.info("DuckDB facts_resolved rows written: %s", written_resolved)
        if deltas_prepared is not None and not deltas_prepared.empty:
            written_deltas = duckdb_io.upsert_dataframe(
                conn,
                "facts_deltas",
                deltas_prepared,
                keys=duckdb_io.FACTS_DELTAS_KEY_COLUMNS,
            )
            LOGGER.info("DuckDB facts_deltas rows written: %s", written_deltas)
        LOGGER.info("DuckDB write complete")
    except Exception as exc:  # pragma: no cover - non fatal for exporter
        LOGGER.error("DuckDB write skipped: %s", exc, exc_info=True)
        print(f"Warning: DuckDB write skipped ({exc}).", file=sys.stderr)
    finally:
        # Shared DuckDB connections are cached per URL; leave them open so
        # subsequent writers/readers reuse the same wrapper instance.
        pass


@dataclass
class ExportResult:
    rows: int
    csv_path: Path
    parquet_path: Optional[Path]
    dataframe: "pd.DataFrame"


class ExportError(RuntimeError):
    pass


def export_facts(
    *,
    inp: Path,
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path = EXPORTS,
    write_db: str | bool | None = None,
    db_url: Optional[str] = None,
) -> ExportResult:
    if not config_path.exists():
        raise ExportError(f"Config not found: {config_path}")

    files = _collect_inputs(inp)
    if not files:
        raise ExportError(f"No supported files found under {inp}")

    cfg = _load_config(config_path)

    frames = [_read_one(f) for f in files]
    staging = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    facts = _apply_mapping(staging, cfg)
    facts = _apply_series_semantics(facts)
    for col in ["as_of_date", "publication_date"]:
        if col in facts.columns:
            parsed = pd.to_datetime(facts[col], errors="coerce")
            iso = parsed.dt.strftime("%Y-%m-%d")
            fallback = facts[col].fillna("").astype(str)
            fallback = fallback.replace({"NaT": "", "<NA>": "", "nan": "", "NaN": ""})
            facts[col] = iso.where(parsed.notna(), fallback).fillna("")
    LOGGER.info("Prepared %s exported rows", len(facts))
    LOGGER.debug("Export dataframe schema: %s", df_schema(facts))
    LOGGER.info(
        "series_semantics distribution: %s",
        dict_counts(facts.get("series_semantics", [])),
    )
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("Sample rows: %s", facts.head(5).to_dict(orient="records"))

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "facts.csv"
    pq_path = out_dir / "facts.parquet"
    facts.to_csv(csv_path, index=False)

    parquet_written: Optional[Path] = None
    try:
        facts.to_parquet(pq_path, index=False)
        parquet_written = pq_path
    except Exception as exc:
        print(f"Warning: could not write Parquet ({exc}). CSV written.", file=sys.stderr)

    _maybe_write_to_db(
        facts_resolved=facts,
        db_url=db_url,
        write_db=_parse_write_db_flag(write_db),
    )

    return ExportResult(
        rows=len(facts),
        csv_path=csv_path,
        parquet_path=parquet_written,
        dataframe=facts,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to staging file or directory")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to export_config.yml")
    ap.add_argument("--out", default=str(EXPORTS), help="Output directory (will create if needed)")
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
        result = export_facts(
            inp=Path(args.inp),
            config_path=Path(args.config),
            out_dir=Path(args.out),
            write_db=args.write_db,
            db_url=args.db_url,
        )
    except ExportError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    print(f"✅ Exported {result.rows} rows")
    if result.parquet_path and result.parquet_path.exists():
        print(f" - parquet: {result.parquet_path}")
    print(f" - csv (diagnostic): {result.csv_path}")

if __name__ == "__main__":
    main()
