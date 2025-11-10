"""CLI to load canonical facts, derive deltas, and export Parquet snapshots."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from resolver.db import duckdb_io
from resolver.db.conn_shared import get_shared_duckdb_conn
from resolver.db.duckdb_io import init_schema
from resolver.transform.resolve_sources import resolve_sources

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:  # pragma: no cover - library default noise guard
    LOGGER.addHandler(logging.NullHandler())

CANONICAL_COLUMNS: list[str] = [
    "event_id",
    "country_name",
    "iso3",
    "hazard_code",
    "hazard_label",
    "hazard_class",
    "metric",
    "unit",
    "as_of_date",
    "value",
    "series_semantics",
    "source",
]


@dataclass(frozen=True)
class PeriodMonths:
    """Container for derived months associated with a quarterly label."""

    label: str
    months: tuple[str, ...]

    @classmethod
    def from_label(cls, label: str) -> "PeriodMonths":
        normalized = label.strip()
        match = re.fullmatch(r"(\d{4})Q([1-4])", normalized)
        if not match:
            if re.match(r"(?i)^(ci|dev|test)", normalized):
                now = dt.datetime.now(dt.timezone.utc)
                quarter = (now.month - 1) // 3 + 1
                alias_label = f"{now.year}Q{quarter}"
                return cls.from_label(alias_label)
            raise ValueError(
                "Invalid --period label %r. Expected format YYYYQ#, e.g. 2025Q4. "
                "Labels starting with 'ci', 'dev', or 'test' are mapped to the "
                "current UTC quarter." % (label,)
            )
        year = int(match.group(1))
        quarter = int(match.group(2))
        start_month = (quarter - 1) * 3 + 1
        months = tuple(
            f"{year}-{month:02d}"
            for month in range(start_month, start_month + 3)
        )
        return cls(label=f"{year}Q{quarter}", months=months)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _read_canonical_dir(canonical_dir: Path) -> pd.DataFrame:
    files: list[Path] = []
    for pattern in ("*.csv", "*.parquet", "*.parq"):
        files.extend(sorted(canonical_dir.glob(pattern)))
    if not files:
        raise FileNotFoundError(
            f"No canonical CSV/Parquet files found in {canonical_dir}"
        )
    frames: list[pd.DataFrame] = []
    for file_path in files:
        if file_path.suffix.lower() == ".csv":
            frames.append(pd.read_csv(file_path))
        else:
            frames.append(pd.read_parquet(file_path))
    combined = pd.concat(frames, ignore_index=True)
    missing = [col for col in CANONICAL_COLUMNS if col not in combined.columns]
    if missing:
        raise ValueError(
            f"Canonical data missing required columns: {missing}"
        )
    extra = [col for col in combined.columns if col not in CANONICAL_COLUMNS]
    if extra:
        combined = combined.drop(columns=extra)
    combined["series_semantics"] = (
        combined["series_semantics"].fillna("stock").astype(str).str.lower()
    )
    combined.loc[
        ~combined["series_semantics"].isin({"stock", "new"}), "series_semantics"
    ] = "stock"
    combined["as_of_date"] = combined["as_of_date"].astype(str)
    combined["value"] = pd.to_numeric(combined["value"], errors="coerce")
    if combined["value"].isna().any():
        raise ValueError("Canonical data contains non-numeric values in 'value'")
    combined["as_of_ts"] = pd.to_datetime(
        combined["as_of_date"], errors="coerce"
    )
    if combined["as_of_ts"].isna().any():
        raise ValueError("Canonical data has invalid 'as_of_date' entries")
    combined["ym"] = combined["as_of_ts"].dt.strftime("%Y-%m")
    return combined


def _ensure_facts_raw_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS facts_raw (
            event_id TEXT,
            country_name TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            hazard_label TEXT,
            hazard_class TEXT,
            metric TEXT,
            unit TEXT,
            as_of_date TEXT,
            value DOUBLE,
            series_semantics TEXT,
            source TEXT
        )
        """
    )


def _insert_dataframe(conn, table: str, frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    cols = list(frame.columns)
    placeholder = ", ".join(cols)
    temp_name = f"tmp_{table}_load"
    conn.register(temp_name, frame)
    try:
        conn.execute(
            f"INSERT INTO {table} ({placeholder}) SELECT {placeholder} FROM {temp_name}"
        )
    finally:
        conn.unregister(temp_name)
    return len(frame)


def _collapse_hazard_codes(codes: pd.Series) -> str:
    values = pd.Series(codes).dropna().astype(str).str.strip()
    uniques: list[str] = []
    for value in values:
        if not value:
            continue
        upper_value = value.upper()
        if upper_value not in uniques:
            uniques.append(upper_value)
    if not uniques:
        return ""
    if len(uniques) == 1:
        return uniques[0]
    return "MULTI"


def _aggregate_deltas_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["iso3", "ym", "metric", "value_new"])

    df = frame.copy()
    df["iso3"] = df["iso3"].astype(str).str.strip().str.upper()
    df["ym"] = df["ym"].astype(str)
    df["metric"] = df["metric"].astype(str)

    if "value_new" not in df.columns and "value" in df.columns:
        df["value_new"] = df["value"]
    df["value_new"] = pd.to_numeric(df["value_new"], errors="coerce").fillna(0.0)

    if "value_stock" in df.columns:
        df["value_stock"] = pd.to_numeric(df["value_stock"], errors="coerce")

    group_cols = ["iso3", "ym", "metric"]

    agg_spec: dict[str, pd.NamedAgg] = {
        "value_new": pd.NamedAgg(column="value_new", aggfunc="sum"),
    }

    if "value_stock" in df.columns:
        agg_spec["value_stock"] = pd.NamedAgg(column="value_stock", aggfunc="max")
    if "source_id" in df.columns:
        agg_spec["source_id"] = pd.NamedAgg(column="source_id", aggfunc="first")
    if "as_of" in df.columns:
        agg_spec["as_of"] = pd.NamedAgg(column="as_of", aggfunc="max")
    if "as_of_date" in df.columns:
        agg_spec["as_of_date"] = pd.NamedAgg(column="as_of_date", aggfunc="max")
    if "publication_date" in df.columns:
        agg_spec["publication_date"] = pd.NamedAgg(column="publication_date", aggfunc="max")
    if "first_observation" in df.columns:
        agg_spec["first_observation"] = pd.NamedAgg(column="first_observation", aggfunc="max")
    if "rebase_flag" in df.columns:
        agg_spec["rebase_flag"] = pd.NamedAgg(column="rebase_flag", aggfunc="max")
    if "delta_negative_clamped" in df.columns:
        agg_spec["delta_negative_clamped"] = pd.NamedAgg(
            column="delta_negative_clamped", aggfunc="max"
        )
    if "hazard_code" in df.columns:
        agg_spec["hazard_code"] = pd.NamedAgg(
            column="hazard_code", aggfunc=_collapse_hazard_codes
        )
    if "series" in df.columns:
        agg_spec["series"] = pd.NamedAgg(column="series", aggfunc="first")

    grouped = (
        df.groupby(group_cols, as_index=False, dropna=False)
        .agg(**agg_spec)
        .sort_values(group_cols)
        .reset_index(drop=True)
    )

    grouped["series_semantics"] = "new"

    defaults: dict[str, object] = {
        "hazard_code": "",
        "value_stock": pd.NA,
        "source_id": pd.NA,
        "as_of": pd.NA,
        "as_of_date": pd.NA,
        "publication_date": pd.NA,
        "first_observation": 0,
        "rebase_flag": 0,
        "delta_negative_clamped": 0,
        "series": pd.NA,
    }
    for column, default in defaults.items():
        if column not in grouped.columns:
            grouped[column] = default

    grouped["value_new"] = pd.to_numeric(grouped["value_new"], errors="coerce").fillna(0.0)
    if "value_stock" in grouped.columns:
        grouped["value_stock"] = pd.to_numeric(
            grouped["value_stock"], errors="coerce"
        )

    ordered_columns = [
        "ym",
        "iso3",
        "hazard_code",
        "metric",
        "value_new",
        "value_stock",
        "series_semantics",
        "as_of",
        "as_of_date",
        "publication_date",
        "source_id",
        "series",
        "first_observation",
        "rebase_flag",
        "delta_negative_clamped",
    ]
    for column in ordered_columns:
        if column not in grouped.columns:
            grouped[column] = pd.NA

    return grouped[ordered_columns]


def _load_into_db(conn, canonical: pd.DataFrame) -> dict[str, int]:
    init_schema(conn)
    _ensure_facts_raw_table(conn)

    raw_counts = _insert_dataframe(conn, "facts_raw", canonical[CANONICAL_COLUMNS])
    LOGGER.info("facts_raw inserted rows: %s", raw_counts)

    stock = canonical[canonical["series_semantics"] == "stock"].copy()
    new = canonical[canonical["series_semantics"] == "new"].copy()

    stock_resolved = pd.DataFrame()
    if not stock.empty:
        stock_resolved = pd.DataFrame({
            "ym": stock["ym"],
            "iso3": stock["iso3"],
            "hazard_code": stock["hazard_code"],
            "hazard_label": stock["hazard_label"],
            "hazard_class": stock["hazard_class"],
            "metric": stock["metric"],
            "series_semantics": "stock",
            "value": stock["value"].astype(float),
            "unit": stock["unit"],
            "as_of": stock["as_of_ts"].dt.date.astype(str),
            "as_of_date": stock["as_of_date"],
            "publication_date": stock["as_of_date"],
            "publisher": stock["source"],
            "source_id": stock["source"],
            "doc_title": stock["source"],
            "event_id": stock["event_id"],
            "confidence": None,
            "provenance_source": stock["source"],
            "provenance_rank": None,
        })
        delete_months(conn, "facts_resolved", stock_resolved["ym"].unique())
        resolved_rows = _insert_dataframe(conn, "facts_resolved", stock_resolved)
    else:
        resolved_rows = 0
    LOGGER.info("facts_resolved inserted rows: %s", resolved_rows)

    deltas_rows = 0
    if not new.empty:
        new_frame = pd.DataFrame({
            "ym": new["ym"],
            "iso3": new["iso3"],
            "hazard_code": new["hazard_code"],
            "metric": new["metric"],
            "value_new": new["value"].astype(float),
            "value_stock": None,
            "series_semantics": "new",
            "as_of": new["as_of_date"],
            "as_of_date": new["as_of_date"],
            "publication_date": new["as_of_date"],
            "source_id": new["source"],
            "first_observation": 0,
            "rebase_flag": 0,
            "delta_negative_clamped": 0,
        })
        aggregated = _aggregate_deltas_frame(new_frame)
        if not aggregated.empty:
            keys = duckdb_io.resolve_upsert_keys("facts_deltas", aggregated)
            result = duckdb_io.upsert_dataframe(
                conn,
                "facts_deltas",
                aggregated,
                keys=keys,
            )
            deltas_rows = int(result.rows_delta)
    LOGGER.info("facts_deltas upserted rows (from canonical new): %s", deltas_rows)

    return {
        "facts_raw": raw_counts,
        "facts_resolved": resolved_rows,
        "facts_deltas": deltas_rows,
    }


def delete_months(conn, table: str, months: Iterable[str]) -> None:
    unique_months = sorted({str(m) for m in months if m})
    for month in unique_months:
        conn.execute(f"DELETE FROM {table} WHERE ym = ?", [month])


def _derive_deltas(
    conn,
    period: PeriodMonths,
    *,
    allow_negatives: bool = True,
) -> int:
    placeholders = ",".join(["?"] * len(period.months))
    query = f"""
        SELECT ym, iso3, hazard_code, hazard_label, hazard_class, metric,
               unit, value, as_of_date, source_id, event_id
        FROM facts_resolved
        WHERE ym IN ({placeholders})
        ORDER BY iso3, hazard_code, metric, unit, source_id, ym
    """
    frame = conn.execute(query, list(period.months)).df()
    if frame.empty:
        LOGGER.info(
            "No facts_resolved rows found for period %s; skipping delta derivation",
            period.label,
        )
        return 0

    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value"])

    records: list[dict[str, object]] = []
    group_cols = ["iso3", "hazard_code", "metric", "unit", "source_id"]

    for _, group in frame.groupby(group_cols, dropna=False):
        group = group.sort_values("ym")
        prev_stock = 0.0
        first = True
        for row in group.to_dict(orient="records"):
            stock_value = float(row["value"])
            delta = stock_value - prev_stock
            if first:
                first = False
            if not allow_negatives and delta < 0:
                delta = 0.0
            record = {
                "ym": row["ym"],
                "iso3": row["iso3"],
                "hazard_code": row["hazard_code"],
                "metric": row["metric"],
                "value_new": float(delta),
                "value_stock": stock_value,
                "series_semantics": "new",
                "as_of": row["as_of_date"],
                "as_of_date": row["as_of_date"],
                "publication_date": row["as_of_date"],
                "source_id": row.get("source_id"),
                "first_observation": 1 if prev_stock == 0.0 else 0,
                "rebase_flag": 0,
                "delta_negative_clamped": 0,
            }
            prev_stock = stock_value
            records.append(record)

    if not records:
        LOGGER.info("No delta records derived for period %s", period.label)
        return 0

    output = pd.DataFrame.from_records(records)
    aggregated = _aggregate_deltas_frame(output)
    if aggregated.empty:
        LOGGER.info("No aggregated delta records for period %s", period.label)
        return 0

    keys = duckdb_io.resolve_upsert_keys("facts_deltas", aggregated)
    result = duckdb_io.upsert_dataframe(
        conn,
        "facts_deltas",
        aggregated,
        keys=keys,
    )
    LOGGER.info("facts_deltas upserted rows (derived): %s", result.rows_delta)
    return int(result.rows_delta)


def _export_parquet(
    conn,
    period: PeriodMonths,
    output_dir: Path,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    placeholders = ",".join(["?"] * len(period.months))

    resolved_query = (
        f"SELECT * FROM facts_resolved WHERE ym IN ({placeholders}) ORDER BY ym"
    )
    deltas_query = (
        f"SELECT * FROM facts_deltas WHERE ym IN ({placeholders}) ORDER BY ym"
    )
    resolved = conn.execute(resolved_query, list(period.months)).df()
    deltas = conn.execute(deltas_query, list(period.months)).df()

    resolved_path = output_dir / "facts_resolved.parquet"
    deltas_path = output_dir / "facts_deltas.parquet"

    resolved.to_parquet(resolved_path, index=False)
    deltas.to_parquet(deltas_path, index=False)

    LOGGER.info("Exported facts_resolved → %s (%s rows)", resolved_path, len(resolved))
    LOGGER.info("Exported facts_deltas → %s (%s rows)", deltas_path, len(deltas))
    return {
        "facts_resolved": len(resolved),
        "facts_deltas": len(deltas),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load canonical facts, derive deltas, and export Parquet artifacts for a period."
        )
    )
    parser.add_argument("--period", required=True, help="Quarter label (e.g. 2025Q3)")
    parser.add_argument(
        "--staging-root",
        default="data/staging",
        help="Root directory containing <period>/canonical data",
    )
    parser.add_argument(
        "--snapshots-root",
        default="data/snapshots",
        help="Destination directory for exported Parquet files",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path or duckdb URL for resolver database (defaults to RESOLVER_DB_URL/RESOLVER_DB_PATH)",
    )
    parser.add_argument(
        "--allow-negatives",
        type=int,
        choices=(0, 1),
        default=1,
        help="Set to 1 to preserve negative derived deltas (default), 0 to clamp",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def _resolve_db_target(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    env_url = os.getenv("RESOLVER_DB_URL")
    if env_url:
        return env_url
    env_path = os.getenv("RESOLVER_DB_PATH")
    if env_path:
        return env_path
    return None


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _configure_logging(args.verbose)

    period = PeriodMonths.from_label(args.period)

    staging_root = Path(args.staging_root).expanduser().resolve()
    canonical_dir = staging_root / args.period / "canonical"
    if not canonical_dir.exists():
        raise FileNotFoundError(
            f"Canonical directory not found at {canonical_dir}"
        )

    db_target = _resolve_db_target(args.db)
    conn, db_path = get_shared_duckdb_conn(db_target)
    LOGGER.info("Using DuckDB at %s", db_path)

    try:
        canonical = _read_canonical_dir(canonical_dir)
        counts = _load_into_db(conn, canonical)
        LOGGER.info("Loaded canonical counts: %s", counts)
        prioritized = resolve_sources(conn)
        LOGGER.info(
            "Applied source resolution: %s prioritized rows", prioritized
        )
        derived = _derive_deltas(
            conn, period, allow_negatives=bool(args.allow_negatives)
        )
        LOGGER.info("Derived %s delta rows for period %s", derived, period.label)
        export_dir = Path(args.snapshots_root).expanduser().resolve() / args.period
        export_counts = _export_parquet(conn, period, export_dir)
        LOGGER.info("Export counts: %s", export_counts)
    finally:
        conn.close()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
