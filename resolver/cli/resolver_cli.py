#!/usr/bin/env python3
"""
resolver_cli.py — answer:
  "By <DATE>, how many people <METRIC> due to <HAZARD> in <COUNTRY>?"

Examples:
  python resolver/cli/resolver_cli.py \
    --country "Philippines" \
    --hazard "Tropical Cyclone" \
    --cutoff 2025-09-30

  python resolver/cli/resolver_cli.py \
    --iso3 PHL --hazard_code TC --cutoff 2025-09-30

Behavior:
  - If cutoff month < current month: read snapshots/YYYY-MM/facts.parquet (preferred)
    - If snapshot not found, optionally fall back to exports/resolved(_reviewed).csv (warn)
  - If cutoff is current month: prefer exports/resolved_reviewed.csv, else exports/resolved.csv
  - Applies selection rules already enforced upstream (precedence engine & review)
  - Defaults to monthly NEW deltas when available (`--series new`); use `--series stock` for totals. Missing deltas return a no-data error unless `RESOLVER_ALLOW_SERIES_FALLBACK=1` permits fallback to stocks.
  - Returns a single record (value + citation) or explains why none exists
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from pydantic import ValidationError

from resolver.api.batch_models import ResolveQuery, ResolveResponseRow
from resolver.utils.json_sanitize import json_default

try:
    import pandas as pd
except ImportError:  # pragma: no cover - guidance for operators
    print("Please 'pip install pandas pyarrow' to run resolver_cli.", file=sys.stderr)
    sys.exit(2)

from resolver.db import duckdb_io  # DEBUG
from resolver.query.selectors import (
    VALID_BACKENDS,
    normalize_backend,
    resolve_point,
    ym_from_cutoff,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

COUNTRIES_CSV = DATA / "countries.csv"
SHOCKS_CSV = DATA / "shocks.csv"

def load_registries() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load registries used for lookup and add normalized helper columns."""
    countries = pd.read_csv(COUNTRIES_CSV, dtype=str).fillna("")
    shocks = pd.read_csv(SHOCKS_CSV, dtype=str).fillna("")

    countries["country_norm"] = countries["country_name"].str.strip().str.lower()
    shocks["hazard_norm"] = shocks["hazard_label"].str.strip().str.lower()
    return countries, shocks


def resolve_country(
    countries: pd.DataFrame, country: Optional[str], iso3: Optional[str]
) -> Tuple[str, str]:
    """Return canonical (name, iso3) pair from either user input."""
    if iso3:
        iso3_code = iso3.strip().upper()
        match = countries[countries["iso3"] == iso3_code]
        if not match.empty:
            row = match.iloc[0]
            return row["country_name"], iso3_code

    if country:
        query = country.strip().lower()
        match = countries[countries["country_norm"] == query]
        if not match.empty:
            row = match.iloc[0]
            return row["country_name"], row["iso3"]

    raise SystemExit(
        "Could not resolve country; provide --country or --iso3 matching the registry."
    )


def resolve_hazard(
    shocks: pd.DataFrame, hazard: Optional[str], hazard_code: Optional[str]
) -> Tuple[str, str, str]:
    """Return canonical (label, code, class) triplet from label or code."""
    if hazard_code:
        hz_code = hazard_code.strip().upper()
        match = shocks[shocks["hazard_code"] == hz_code]
        if not match.empty:
            row = match.iloc[0]
            return row["hazard_label"], row["hazard_code"], row["hazard_class"]

    if hazard:
        query = hazard.strip().lower()
        match = shocks[shocks["hazard_norm"] == query]
        if not match.empty:
            row = match.iloc[0]
            return row["hazard_label"], row["hazard_code"], row["hazard_class"]

    raise SystemExit(
        "Could not resolve hazard; provide --hazard or --hazard_code matching the registry."
    )


def _map_backend(value: Optional[str], *, default: str) -> str:
    if not value:
        return default
    backend = str(value).strip().lower()
    if backend == "csv":
        backend = "files"
    if backend not in VALID_BACKENDS:
        return default
    return backend


def _read_batch_queries(path: Path) -> List[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, dtype=str).fillna("")
        return df.to_dict(orient="records")
    if suffix in {".json", ".jsonl"}:
        with path.open("r", encoding="utf-8") as handle:
            if suffix == ".jsonl":
                return [json.loads(line) for line in handle if line.strip()]
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        raise ValueError("JSON batch input must be a list of query objects")
    raise ValueError("Batch input must be .csv, .json, or .jsonl")


def _write_batch_output(rows: Iterable[dict], path: Path) -> None:
    records = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        pd.DataFrame(records).to_csv(path, index=False)
        return
    if suffix in {".json", ".jsonl"}:
        with path.open("w", encoding="utf-8") as handle:
            if suffix == ".json":
                json.dump(records, handle, default=json_default, ensure_ascii=False, indent=2)
            else:
                for row in records:
                    handle.write(json.dumps(row, default=json_default, ensure_ascii=False) + "\n")
        return
    raise ValueError("Batch output must be .csv, .json, or .jsonl")


def _resolve_batch_query(
    raw_query: dict,
    *,
    countries: pd.DataFrame,
    shocks: pd.DataFrame,
    default_backend: str,
    default_series: Optional[str],
) -> Optional[dict]:
    payload = dict(raw_query)
    if default_series and not payload.get("series"):
        payload["series"] = default_series
    payload["backend"] = _map_backend(payload.get("backend"), default=default_backend)

    try:
        query = ResolveQuery.parse_obj(payload)
    except ValidationError:
        return None

    try:
        country_name, iso3_code = resolve_country(countries, query.country, query.iso3)
        hazard_label, hazard_code, hazard_class = resolve_hazard(
            shocks, query.hazard, query.hazard_code
        )
    except SystemExit:
        return None

    result = resolve_point(
        iso3=iso3_code,
        hazard_code=hazard_code,
        cutoff=query.cutoff,
        series=query.series,
        metric="in_need",
        backend=query.backend or default_backend,
    )

    if not result:
        return None

    result.setdefault("country_name", country_name)
    result.setdefault("hazard_label", hazard_label)
    result.setdefault("hazard_class", hazard_class)
    result.setdefault("cutoff", query.cutoff)
    result.setdefault("series_requested", query.series)

    return ResolveResponseRow.parse_obj(result).dict()


def run_batch_resolve(
    input_path: Path,
    output_path: Path,
    *,
    backend: str,
    series: Optional[str],
    max_workers: int,
) -> None:
    queries = _read_batch_queries(input_path)
    if not queries:
        _write_batch_output([], output_path)
        return

    countries, shocks = load_registries()
    backend_choice = _map_backend(backend, default=normalize_backend(None, default="files"))

    def worker(query: dict) -> Optional[dict]:
        return _resolve_batch_query(
            query,
            countries=countries,
            shocks=shocks,
            default_backend=backend_choice,
            default_series=series,
        )

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(queries)))) as executor:
        results = list(filter(None, executor.map(worker, queries)))

    _write_batch_output(results, output_path)


def _run_single(args: List[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", help="Country name (as in countries.csv)")
    parser.add_argument("--iso3", help="Country ISO3 code")
    parser.add_argument("--hazard", help="Hazard label (as in shocks.csv)")
    parser.add_argument("--hazard_code", help="Hazard code (as in shocks.csv)")
    parser.add_argument("--cutoff", required=True, help="Cut-off date YYYY-MM-DD (23:59 Europe/Istanbul)")
    parser.add_argument(
        "--series",
        choices=["new", "stock"],
        default="new",
        help="Return monthly NEW deltas (default) or STOCK totals.",
    )
    parser.add_argument("--json_only", action="store_true", help="Print JSON only (no human summary)")
    default_backend = normalize_backend(
        os.environ.get("RESOLVER_CLI_BACKEND"), default="files"
    )
    parser.add_argument(
        "--backend",
        choices=["files", "db", "auto"],
        default=default_backend,
        help=(
            "Backend to use for data access: files (snapshots/exports), db (DuckDB), or auto "
            "(prefer db when available). Default is files; override with RESOLVER_CLI_BACKEND."
        ),
    )
    args = parser.parse_args(args)

    countries, shocks = load_registries()
    country_name, iso3 = resolve_country(countries, args.country, args.iso3)
    hazard_label, hazard_code, hazard_class = resolve_hazard(shocks, args.hazard, args.hazard_code)

    series_requested = args.series
    backend_choice = args.backend

    print(
        f"DBG resolver_cli backend={backend_choice} series={series_requested} cutoff={args.cutoff}",
        file=sys.stderr,
        flush=True,
    )  # DEBUG

    if series_requested == "new" and backend_choice in {"db", "auto"}:
        print(
            "DBG resolver_cli path: load_series_from_db(new, db)",
            file=sys.stderr,
            flush=True,
        )  # DEBUG

    def emit_no_data(message: str) -> None:
        payload = {
            "ok": False,
            "reason": message,
            "iso3": iso3,
            "hazard_code": hazard_code,
            "cutoff": args.cutoff,
            "series_requested": series_requested,
        }
        print(
            json.dumps(payload, default=json_default, ensure_ascii=False),
            flush=True,
        )
        if not args.json_only:
            print("\n" + message, file=sys.stderr)
        sys.exit(1)

    result = resolve_point(
        iso3=iso3,
        hazard_code=hazard_code,
        cutoff=args.cutoff,
        series=series_requested,
        metric="in_need",
        backend=backend_choice,
    )

    if not result:
        try:  # DEBUG
            if backend_choice in {"db", "auto"}:
                db_url = os.environ.get("RESOLVER_DB_URL")
                if db_url:
                    conn = duckdb_io.get_db(db_url)
                    ym = ym_from_cutoff(args.cutoff)
                    c_all = conn.execute("SELECT COUNT(*) FROM facts_deltas").fetchone()[0]
                    c_ym = conn.execute(
                        "SELECT COUNT(*) FROM facts_deltas WHERE ym = ?",
                        [ym],
                    ).fetchone()[0]
                    c_key = conn.execute(
                        "SELECT COUNT(*) FROM facts_deltas WHERE ym = ? AND iso3 = ? AND hazard_code = ?",
                        [ym, iso3, hazard_code],
                    ).fetchone()[0]
                    print(
                        "DBG resolver_cli no-data diag: "
                        f"ym={ym} iso3={iso3} hazard={hazard_code} "
                        f"counts total={c_all} ym={c_ym} ym+keys={c_key}",
                        file=sys.stderr,
                        flush=True,
                    )  # DEBUG
        except Exception as exc:  # DEBUG
            print(
                f"DBG resolver_cli no-data diag ERROR: {exc}",
                file=sys.stderr,
                flush=True,
            )  # DEBUG
        dataset_hint = (
            "DuckDB table facts_deltas (value_new)"
            if series_requested == "new"
            else "DuckDB table facts_resolved (value)"
        )
        message = (
            "No data found for "
            f"iso3={iso3}, hazard={hazard_code}, series={series_requested} at cutoff {args.cutoff} "
            f"(backend {backend_choice}; checked {dataset_hint})."
        )
        emit_no_data(message)

    row_series = (
        str(result.get("series_returned", series_requested)).strip().lower() or series_requested
    )
    ym_value = result.get("ym", ym_from_cutoff(args.cutoff))
    output = {
        "ok": True,
        "iso3": iso3,
        "country_name": country_name,
        "hazard_code": hazard_code,
        "hazard_label": hazard_label,
        "hazard_class": hazard_class,
        "cutoff": args.cutoff,
        "metric": result.get("metric", ""),
        "unit": result.get("unit", "persons"),
        "value": result.get("value", ""),
        "as_of_date": result.get("as_of_date", ""),
        "publication_date": result.get("publication_date", ""),
        "publisher": result.get("publisher", ""),
        "source_type": result.get("source_type", ""),
        "source_url": result.get("source_url", ""),
        "doc_title": result.get("doc_title", ""),
        "definition_text": result.get("definition_text", ""),
        "precedence_tier": result.get("precedence_tier", ""),
        "event_id": result.get("event_id", ""),
        "confidence": result.get("confidence", ""),
        "proxy_for": result.get("proxy_for", ""),
        "source": result.get("source", ""),
        "source_dataset": result.get("source_dataset", ""),
        "source_id": result.get("source_id", ""),
        "series_semantics": row_series,
        "series_requested": result.get("series_requested", series_requested),
        "series_returned": row_series,
        "ym": ym_value,
    }
    if result.get("fallback_used"):
        output["fallback_used"] = True

    print(json.dumps(output, default=json_default, ensure_ascii=False), flush=True)

    if args.json_only:
        return

    print("\n=== Resolver ===")
    print(f"{country_name} ({iso3}) — {hazard_label} [{hazard_code}]")
    value = output["value"]
    metric = output["metric"] or "value"
    unit = output["unit"]
    try:
        human_value = f"{int(value):,}"
    except Exception:
        human_value = f"{value}"
    print(f"By {args.cutoff}: {human_value} {metric.replace('_', ' ')} ({unit})")
    if output["series_returned"] != output["series_requested"]:
        print(
            f"Series returned: {output['series_returned']} (requested {output['series_requested']})"
        )
    else:
        print(f"Series: {output['series_returned']}")
    print("— source —")
    print(f"{output['publisher']} | as-of {output['as_of_date']} | pub {output['publication_date']}")
    if output["source_url"]:
        print(output["source_url"])
    if output["definition_text"]:
        definition = output["definition_text"]
        trimmed = definition[:200]
        print(f"def: {trimmed}{'...' if len(definition) > 200 else ''}")
    if output["proxy_for"]:
        print(f"(proxy for {output['proxy_for']})")
    if output["precedence_tier"]:
        print(f"tier: {output['precedence_tier']}")
    if output["confidence"]:
        print(f"confidence: {output['confidence']}")
    dataset_label = output.get("source_dataset")
    detail = f" ({dataset_label})" if dataset_label else ""
    print(f"[source bucket: {output['source']}{detail}]")


def main() -> None:
    _run_single(sys.argv[1:])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "batch-resolve":
        batch_parser = argparse.ArgumentParser(prog="resolver_cli.py batch-resolve")
        batch_parser.add_argument("--in", dest="input_path", required=True, help="Input file (.csv/.json/.jsonl)")
        batch_parser.add_argument(
            "--out",
            dest="output_path",
            required=True,
            help="Output file (.csv/.json/.jsonl)",
        )
        default_backend = normalize_backend(
            os.environ.get("RESOLVER_CLI_BACKEND"), default="files"
        )
        batch_parser.add_argument(
            "--backend",
            choices=["db", "csv", "files", "auto"],
            default=default_backend,
            help="Backend to use: db, csv/files, or auto.",
        )
        batch_parser.add_argument(
            "--series",
            choices=["new", "stock"],
            default=None,
            help="Default series when a query omits the column.",
        )
        batch_parser.add_argument(
            "--workers",
            type=int,
            default=os.cpu_count() or 4,
            help="Maximum worker threads for resolving queries.",
        )
        batch_args = batch_parser.parse_args(sys.argv[2:])
        run_batch_resolve(
            Path(batch_args.input_path),
            Path(batch_args.output_path),
            backend=batch_args.backend,
            series=batch_args.series,
            max_workers=batch_args.workers,
        )
    else:
        main()
