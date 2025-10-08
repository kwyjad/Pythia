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
from pathlib import Path
from typing import Optional, Tuple
from resolver.utils.json_sanitize import json_default

try:
    import pandas as pd
except ImportError:  # pragma: no cover - guidance for operators
    print("Please 'pip install pandas pyarrow' to run resolver_cli.", file=sys.stderr)
    sys.exit(2)

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




def main() -> None:
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
    args = parser.parse_args()

    countries, shocks = load_registries()
    country_name, iso3 = resolve_country(countries, args.country, args.iso3)
    hazard_label, hazard_code, hazard_class = resolve_hazard(shocks, args.hazard, args.hazard_code)

    series_requested = args.series
    backend_choice = args.backend

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
        message = (
            "No data found for "
            f"iso3={iso3}, hazard={hazard_code}, series={series_requested} at cutoff {args.cutoff} "
            f"(backend {backend_choice})."
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


if __name__ == "__main__":
    main()
