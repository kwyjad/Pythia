# Pythia / Copyright (c) 2025 Kevin Wyjad
"""
Seasonal TC Forecast Runner
============================
Orchestrates all three scrapers (TSR, NOAA CPC, BoM) to produce a single
unified JSON file of seasonal tropical cyclone forecasts for all covered basins.

This is the entry point for Pythia's TC prompt grounding pipeline.

Output: seasonal_tc_forecasts.json — an array of forecast objects, each with
a to_prompt_context() text block suitable for injection into TC question prompts.

Usage:
    python seasonal_tc_runner.py                        # run all scrapers
    python seasonal_tc_runner.py --year 2026            # target a specific season year
    python seasonal_tc_runner.py --output path/to/out   # custom output path
    python seasonal_tc_runner.py --dry-run              # show what would be fetched
    python seasonal_tc_runner.py --prompt-context       # also write prompt_context.txt

Directory layout expected:
    seasonal_tc/
        seasonal_tc_runner.py       <-- this file
        tsr_seasonal_extractor.py
        noaa_cpc_scraper.py
        bom_scraper.py
        output/
            seasonal_tc_forecasts.json
            prompt_context.txt       (optional, human-readable)
"""

import json
import logging
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Ensure sibling modules are importable
sys.path.insert(0, str(Path(__file__).parent))

from tsr_seasonal_extractor import (
    discover_and_extract as tsr_discover,
    process_url as tsr_process_url,
    SeasonalForecast as TSRForecast,
)
from noaa_cpc_scraper import (
    process_known_urls as noaa_process,
    auto_extract as noaa_auto_extract,
    KNOWN_URLS as NOAA_KNOWN_URLS,
)
from bom_scraper import (
    process_all as bom_process,
    extract_australian_outlook as bom_aus_extract,
    extract_south_pacific_outlook as bom_sp_extract,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Basin metadata (for coverage reporting)
# ---------------------------------------------------------------------------

ALL_BASINS = {
    "ATL": "North Atlantic",
    "ENP": "Eastern North Pacific",
    "CP":  "Central Pacific",
    "NWP": "Northwest Pacific",
    "AUS": "Australian Region",
    "SP":  "South Pacific",
    "SWI": "South-West Indian Ocean",
    "NIO": "North Indian Ocean",
}


# ---------------------------------------------------------------------------
# Unified output format
# ---------------------------------------------------------------------------

def normalize_forecast(raw: dict, source_module: str) -> dict:
    """
    Normalize a forecast dict from any scraper into a common schema.
    All scrapers already produce similar structures, so this is mostly
    about ensuring consistent field names and adding a prompt_context field.
    """
    normalized = dict(raw)
    normalized["_source_module"] = source_module

    # Generate prompt context if not already present
    if "prompt_context" not in normalized:
        # Reconstruct from the source object's method if possible
        # (we store it during collection below)
        pass

    return normalized


# ---------------------------------------------------------------------------
# Collection orchestrator
# ---------------------------------------------------------------------------

def collect_tsr(year: int) -> list[dict]:
    """Collect TSR forecasts for a given season year."""
    logger.info(f"=== TSR: Discovering forecasts for {year} ===")
    results = []
    try:
        forecasts = tsr_discover(year)
        for f in forecasts:
            d = f.to_dict()
            d["prompt_context"] = f.to_prompt_context()
            d["_source_module"] = "tsr_seasonal_extractor"
            results.append(d)
        logger.info(f"  TSR: collected {len(results)} forecasts")
    except Exception as e:
        logger.error(f"  TSR failed: {e}")
    return results


def collect_noaa(year: int) -> list[dict]:
    """Collect NOAA CPC forecasts for a given season year."""
    logger.info(f"=== NOAA CPC: Processing known URLs for {year} ===")
    results = []
    try:
        forecasts = noaa_process(year)
        for f in forecasts:
            d = f.to_dict()
            d["prompt_context"] = f.to_prompt_context()
            d["_source_module"] = "noaa_cpc_scraper"
            results.append(d)
        logger.info(f"  NOAA: collected {len(results)} forecasts")
    except Exception as e:
        logger.error(f"  NOAA failed: {e}")
    return results


def collect_bom() -> list[dict]:
    """Collect BoM forecasts (current season)."""
    logger.info("=== BoM: Fetching current outlooks ===")
    results = []
    try:
        forecasts = bom_process(fetch_live=True)
        for f in forecasts:
            d = f.to_dict()
            d["prompt_context"] = f.to_prompt_context()
            d["_source_module"] = "bom_scraper"
            results.append(d)
        logger.info(f"  BoM: collected {len(results)} forecasts")
    except Exception as e:
        logger.error(f"  BoM failed: {e}")
    return results


def collect_all(year: int) -> list[dict]:
    """Run all scrapers and combine results."""
    all_forecasts = []

    # TSR (ATL + NWP, possibly AUS)
    all_forecasts.extend(collect_tsr(year))

    # NOAA CPC (ATL + ENP + CP)
    all_forecasts.extend(collect_noaa(year))

    # BoM (AUS + SP)
    all_forecasts.extend(collect_bom())

    return all_forecasts


# ---------------------------------------------------------------------------
# Fetch, store, and cache (programmatic entry point)
# ---------------------------------------------------------------------------

def fetch_and_store_seasonal_tc() -> bool:
    """Collect seasonal TC outlooks, store them in DuckDB, and cache per-country context.

    This is the programmatic entry point used by the ingest pipeline.
    It calls :func:`collect_all` and :func:`deduplicate`, stores the raw
    outlooks via :func:`store_seasonal_tc_outlooks`, and pre-generates
    per-country context for every country in ``COUNTRY_TO_BASINS``.

    Returns True on success, False on failure.  Non-fatal — catches all
    exceptions so callers are never disrupted.
    """
    try:
        from horizon_scanner.seasonal_tc import (
            COUNTRY_TO_BASINS,
            store_seasonal_tc_outlooks,
            store_seasonal_tc_context_cache,
        )

        year = datetime.utcnow().year
        logger.info("fetch_and_store_seasonal_tc: collecting for year %d", year)

        all_forecasts = collect_all(year)
        all_forecasts = deduplicate(all_forecasts)

        if not all_forecasts:
            logger.warning("fetch_and_store_seasonal_tc: no forecasts collected")
            return False

        # Store raw outlooks
        stored = store_seasonal_tc_outlooks(all_forecasts)
        logger.info("fetch_and_store_seasonal_tc: stored %d outlooks", stored)

        # Generate and cache per-country context
        cached = 0
        for iso3, basins in COUNTRY_TO_BASINS.items():
            blocks = []
            seen = set()
            for f in all_forecasts:
                basin = f.get("basin", "")
                ctx = f.get("prompt_context", "")
                if basin in basins and ctx:
                    key = (f.get("source", ""), basin, f.get("forecast_type", ""))
                    if key not in seen:
                        seen.add(key)
                        blocks.append(ctx)
            if blocks:
                text = "\n\n".join(blocks)
                if store_seasonal_tc_context_cache(iso3, text):
                    cached += 1

        logger.info(
            "fetch_and_store_seasonal_tc: cached context for %d / %d countries",
            cached, len(COUNTRY_TO_BASINS),
        )
        return True
    except Exception as exc:
        logger.error("fetch_and_store_seasonal_tc failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Deduplication and selection
# ---------------------------------------------------------------------------

def deduplicate(forecasts: list[dict]) -> list[dict]:
    """
    When multiple sources cover the same basin, keep the most recent
    forecast from each source (don't deduplicate across sources — having
    both TSR and NOAA for the Atlantic is valuable).

    Dedup key: (source, basin, forecast_type)
    Keep: the one with the latest issue_date.
    """
    seen = {}
    for f in forecasts:
        key = (f.get("source", ""), f.get("basin", ""), f.get("forecast_type", ""))
        existing = seen.get(key)
        if existing is None:
            seen[key] = f
        else:
            # Keep the more recent one
            if (f.get("issue_date", "") or "") > (existing.get("issue_date", "") or ""):
                seen[key] = f

    return list(seen.values())


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def print_coverage(forecasts: list[dict]):
    """Print a basin coverage summary."""
    covered = {}
    for f in forecasts:
        basin = f.get("basin", "?")
        source = f.get("source", "?")
        ftype = f.get("forecast_type", "?")
        if basin not in covered:
            covered[basin] = []
        covered[basin].append(f"{source} ({ftype})")

    logger.info("\n=== BASIN COVERAGE ===")
    for code, name in ALL_BASINS.items():
        sources = covered.get(code, [])
        if sources:
            logger.info(f"  {code:4s} {name:30s} ✓  {', '.join(sources)}")
        else:
            logger.info(f"  {code:4s} {name:30s} ✗  (no data)")
    logger.info("")


# ---------------------------------------------------------------------------
# Prompt context generation
# ---------------------------------------------------------------------------

def generate_prompt_context_file(forecasts: list[dict]) -> str:
    """
    Generate a single text file with all prompt context blocks,
    organized by basin. This is what gets injected into TC question prompts.
    """
    lines = [
        "# Seasonal Tropical Cyclone Forecasts",
        f"# Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"# Sources: TSR, NOAA CPC, BoM",
        f"# Forecasts: {len(forecasts)}",
        "",
    ]

    # Group by basin
    by_basin = {}
    for f in forecasts:
        basin = f.get("basin", "UNKNOWN")
        if basin not in by_basin:
            by_basin[basin] = []
        by_basin[basin].append(f)

    # Output in a logical basin order
    basin_order = ["ATL", "ENP", "CP", "NWP", "AUS", "SP", "SWI", "NIO"]
    for basin_code in basin_order:
        if basin_code in by_basin:
            for f in by_basin[basin_code]:
                ctx = f.get("prompt_context", "")
                if ctx:
                    lines.append(ctx)
                    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect seasonal TC forecasts from all sources"
    )
    parser.add_argument(
        "--year", type=int, default=datetime.utcnow().year,
        help="Target season year (default: current year)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: ./output/)"
    )
    parser.add_argument(
        "--prompt-context", action="store_true",
        help="Also generate prompt_context.txt"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be fetched without actually fetching"
    )
    parser.add_argument(
        "--json-only", action="store_true",
        help="Only output JSON to stdout, no logging"
    )
    args = parser.parse_args()

    if args.json_only:
        logging.disable(logging.CRITICAL)

    year = args.year
    output_dir = Path(args.output) if args.output else Path(__file__).parent / "output"

    if args.dry_run:
        logger.info(f"DRY RUN for season year {year}")
        logger.info(f"Would fetch:")
        logger.info(f"  TSR: discover URLs for {year} (ATL extended range Dec {year-1}, ATL Apr-Aug {year}, NWP Apr-Aug {year})")
        noaa_urls = NOAA_KNOWN_URLS.get(year, {})
        for k, v in noaa_urls.items():
            logger.info(f"  NOAA: {k} -> {v}")
        if not noaa_urls:
            logger.info(f"  NOAA: no known URLs for {year} — add them to KNOWN_URLS in noaa_cpc_scraper.py")
        logger.info(f"  BoM: current Australian + South Pacific outlook pages")
        logger.info(f"Output: {output_dir}/")
        return

    # Collect
    logger.info(f"Collecting seasonal TC forecasts for {year}")
    all_forecasts = collect_all(year)

    # Deduplicate
    all_forecasts = deduplicate(all_forecasts)

    # Coverage report
    print_coverage(all_forecasts)

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "seasonal_tc_forecasts.json"
    # Strip internal fields for clean output
    clean = [{k: v for k, v in f.items() if not k.startswith("_")} for f in all_forecasts]
    json_path.write_text(json.dumps(clean, indent=2, ensure_ascii=False))
    logger.info(f"Wrote {len(clean)} forecasts to {json_path}")

    if args.prompt_context:
        ctx_path = output_dir / "prompt_context.txt"
        ctx_text = generate_prompt_context_file(all_forecasts)
        ctx_path.write_text(ctx_text)
        logger.info(f"Wrote prompt context to {ctx_path}")

    if args.json_only:
        print(json.dumps(clean, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
