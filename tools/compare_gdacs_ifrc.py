# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Compare GDACS vs IFRC Montandon population-affected figures.

Generates a side-by-side report to evaluate whether to switch Pythia's PA
source from IFRC to GDACS for drought, flood, and tropical cyclone hazards.

Usage:
    python -m tools.compare_gdacs_ifrc --backend db
    python -m tools.compare_gdacs_ifrc --backend files
    python -m tools.compare_gdacs_ifrc --hazards DT,FL,TC
    python -m tools.compare_gdacs_ifrc --start 2020-01 --end 2025-12
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DIAGNOSTICS_DIR = ROOT / "diagnostics"
RESOLVER_DB_DEFAULT = ROOT / "resolver" / "db" / "resolver.duckdb"
EXPORTS_DIR = ROOT / "resolver" / "exports"

# Hazard codes we compare (DR=drought, FL=flood, TC=tropical cyclone)
# GDACS uses DR; IFRC Montandon may use DR or DT — we normalise to DR.
_HAZARD_NORMALISE = {"DT": "DR"}

_HAZARD_LABELS = {
    "DR": "Drought",
    "FL": "Flood",
    "TC": "Tropical Cyclone",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_from_db(db_path: str | None = None) -> pd.DataFrame:
    """Load facts_resolved from the Resolver DuckDB."""
    from resolver.db.duckdb_io import get_db

    conn = get_db(db_path)
    query = """
        SELECT iso3, hazard_code, hazard_label, ym, value,
               COALESCE(publisher, '') AS publisher,
               COALESCE(source_type, '') AS source_type,
               metric, series_semantics,
               COALESCE(provenance_source, '') AS provenance_source,
               COALESCE(source_url, '') AS source_url
        FROM facts_resolved
        WHERE hazard_code IN ('DR', 'DT', 'FL', 'TC')
    """
    df = conn.execute(query).fetchdf()
    return df


def _load_from_files() -> pd.DataFrame:
    """Load facts_resolved from CSV exports."""
    candidates = [
        EXPORTS_DIR / "facts.csv",
        EXPORTS_DIR / "resolved.csv",
        EXPORTS_DIR / "resolved_reviewed.csv",
        DIAGNOSTICS_DIR / "ingestion" / "export_preview" / "facts.csv",
    ]
    for path in candidates:
        if path.exists():
            LOG.info("Loading CSV from %s", path)
            df = pd.read_csv(path, dtype=str)
            # Filter to relevant hazards
            if "hazard_code" in df.columns:
                df = df[df["hazard_code"].isin(["DR", "DT", "FL", "TC"])].copy()
            return df
    sys.exit(
        f"No CSV export found. Tried: {', '.join(str(p) for p in candidates)}"
    )


def _classify_source(row: pd.Series) -> str:
    """Classify a row as 'ifrc', 'gdacs', or 'unknown'.

    Primary: check publisher / source_url / provenance_source / source_type.
    Fallback: use metric + hazard heuristic for rows where publisher is NULL
    (pre-fix pipeline wrote GDACS rows with no metadata).
    """
    publisher = str(row.get("publisher", "")).lower()
    source_url = str(row.get("source_url", "")).lower()
    provenance = str(row.get("provenance_source", "")).lower()
    metric = str(row.get("metric", "")).lower()
    source_type = str(row.get("source_type", "")).lower()

    # Primary: check publisher/source fields
    if "gdacs" in publisher or "gdacs" in source_url or "gdacs" in provenance:
        return "gdacs"
    if "satellite_derived" in source_type:
        return "gdacs"
    if any(x in publisher or x in source_url or x in provenance
            for x in ("ifrc", "montandon")):
        return "ifrc"

    # Fallback: GDACS writes in_need for DR/FL/TC with no publisher
    # IFRC writes affected/displaced/fatalities/injured/missing
    hazard = str(row.get("hazard_code", "")).upper()
    if metric == "in_need" and hazard in ("DR", "FL", "TC") and not publisher.strip():
        return "gdacs"
    if metric in ("affected", "displaced", "fatalities", "injured", "missing"):
        return "ifrc"

    return "unknown"


def _load_and_split(
    backend: str,
    db_path: str | None,
    hazards: list[str] | None,
    start_ym: str | None,
    end_ym: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load data and split into IFRC / GDACS DataFrames.

    Returns (ifrc_df, gdacs_df) each with columns:
        iso3, hazard_code, ym, value
    """
    if backend == "db":
        df = _load_from_db(db_path)
    else:
        df = _load_from_files()

    if df.empty:
        sys.exit("No data loaded — check your backend / DB path.")

    # Normalise hazard codes (DT -> DR)
    df["hazard_code"] = df["hazard_code"].replace(_HAZARD_NORMALISE)

    # Ensure value is numeric
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Classify source
    df["_source"] = df.apply(_classify_source, axis=1)

    # Filter hazards
    if hazards:
        df = df[df["hazard_code"].isin(hazards)]

    # Filter date range
    if start_ym:
        df = df[df["ym"] >= start_ym]
    if end_ym:
        df = df[df["ym"] <= end_ym]

    ifrc_df = df[df["_source"] == "ifrc"][["iso3", "hazard_code", "ym", "value"]].copy()
    gdacs_df = df[df["_source"] == "gdacs"][["iso3", "hazard_code", "ym", "value"]].copy()

    # Aggregate: sum values per (iso3, hazard_code, ym) in case of duplicates
    ifrc_df = (
        ifrc_df.groupby(["iso3", "hazard_code", "ym"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "ifrc_value"})
    )
    gdacs_df = (
        gdacs_df.groupby(["iso3", "hazard_code", "ym"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "gdacs_value"})
    )

    LOG.info(
        "IFRC: %d country-months, GDACS: %d country-months",
        len(ifrc_df), len(gdacs_df),
    )
    return ifrc_df, gdacs_df


# ---------------------------------------------------------------------------
# Join & derive columns
# ---------------------------------------------------------------------------


def _load_country_names() -> dict[str, str]:
    """Load iso3 -> country_name mapping from countries.csv."""
    csv_path = ROOT / "resolver" / "data" / "countries.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path, dtype=str)
    return dict(zip(df["iso3"].str.upper(), df["country_name"]))


def _build_merged(
    ifrc_df: pd.DataFrame, gdacs_df: pd.DataFrame,
) -> pd.DataFrame:
    """Full outer join on (iso3, hazard_code, ym) with derived columns."""
    merged = pd.merge(
        ifrc_df, gdacs_df,
        on=["iso3", "hazard_code", "ym"],
        how="outer",
    )

    # Country names
    names = _load_country_names()
    merged["country_name"] = merged["iso3"].map(names).fillna("")

    # Hazard labels
    merged["hazard_label"] = merged["hazard_code"].map(_HAZARD_LABELS).fillna("")

    # Ratio (GDACS / IFRC) where both exist
    both_mask = merged["ifrc_value"].notna() & merged["gdacs_value"].notna()
    merged["ratio"] = np.where(
        both_mask & (merged["ifrc_value"] > 0),
        merged["gdacs_value"] / merged["ifrc_value"],
        np.nan,
    )

    # Coverage flags
    merged["ifrc_only"] = merged["ifrc_value"].notna() & merged["gdacs_value"].isna()
    merged["gdacs_only"] = merged["gdacs_value"].notna() & merged["ifrc_value"].isna()

    # Sort for readability
    merged = merged.sort_values(["hazard_code", "iso3", "ym"]).reset_index(drop=True)

    # Reorder columns
    merged = merged[
        [
            "iso3", "country_name", "hazard_code", "hazard_label", "ym",
            "ifrc_value", "gdacs_value", "ratio", "ifrc_only", "gdacs_only",
        ]
    ]
    return merged


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def _per_group_stats(group: pd.DataFrame) -> dict:
    """Compute summary stats for a subset of the merged DataFrame."""
    both = group[group["ifrc_value"].notna() & group["gdacs_value"].notna()]
    ifrc_only = group["ifrc_only"].sum()
    gdacs_only = group["gdacs_only"].sum()
    n_both = len(both)

    ratios = both["ratio"].dropna()
    median_ratio = float(ratios.median()) if len(ratios) > 0 else np.nan
    mean_ratio = float(ratios.mean()) if len(ratios) > 0 else np.nan
    p25_ratio = float(ratios.quantile(0.25)) if len(ratios) > 0 else np.nan
    p75_ratio = float(ratios.quantile(0.75)) if len(ratios) > 0 else np.nan

    # Correlation
    corr = np.nan
    if n_both >= 3:
        iv = both["ifrc_value"]
        gv = both["gdacs_value"]
        if iv.std() > 0 and gv.std() > 0:
            corr = float(iv.corr(gv))

    return {
        "n_both": int(n_both),
        "n_ifrc_only": int(ifrc_only),
        "n_gdacs_only": int(gdacs_only),
        "median_ratio": median_ratio,
        "mean_ratio": mean_ratio,
        "p25_ratio": p25_ratio,
        "p75_ratio": p75_ratio,
        "correlation": corr,
    }


def _build_summary(merged: pd.DataFrame) -> pd.DataFrame:
    """Build summary stats per hazard and per country (top 20)."""
    rows: list[dict] = []

    # Overall
    overall = _per_group_stats(merged)
    overall["group_type"] = "overall"
    overall["group_key"] = "ALL"
    rows.append(overall)

    # Per hazard
    for haz, hdf in merged.groupby("hazard_code"):
        stats = _per_group_stats(hdf)
        stats["group_type"] = "hazard"
        stats["group_key"] = haz
        rows.append(stats)

    # Per country — top 20 by total data volume
    country_volume = merged.groupby("iso3").size().sort_values(ascending=False)
    top20 = country_volume.head(20).index
    for iso3 in top20:
        cdf = merged[merged["iso3"] == iso3]
        stats = _per_group_stats(cdf)
        stats["group_type"] = "country"
        stats["group_key"] = iso3
        rows.append(stats)

    summary = pd.DataFrame(rows)
    col_order = [
        "group_type", "group_key",
        "n_both", "n_ifrc_only", "n_gdacs_only",
        "median_ratio", "mean_ratio", "p25_ratio", "p75_ratio",
        "correlation",
    ]
    return summary[col_order]


# ---------------------------------------------------------------------------
# Coverage overview (Venn-style)
# ---------------------------------------------------------------------------


def _coverage_overview(merged: pd.DataFrame) -> str:
    """Return a text block summarising Venn-style coverage."""
    both = int((merged["ifrc_value"].notna() & merged["gdacs_value"].notna()).sum())
    ifrc_only = int(merged["ifrc_only"].sum())
    gdacs_only = int(merged["gdacs_only"].sum())
    total_ifrc = both + ifrc_only
    total_gdacs = both + gdacs_only

    lines = [
        "=== Coverage Overview (country-months) ===",
        f"  IFRC total:        {total_ifrc:>8,}",
        f"  GDACS total:       {total_gdacs:>8,}",
        f"  Both sources:      {both:>8,}",
        f"  IFRC only:         {ifrc_only:>8,}",
        f"  GDACS only:        {gdacs_only:>8,}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------


def _format_summary_table(summary: pd.DataFrame) -> str:
    """Format summary DataFrame as an aligned text table."""
    lines: list[str] = []

    def _section(title: str, rows: pd.DataFrame) -> None:
        lines.append(f"\n=== {title} ===")
        lines.append(
            f"{'Key':<12} {'Both':>7} {'IFRC-only':>10} {'GDACS-only':>11} "
            f"{'Med ratio':>10} {'Mean ratio':>11} {'P25':>8} {'P75':>8} {'Corr':>8}"
        )
        lines.append("-" * 100)
        for _, r in rows.iterrows():
            lines.append(
                f"{r['group_key']:<12} {r['n_both']:>7,} {r['n_ifrc_only']:>10,} "
                f"{r['n_gdacs_only']:>11,} "
                f"{r['median_ratio']:>10.2f} {r['mean_ratio']:>11.2f} "
                f"{r['p25_ratio']:>8.2f} {r['p75_ratio']:>8.2f} "
                f"{r['correlation']:>8.3f}"
            )

    overall = summary[summary["group_type"] == "overall"]
    hazard = summary[summary["group_type"] == "hazard"]
    country = summary[summary["group_type"] == "country"]

    _section("Overall", overall)
    _section("Per Hazard", hazard)
    _section("Per Country (top 20 by data volume)", country)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Key findings
# ---------------------------------------------------------------------------


def _key_findings(merged: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Generate a short textual summary answering the key comparison questions."""
    lines = ["\n=== Key Findings ==="]

    # 1. Temporal coverage per hazard
    hazard_rows = summary[summary["group_type"] == "hazard"]
    lines.append("\n1. Temporal coverage by hazard:")
    for _, r in hazard_rows.iterrows():
        total_ifrc = r["n_both"] + r["n_ifrc_only"]
        total_gdacs = r["n_both"] + r["n_gdacs_only"]
        label = _HAZARD_LABELS.get(r["group_key"], r["group_key"])
        lines.append(
            f"   {label} ({r['group_key']}): IFRC={total_ifrc:,} GDACS={total_gdacs:,} "
            f"country-months (overlap={r['n_both']:,})"
        )

    # 2. Ratio stability
    lines.append("\n2. GDACS/IFRC ratio (where both exist):")
    for _, r in hazard_rows.iterrows():
        label = _HAZARD_LABELS.get(r["group_key"], r["group_key"])
        if np.isnan(r["median_ratio"]):
            lines.append(f"   {label}: no overlapping data")
        else:
            iqr = r["p75_ratio"] - r["p25_ratio"]
            stability = "stable" if iqr < 2.0 else "variable" if iqr < 10.0 else "highly variable"
            lines.append(
                f"   {label}: median={r['median_ratio']:.2f}x, IQR={iqr:.2f} ({stability})"
            )

    # 3. Blind spots — countries where one source has data but not the other
    ifrc_only_countries = (
        merged[merged["ifrc_only"]]
        .groupby("iso3")
        .size()
        .sort_values(ascending=False)
    )
    gdacs_only_countries = (
        merged[merged["gdacs_only"]]
        .groupby("iso3")
        .size()
        .sort_values(ascending=False)
    )
    lines.append(f"\n3. Blind spots:")
    lines.append(
        f"   Countries with IFRC data but no GDACS: {len(ifrc_only_countries)}"
    )
    if len(ifrc_only_countries) > 0:
        top5 = ifrc_only_countries.head(5)
        lines.append(
            f"     Top 5: {', '.join(f'{c} ({n})' for c, n in top5.items())}"
        )
    lines.append(
        f"   Countries with GDACS data but no IFRC: {len(gdacs_only_countries)}"
    )
    if len(gdacs_only_countries) > 0:
        top5 = gdacs_only_countries.head(5)
        lines.append(
            f"     Top 5: {', '.join(f'{c} ({n})' for c, n in top5.items())}"
        )

    # 4. Drought-specific coverage
    dr_merged = merged[merged["hazard_code"] == "DR"]
    if not dr_merged.empty:
        dr_gdacs_only = dr_merged[dr_merged["gdacs_only"]]
        dr_ifrc_only = dr_merged[dr_merged["ifrc_only"]]
        lines.append(f"\n4. Drought coverage gap analysis:")
        lines.append(
            f"   GDACS fills {len(dr_gdacs_only):,} country-months "
            f"where IFRC has no DR data"
        )
        lines.append(
            f"   IFRC has {len(dr_ifrc_only):,} country-months "
            f"where GDACS has no DR data"
        )
        dr_gdacs_countries = dr_gdacs_only["iso3"].nunique()
        lines.append(
            f"   GDACS covers {dr_gdacs_countries} unique countries "
            f"with no IFRC DR coverage"
        )
    else:
        lines.append("\n4. No drought data found in either source.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare GDACS vs IFRC Montandon population-affected figures",
    )
    parser.add_argument(
        "--backend",
        choices=["db", "files"],
        default="db",
        help="Data source: 'db' for DuckDB, 'files' for CSV exports (default: db)",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to resolver DuckDB (overrides RESOLVER_DB_URL)",
    )
    parser.add_argument(
        "--hazards",
        default=None,
        help="Comma-separated hazard codes to include (e.g. DR,FL,TC)",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start year-month inclusive (e.g. 2020-01)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End year-month inclusive (e.g. 2025-12)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    hazards = [h.strip().upper() for h in args.hazards.split(",")] if args.hazards else None
    # Normalise hazard filter (DT -> DR)
    if hazards:
        hazards = [_HAZARD_NORMALISE.get(h, h) for h in hazards]

    # Load and split
    ifrc_df, gdacs_df = _load_and_split(
        backend=args.backend,
        db_path=args.db_path,
        hazards=hazards,
        start_ym=args.start,
        end_ym=args.end,
    )

    if ifrc_df.empty and gdacs_df.empty:
        print("No IFRC or GDACS data found. Nothing to compare.")
        return

    # Build merged dataset
    merged = _build_merged(ifrc_df, gdacs_df)
    print(f"\nMerged dataset: {len(merged):,} rows")

    # Summary stats
    summary = _build_summary(merged)

    # Print to stdout
    print(_coverage_overview(merged))
    print(_format_summary_table(summary))
    print(_key_findings(merged, summary))

    # Write outputs
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    merged_path = DIAGNOSTICS_DIR / "gdacs_ifrc_comparison.csv"
    merged.to_csv(merged_path, index=False)
    print(f"\nFull comparison written to {merged_path}")

    summary_path = DIAGNOSTICS_DIR / "gdacs_ifrc_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary stats written to {summary_path}")


if __name__ == "__main__":
    main()
