# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""
Diagnostic script: analyse the distribution of FEWS NET Phase 3+ in-need
values in facts_resolved and assess how well the current DR_PHASE3_BUCKETS
boundaries capture the data.

Usage:
    python -m tools.analyze_fewsnet_distribution
"""

from __future__ import annotations

import sys
from typing import Sequence

from pythia.buckets import DR_PHASE3_BUCKETS, BucketSpec
from pythia.db.schema import connect


def _percentile_sql(alias: str, p: int) -> str:
    return f"PERCENTILE_CONT({p / 100.0}) WITHIN GROUP (ORDER BY value) AS {alias}"


def _print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _bucket_case_sql(specs: Sequence[BucketSpec]) -> str:
    lines = []
    for spec in specs:
        if spec.upper is None:
            lines.append(f"WHEN CAST(value AS DOUBLE) >= {float(spec.lower)} THEN {spec.idx}")
        else:
            lines.append(
                f"WHEN CAST(value AS DOUBLE) >= {float(spec.lower)} "
                f"AND CAST(value AS DOUBLE) < {float(spec.upper)} THEN {spec.idx}"
            )
    return "\n            ".join(lines)


def main() -> None:
    print("Connecting to Pythia DuckDB...")
    conn = connect(read_only=True)

    try:
        # ------------------------------------------------------------------
        # 1. Basic stats
        # ------------------------------------------------------------------
        _print_header("FEWS NET Phase 3+ Distribution Analysis")

        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_rows,
                COUNT(DISTINCT iso3) AS n_countries,
                MIN(ym) AS min_ym,
                MAX(ym) AS max_ym
            FROM facts_resolved
            WHERE lower(metric) = 'phase3plus_in_need'
              AND lower(hazard_code) = 'dr'
              AND value IS NOT NULL
            """
        ).fetchone()

        total_rows, n_countries, min_ym, max_ym = row
        print(f"\nTotal rows:       {total_rows:,}")
        print(f"Unique countries: {n_countries}")
        print(f"Date range:       {min_ym} to {max_ym}")

        if total_rows == 0:
            print("\nNo data found. Exiting.")
            return

        # ------------------------------------------------------------------
        # 2. Distribution statistics
        # ------------------------------------------------------------------
        _print_header("Distribution Statistics")

        stats = conn.execute(
            f"""
            SELECT
                MIN(value) AS min_val,
                {_percentile_sql('p10', 10)},
                {_percentile_sql('p25', 25)},
                {_percentile_sql('p50', 50)},
                {_percentile_sql('p75', 75)},
                {_percentile_sql('p90', 90)},
                MAX(value) AS max_val,
                AVG(value) AS mean_val,
                STDDEV(value) AS std_val
            FROM facts_resolved
            WHERE lower(metric) = 'phase3plus_in_need'
              AND lower(hazard_code) = 'dr'
              AND value IS NOT NULL
            """
        ).fetchone()

        labels = ["Min", "P10", "P25", "Median", "P75", "P90", "Max", "Mean", "StdDev"]
        for label, val in zip(labels, stats):
            print(f"  {label:>8s}: {val:>14,.0f}")

        # ------------------------------------------------------------------
        # 3. Histogram by current buckets
        # ------------------------------------------------------------------
        _print_header("Histogram by DR_PHASE3_BUCKETS")

        case_sql = _bucket_case_sql(DR_PHASE3_BUCKETS)

        bucket_rows = conn.execute(
            f"""
            SELECT
                CASE
                    {case_sql}
                    ELSE 0
                END AS bucket_idx,
                COUNT(*) AS n,
                MIN(value) AS bkt_min,
                AVG(value) AS bkt_mean,
                MAX(value) AS bkt_max
            FROM facts_resolved
            WHERE lower(metric) = 'phase3plus_in_need'
              AND lower(hazard_code) = 'dr'
              AND value IS NOT NULL
            GROUP BY bucket_idx
            ORDER BY bucket_idx
            """
        ).fetchall()

        print(f"\n  {'Bucket':<20s} {'Count':>8s} {'%':>7s} {'Min':>14s} {'Mean':>14s} {'Max':>14s}")
        print(f"  {'-' * 20} {'-' * 8} {'-' * 7} {'-' * 14} {'-' * 14} {'-' * 14}")

        spec_by_idx = {s.idx: s for s in DR_PHASE3_BUCKETS}
        for bucket_idx, n, bkt_min, bkt_mean, bkt_max in bucket_rows:
            spec = spec_by_idx.get(bucket_idx)
            label = spec.label if spec else "UNMATCHED"
            pct = 100.0 * n / total_rows
            print(
                f"  {label:<20s} {n:>8,d} {pct:>6.1f}% {bkt_min:>14,.0f} {bkt_mean:>14,.0f} {bkt_max:>14,.0f}"
            )

        # ------------------------------------------------------------------
        # 4. Country-level summary
        # ------------------------------------------------------------------
        _print_header("Country-Level Summary (top 20 by median value)")

        country_rows = conn.execute(
            f"""
            SELECT
                iso3,
                COUNT(*) AS n_obs,
                MIN(value) AS cmin,
                {_percentile_sql('cmedian', 50)},
                MAX(value) AS cmax
            FROM facts_resolved
            WHERE lower(metric) = 'phase3plus_in_need'
              AND lower(hazard_code) = 'dr'
              AND value IS NOT NULL
            GROUP BY iso3
            ORDER BY cmedian DESC
            LIMIT 20
            """
        ).fetchall()

        print(f"\n  {'ISO3':<6s} {'N':>5s} {'Min':>14s} {'Median':>14s} {'Max':>14s}")
        print(f"  {'-' * 6} {'-' * 5} {'-' * 14} {'-' * 14} {'-' * 14}")
        for iso3, n_obs, cmin, cmedian, cmax in country_rows:
            print(f"  {iso3:<6s} {n_obs:>5d} {cmin:>14,.0f} {cmedian:>14,.0f} {cmax:>14,.0f}")

        # ------------------------------------------------------------------
        # 5. Bucket balance assessment
        # ------------------------------------------------------------------
        _print_header("Bucket Balance Assessment")

        bucket_counts = {r[0]: r[1] for r in bucket_rows}
        n_buckets = len(DR_PHASE3_BUCKETS)
        ideal_pct = 100.0 / n_buckets

        print(f"\n  Ideal per-bucket share: {ideal_pct:.1f}%")
        print()

        empty_buckets = []
        imbalanced = []
        for spec in DR_PHASE3_BUCKETS:
            count = bucket_counts.get(spec.idx, 0)
            pct = 100.0 * count / total_rows if total_rows else 0.0
            status = "OK"
            if count == 0:
                status = "EMPTY"
                empty_buckets.append(spec.label)
            elif pct < ideal_pct * 0.25:
                status = "SPARSE"
                imbalanced.append(spec.label)
            elif pct > ideal_pct * 3.0:
                status = "HEAVY"
                imbalanced.append(spec.label)
            print(f"  Bucket {spec.idx} ({spec.label:>10s}): {pct:>6.1f}%  [{status}]")

        print()
        if empty_buckets:
            print(f"  WARNING: Empty buckets: {', '.join(empty_buckets)}")
        if imbalanced:
            print(f"  WARNING: Imbalanced buckets: {', '.join(imbalanced)}")
        if not empty_buckets and not imbalanced:
            print("  All buckets have reasonable coverage.")

    finally:
        conn.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
