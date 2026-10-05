# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Read-only probe: what is the FEWS NET ``value`` column?

The connector stores one number per country-month, ``value``. The
ipcpopulationsize feed also carries ``low_value``, ``high_value`` and
``population_range``, and the figures cluster on round numbers (1,000,000
appears in hundreds of rows). If ``value`` is one end of a published RANGE,
scoring it as a point puts every outcome on a bucket edge for no reason.

This answers the question from the feed itself: for every row it reports
whether ``value`` equals the low end, the high end, the midpoint, or none,
and lists the rows at 1,000,000 with their range. It writes nothing but its
report and always exits 0.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ENDPOINT = "https://fdw.fews.net/api/ipcpopulationsize.csv"
SCENARIOS = ("Current Situation", "Most Likely")


def _num(raw: object) -> float | None:
    try:
        text = str(raw).replace(",", "").strip()
        return float(text) if text else None
    except ValueError:
        return None


def classify(value: float | None, low: float | None, high: float | None) -> str:
    """Where ``value`` sits against its own published range."""
    if value is None:
        return "no_value"
    if low is None and high is None:
        return "no_range"
    if low is not None and high is not None and low == high == value:
        return "point_range"
    if low is not None and value == low:
        return "equals_low"
    if high is not None and value == high:
        return "equals_high"
    if low is not None and high is not None and abs(value - (low + high) / 2.0) < 0.5:
        return "midpoint"
    if low is not None and high is not None and low < value < high:
        return "inside_range"
    return "outside_range"


def summarise(rows: list[dict]) -> dict:
    columns = list(rows[0].keys()) if rows else []
    by_class: Counter = Counter()
    by_class_scenario: dict[str, Counter] = {s: Counter() for s in SCENARIOS}
    ranges_seen: Counter = Counter()
    at_million: list[dict] = []
    for r in rows:
        scen = str(r.get("scenario_name") or "")
        if scen not in SCENARIOS:
            continue
        v, lo, hi = (_num(r.get(k)) for k in ("value", "low_value", "high_value"))
        cls = classify(v, lo, hi)
        by_class[cls] += 1
        by_class_scenario[scen][cls] += 1
        ranges_seen[str(r.get("population_range") or "")] += 1
        if v == 1_000_000 and len(at_million) < 60:
            at_million.append({
                k: r.get(k) for k in (
                    "country_code", "scenario_name", "projection_start",
                    "projection_end", "reporting_date", "value",
                    "low_value", "high_value", "population_range",
                    "classification_scale", "phase",
                )
            })
    return {
        "n_rows": len(rows),
        "columns": columns,
        "value_vs_range": dict(by_class.most_common()),
        "value_vs_range_by_scenario": {s: dict(c.most_common()) for s, c in by_class_scenario.items()},
        "population_range_top": dict(ranges_seen.most_common(40)),
        "rows_at_1000000": at_million,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="diagnostics/fewsnet_values")
    ap.add_argument("--start-date", default=f"{date.today().year - 3}-01-01")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"endpoint": ENDPOINT, "start_date": args.start_date}
    try:
        import requests

        resp = requests.get(
            ENDPOINT,
            params={"start_date": args.start_date, "format": "csv"},
            headers={"User-Agent": "Mozilla/5.0 (Pythia probe)", "Accept": "text/csv"},
            timeout=180,
        )
        report["status"] = resp.status_code
        text = resp.content.decode("utf-8-sig", errors="replace")
        (out / "head.csv").write_text("\n".join(text.splitlines()[:40]), encoding="utf-8")
        rows = list(csv.DictReader(io.StringIO(text)))
        report.update(summarise(rows))
    except Exception as exc:  # a finding, not a build failure
        report["error"] = f"{type(exc).__name__}: {exc}"
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows_at_1000000"}, indent=2, default=str))
    for row in report.get("rows_at_1000000", [])[:20]:
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
