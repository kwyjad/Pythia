# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""How often each country crosses a drought-gate threshold on NMME (read-only).

For the last N issue months it reads CPC's adjusted probability of
below-normal precipitation (the relative measure) and, where the archive
still holds it, the ENSMEAN precipitation anomaly in mm/day (the absolute
measure the gate used before), takes the lead-1 country means through the
ingester's own aggregation, and reports for each candidate threshold the
share of country-months that cross it, how many countries ever cross, and
each country's crossing count. A threshold is chosen from this table; one
that no hyper-arid country can ever meet, or one a third of the world meets
every month, shows up here before it ships. Writes
``diagnostics/nmme_prob_crossing.json`` and always exits 0.
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import date
from ftplib import FTP
from pathlib import Path

OUT = Path("diagnostics/nmme_prob_crossing.json")
PROB_THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60)
ABS_THRESHOLDS = (-0.5, -1.0)
NAMED = ("EGY", "LBY", "SAU", "NER", "MLI", "SOM", "ETH", "KEN", "SDN",
         "AFG", "PAK", "GTM", "HND", "HTI", "ZWE", "MDG", "PHL", "BGD")


def _issue_months(latest: str, n: int) -> list[str]:
    y, m = int(latest[:4]), int(latest[4:6])
    idx = y * 12 + m - 1
    return [f"{(idx - k) // 12:04d}{(idx - k) % 12 + 1:02d}" for k in range(n - 1, -1, -1)]


def crossing_table(values: dict[str, dict[str, float]], thresholds, *, below: bool) -> list[dict]:
    """``values`` is {iso3: {issue_ym: value}}. Pure, so it is tested."""
    out = []
    cells = [(iso, v) for iso, by in values.items() for v in by.values()]
    for t in thresholds:
        hit = (lambda v: v <= t) if below else (lambda v: v >= t)
        per = {iso: sum(1 for v in by.values() if hit(v)) for iso, by in values.items()}
        out.append({
            "threshold": t,
            "country_months": len(cells),
            "share_crossing": round(sum(1 for _, v in cells if hit(v)) / len(cells), 4) if cells else None,
            "countries_ever_crossing": sum(1 for c in per.values() if c > 0),
            "countries": len(per),
            "per_country": dict(sorted(per.items())),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    n = int((argv or sys.argv[1:] or ["12"])[0])
    from resolver.ingestion import nmme

    report: dict = {"months_requested": n, "problems": []}
    prob: dict[str, dict[str, float]] = {}
    absolute: dict[str, dict[str, float]] = {}
    try:
        with FTP(nmme.FTP_HOST) as ftp:
            ftp.login()
            ftp.cwd(nmme.PROB_FTP_DIR)
            names = [x for x in ftp.nlst() if x.startswith("prate.") and x.endswith(".prob.adj.mon.nc")]
        latest = max(x.split(".")[1] for x in names)
    except Exception as exc:  # noqa: BLE001
        report["problems"].append(f"listing failed: {exc}")
        latest = date.today().strftime("%Y%m")
    months = _issue_months(latest, n)
    report["issue_months"] = months
    tmp = Path(tempfile.mkdtemp(prefix="nmme_cross_"))
    for ym in months:
        path = nmme._download_prob_file(ym, tmp)
        if path is None:
            report["problems"].append(f"{ym}: no probability file")
        else:
            for lead, df in nmme._aggregate_prob_nc(path, max_leads=1):
                for iso, v in zip(df["iso3"], df["anomaly_value"]):
                    prob.setdefault(iso, {})[ym] = float(v)
        try:
            files = nmme._download_nc_files(f"{ym}0800", tmp, variables={"prate": "prate_anom"}, max_leads=1)
            for entry in files:
                for lead, df in nmme._aggregate_multi_lead_nc(entry["path"], "prate", max_leads=1):
                    for iso, v in zip(df["iso3"], df["anomaly_value"]):
                        absolute.setdefault(iso, {})[ym] = float(v)
        except Exception as exc:  # noqa: BLE001 - realtime_anom keeps few vintages
            report["problems"].append(f"{ym}: anomaly not read ({exc})")
    report["prob_below"] = crossing_table(prob, PROB_THRESHOLDS, below=False)
    report["absolute_mm_day"] = crossing_table(absolute, ABS_THRESHOLDS, below=True)
    report["named_prob_below"] = {iso: prob.get(iso, {}) for iso in NAMED}
    report["named_absolute"] = {iso: absolute.get(iso, {}) for iso in NAMED}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, sort_keys=True))
    for key in ("prob_below", "absolute_mm_day"):
        print(f"== {key}")
        for row in report[key]:
            print(f"  t={row['threshold']}: {row['share_crossing']} of {row['country_months']} "
                  f"country-months; {row['countries_ever_crossing']}/{row['countries']} countries ever")
    for iso in NAMED:
        pb = report["named_prob_below"][iso]
        ab = report["named_absolute"][iso]
        print(f"  {iso}: prob_below {[round(v, 2) for _, v in sorted(pb.items())]} "
              f"| mm/day {[round(v, 2) for _, v in sorted(ab.items())]}")
    for p in report["problems"]:
        print("problem:", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
