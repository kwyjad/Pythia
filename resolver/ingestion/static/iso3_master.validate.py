#!/usr/bin/env python3
"""Simple validator for the iso3_master.csv roster."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"admin0Pcode", "admin0Name"}


def _load_roster(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        engine="python",
        quoting=csv.QUOTE_MINIMAL,
    )


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    roster_path = Path(argv[0]) if argv else Path(__file__).with_name("iso3_master.csv")
    try:
        frame = _load_roster(roster_path)
    except Exception as exc:  # pragma: no cover - CLI helper
        print(f"Failed to load {roster_path}: {exc}", file=sys.stderr)
        return 1

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        print(f"Missing required columns: {sorted(missing)}", file=sys.stderr)
        return 1

    frame = frame.fillna("")
    frame["admin0Pcode"] = frame["admin0Pcode"].astype(str).str.strip()
    frame["admin0Name"] = frame["admin0Name"].astype(str).str.strip()

    if (frame["admin0Pcode"] == "").any() or (frame["admin0Name"] == "").any():
        print("Found empty admin0Pcode/admin0Name entries", file=sys.stderr)
        return 1

    if len(frame) < 180:
        print(f"Unexpected row count: {len(frame)} (<180)", file=sys.stderr)
        return 1

    print(f"iso3_master.csv OK: {len(frame)} rows")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
