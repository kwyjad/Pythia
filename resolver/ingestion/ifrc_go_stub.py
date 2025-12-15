#!/usr/bin/env python3
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import argparse
import os
from pathlib import Path

from ._stub_utils import load_registries, now_dates, write_staging

OUT = Path(__file__).resolve().parents[1] / "staging" / "ifrc_go.csv"

def make_rows():
    countries, shocks = load_registries()
    as_of, pub, ing = now_dates()

    ctry = countries.tail(2)
    hz = shocks[shocks["hazard_code"].isin(["FL","TC"])]

    rows = []
    for _, c in ctry.iterrows():
        for _, h in hz.iterrows():
            event_id = f"{c.iso3}-{h.hazard_code}-ifrc-stub-r1"
            rows.append([
                event_id, c.country_name, c.iso3,
                h.hazard_code, h.hazard_label, h.hazard_class,
                "affected", "75000", "persons",
                as_of, pub,
                "IFRC", "sitrep", "https://example.org/ifrc", f"{h.hazard_label} DREF",
                f"People affected per IFRC GO DREF for {h.hazard_label}.",
                "api", "med", 1, ing
            ])
    return rows

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Emit IFRC GO stub data.")
    parser.add_argument(
        "--out",
        help="Directory or CSV file path for stub output; defaults to the resolver staging location.",
        default=None,
    )
    args = parser.parse_args(argv)

    if args.out:
        target = Path(args.out).expanduser()
        os.environ.pop("RESOLVER_OUTPUT_PATH", None)
        os.environ.pop("RESOLVER_OUTPUT_DIR", None)
        if target.suffix.lower() == ".csv":
            os.environ["RESOLVER_OUTPUT_PATH"] = str(target)
        else:
            os.environ["RESOLVER_OUTPUT_DIR"] = str(target)

    written = write_staging(make_rows(), OUT, series_semantics="stock")
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
