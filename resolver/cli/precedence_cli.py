"""
Opt-in CLI to run precedence selection locally or in canary jobs.
Does NOT run in existing workflows until explicitly wired.

Usage:
  python -m resolver.cli.precedence_cli \
      --config tools/precedence_config.yml \
      --candidates diagnostics/ingestion/export_preview/facts.csv \
      --out diagnostics/precedence/selected.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from tools.precedence_engine import apply_precedence


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run config-driven precedence selection")
    parser.add_argument("--config", required=True, help="Path to precedence YAML config")
    parser.add_argument("--candidates", required=True, help="CSV of candidate rows")
    parser.add_argument("--out", required=True, help="Destination CSV for selected rows")
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    candidates = pd.read_csv(args.candidates, parse_dates=["as_of_date"])

    selected = apply_precedence(candidates, config)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False)
    print(f"Wrote {len(selected):,} rows → {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
