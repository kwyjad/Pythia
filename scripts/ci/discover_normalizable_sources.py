#!/usr/bin/env python3
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""List resolver normalization sources available in staging."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from resolver.transform.normalize import ADAPTER_REGISTRY


def main(period: str) -> int:
    raw_dir = Path("data/staging") / period / "raw"
    if not raw_dir.exists():
        raise SystemExit(f"Raw directory missing: {raw_dir}")

    available = [p.stem for p in sorted(raw_dir.glob("*.csv")) if p.stem in ADAPTER_REGISTRY]
    if not available:
        raise SystemExit("No canonicalizable sources found in raw staging data")

    print("SOURCES=" + ",".join(available))
    return 0


def resolve_period_from_args() -> str:
    if len(sys.argv) > 1 and sys.argv[1]:
        return sys.argv[1]
    period = os.environ.get("PERIOD_LABEL")
    if not period:
        raise SystemExit("PERIOD_LABEL argument or environment variable is required")
    return period


if __name__ == "__main__":
    sys.exit(main(resolve_period_from_args()))
