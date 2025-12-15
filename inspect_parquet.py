#!/usr/bin/env python3
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import pandas as pd
import os
import sys

def main():
    path = os.path.join("Dashboard", "data", "forecasts.parquet")
    if not os.path.exists(path):
        print(f"ERROR: Parquet file not found at: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(path)
    print("=== Column names ===")
    for c in df.columns.tolist():
        print(" -", c)
    print("\n=== First 5 rows ===")
    print(df.head(5))

if __name__ == "__main__":
    main()
