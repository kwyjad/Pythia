#!/usr/bin/env bash
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

set -euo pipefail

if [[ "${RESOLVER_PRECEDENCE_ENABLE:-}" != "1" ]]; then
  echo "RESOLVER_PRECEDENCE_ENABLE is not set to 1; skipping precedence canary." >&2
  exit 0
fi

PRECEDENCE_CFG=${PRECEDENCE_CFG:-tools/precedence_config.yml}
CANDIDATES_DIR=${CANDIDATES_DIR:-artifacts/precedence/candidates}
DIAGNOSTICS_DIR=${DIAGNOSTICS_DIR:-diagnostics/precedence}
SELECTED_PATH="${DIAGNOSTICS_DIR}/selected.csv"

mkdir -p "${CANDIDATES_DIR}"
export CANDIDATES_DIR
export DIAGNOSTICS_DIR

python scripts/precedence/union_candidates.py

python -m resolver.cli.precedence_cli \
  --config "${PRECEDENCE_CFG}" \
  --candidates "${DIAGNOSTICS_DIR}/union_candidates.csv" \
  --out "${SELECTED_PATH}"

if [[ -s "${SELECTED_PATH}" ]]; then
  SELECTED_PATH="${SELECTED_PATH}" python - <<'PY'
import os
import pandas as pd

path = os.environ["SELECTED_PATH"]
df = pd.read_csv(path)
metrics = ", ".join(
    f"{metric}:{count}"
    for metric, count in df["metric"].value_counts().sort_index().items()
)
countries = ", ".join(sorted(df["iso3"].dropna().unique()))
print(
    f"Selected {len(df)} rows | metrics: {metrics or 'none'} | countries: {countries or 'none'}",
    flush=True,
)
PY
else
  echo "WARNING: precedence selection produced no rows." >&2
fi
