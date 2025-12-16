#!/usr/bin/env bash
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

set -euo pipefail

# Guardrail: fail fast if a dev script tries to fetch legacy resolver repos
legacy_a="spa"
legacy_b="gbot"
legacy_c="meta"
legacy_d="c-bot"
legacy_pattern="${legacy_a}${legacy_b}|${legacy_c}${legacy_d}"
if grep -RniE "$legacy_pattern" . >/dev/null 2>&1; then
  echo "❌ postCreate detected legacy repo references. Please remove them."
  exit 1
fi

echo ">> postCreate: Detecting Python interpreter used by VS Code..."
PY_BIN="${PY_BIN:-/usr/local/bin/python}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN="$(command -v python)"
fi
echo ">> Using PY_BIN=$PY_BIN"

echo "[postCreate] Robust install start"
"$PY_BIN" -m pip install -U pip wheel setuptools "poetry-core>=1.9"
if [ -f requirements-dev.txt ]; then "$PY_BIN" -m pip install -r requirements-dev.txt || true; fi
if [ -f requirements.txt ]; then "$PY_BIN" -m pip install -r requirements.txt || true; fi
"$PY_BIN" -m pip install --only-binary=:all: "duckdb==1.1.3" || true
if "$PY_BIN" -m pip install -e ".[db]" --no-build-isolation; then
  echo "[postCreate] Editable install OK"
else
  echo "[postCreate] Editable install failed; falling back to PYTHONPATH mode"
  workspace_name="$(basename "$PWD")"
  if ! grep -q "PYTHONPATH" ~/.bashrc 2>/dev/null; then
    echo "export PYTHONPATH=\"\$PYTHONPATH:/workspaces/${workspace_name}\"" >> ~/.bashrc
  fi
fi

echo ">> Verifying duckdb importability with the same interpreter..."
"$PY_BIN" - <<'PYCODE'
import sys
print("Interpreter:", sys.executable)
import duckdb
print("duckdb installed:", duckdb.__version__)
PYCODE

echo ">> postCreate complete."
