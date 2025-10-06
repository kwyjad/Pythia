#!/usr/bin/env bash
set -euo pipefail

echo ">> postCreate: Detecting Python interpreter used by VS Code..."
PY_BIN="${PY_BIN:-/usr/local/bin/python}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN="$(command -v python)"
fi
echo ">> Using PY_BIN=$PY_BIN"

echo ">> Upgrading pip and installing DB deps (offline-first)..."
"$PY_BIN" -m pip install --upgrade pip

OFFLINE_FAIL=0
if [ -f "tools/offline_wheels/constraints-db.txt" ]; then
  echo ">> Attempting offline install from tools/offline_wheels ..."
  if ! "$PY_BIN" -m pip install --no-index --find-links tools/offline_wheels -r tools/offline_wheels/constraints-db.txt; then
    OFFLINE_FAIL=1
  fi
else
  OFFLINE_FAIL=1
fi

if [ "$OFFLINE_FAIL" = "1" ]; then
  echo ">> Offline install not available or failed; falling back to online extras..."
  if ! "$PY_BIN" -m pip install -e ".[db]"; then
    "$PY_BIN" -m pip install duckdb pytest
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
