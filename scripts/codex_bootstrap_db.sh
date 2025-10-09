#!/usr/bin/env bash
set -euo pipefail

echo "[codex] Installing project with DB extra..."
if command -v poetry >/dev/null 2>&1; then
  poetry lock --no-update || true
  poetry install --with dev -E db
else
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -e ".[db]"
fi

echo "[codex] Verifying duckdb is importable..."
python - <<'PY'
try:
    import duckdb  # noqa: F401
    print("duckdb import OK")
except Exception as e:
    raise SystemExit(f"duckdb import FAILED: {e}")
PY

echo "[codex] Done."
