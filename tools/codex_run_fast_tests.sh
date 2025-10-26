#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

python -m pip install --upgrade pip
pip install -r "${REPO_ROOT}/requirements-test.txt"

cd "${REPO_ROOT}"

echo "[codex] Python: $(python --version)"
python - <<'PY'
import duckdb, pyarrow, pandas
print("[codex] Sanity:", "duckdb", duckdb.__version__, "| pyarrow", pyarrow.__version__, "| pandas", pandas.__version__)
PY

# Default to skip online connectors for the fast suite; individual tests can override.
export RESOLVER_SKIP_DTM="${RESOLVER_SKIP_DTM:-1}"

# Pass through any extra args Codex provides (e.g., -q, -k).
pytest -q "$@"
