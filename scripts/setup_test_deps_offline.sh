#!/usr/bin/env bash
set -euo pipefail

# Offline helper to install vendored test dependencies (currently duckdb) without network access.
# Exits with a clear message if the expected wheel is missing.

VERSION="1.1.0"
WHEEL_PATTERN="duckdb-${VERSION}-cp311-cp311-manylinux_2_17_x86_64.whl"
WHEEL_PATH="vendor/wheels/${WHEEL_PATTERN}"

if python - <<'PY'
try:
    import duckdb  # type: ignore
    print("duckdb already installed")
    raise SystemExit(0)
except ImportError:
    raise SystemExit(1)
PY
then
    exit 0
fi

if [[ ! -f "${WHEEL_PATH}" ]]; then
    echo "duckdb wheel not found at ${WHEEL_PATH}" >&2
    echo "Download with: python -m pip download --only-binary=:all: --python-version 311 --platform manylinux_2_17_x86_64 --implementation cp --abi cp311 duckdb==${VERSION} -d vendor/wheels" >&2
    exit 1
fi

python -m pip install --no-index --find-links vendor/wheels "duckdb==${VERSION}"
