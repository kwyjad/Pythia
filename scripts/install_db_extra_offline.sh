#!/usr/bin/env bash
set -euo pipefail

WHEEL_DIR="${WHEEL_DIR:-tools/offline_wheels}"

python -m pip install --upgrade pip
python -m pip install --no-index --find-links "$WHEEL_DIR" -r "$WHEEL_DIR/constraints-db.txt"
