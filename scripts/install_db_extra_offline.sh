#!/usr/bin/env bash
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

set -euo pipefail

WHEEL_DIR="${WHEEL_DIR:-tools/offline_wheels}"

python -m pip install --upgrade pip
python -m pip install --no-index --find-links "$WHEEL_DIR" -r "$WHEEL_DIR/constraints-db.txt"
