#!/usr/bin/env bash
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

set -euo pipefail
BASE="${1:-}" 
if [[ -z "$BASE" ]]; then
  BASE="origin/main"
fi
python tools/context_pack.py --base "$BASE"
echo "Context pack created under ./context/"
