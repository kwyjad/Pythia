#!/usr/bin/env bash
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

set -euo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"
echo "🔎 Searching for series semantics hotspots in resolver/ ..."
rg -n --hidden --glob '!**/.venv/**' '(series_semantics|canonicali[sz]e|stock_estimate|compute_series_semantics)' resolver | sed 's/^/• /'
echo
echo "Tip: open docs at resolver/docs/series_semantics_map.md"
