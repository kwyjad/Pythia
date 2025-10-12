#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"
echo "🔎 Searching for series semantics hotspots in resolver/ ..."
rg -n --hidden --glob '!**/.venv/**' '(series_semantics|canonicali[sz]e|stock_estimate|compute_series_semantics)' resolver | sed 's/^/• /'
echo
echo "Tip: open docs at resolver/docs/series_semantics_map.md"
