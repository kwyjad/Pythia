#!/usr/bin/env bash
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

set -euo pipefail
# Scan the repo for any accidental references to Spagbot repos/paths.
if git grep -n -i -E 'spagbot|metac-bot' -- . ':!scripts/ci/assert_no_spagbot_refs.sh' >/dev/null; then
  echo "❌ Found references to 'spagbot' or 'metac-bot' in the repository. Please remove them."
  git grep -n -i -E 'spagbot|metac-bot' -- . ':!scripts/ci/assert_no_spagbot_refs.sh' || true
  exit 1
fi
echo "✅ No spagbot references found."
