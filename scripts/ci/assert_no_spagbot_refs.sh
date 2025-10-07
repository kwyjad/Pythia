#!/usr/bin/env bash
set -euo pipefail
# Scan the repo for any accidental references to Spagbot repos/paths.
if git grep -n -i -E 'spagbot|metac-bot' -- . ':!scripts/ci/assert_no_spagbot_refs.sh' >/dev/null; then
  echo "❌ Found references to 'spagbot' or 'metac-bot' in the repository. Please remove them."
  git grep -n -i -E 'spagbot|metac-bot' -- . ':!scripts/ci/assert_no_spagbot_refs.sh' || true
  exit 1
fi
echo "✅ No spagbot references found."
