#!/usr/bin/env bash
# Snapshot prompt source files for versioning on the About page.
# Usage: bash scripts/snapshot_prompts.sh [YYYY-MM-DD] [label]
# If no date is provided, today's date is used.
# If no label is provided, "Snapshot" is used.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATE="${1:-$(date +%Y-%m-%d)}"
LABEL="${2:-Snapshot}"
DIR="$REPO_ROOT/docs/prompts/$DATE"
MANIFEST="$REPO_ROOT/docs/prompts/versions.json"

# Source files to archive
SRC_FORECASTER="$REPO_ROOT/forecaster/prompts.py"
SRC_HS="$REPO_ROOT/horizon_scanner/prompts.py"
SRC_GEMINI="$REPO_ROOT/pythia/web_research/backends/gemini_grounding.py"

# Check source files exist
for f in "$SRC_FORECASTER" "$SRC_HS" "$SRC_GEMINI"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: Source file not found: $f"
    exit 1
  fi
done

# Create snapshot directory
mkdir -p "$DIR"

# Copy source files
cp "$SRC_FORECASTER" "$DIR/forecaster_prompts.py"
cp "$SRC_HS"         "$DIR/hs_prompts.py"
cp "$SRC_GEMINI"     "$DIR/gemini_grounding.py"

# Update versions.json manifest
if [ ! -f "$MANIFEST" ]; then
  echo '[]' > "$MANIFEST"
fi

# Check if this date already exists in the manifest
if python3 -c "
import json, sys
with open('$MANIFEST') as f:
    versions = json.load(f)
for v in versions:
    if v['date'] == '$DATE':
        v['label'] = '$LABEL'
        with open('$MANIFEST', 'w') as f:
            json.dump(versions, f, indent=2)
        sys.exit(0)
versions.append({'date': '$DATE', 'label': '$LABEL'})
versions.sort(key=lambda v: v['date'], reverse=True)
with open('$MANIFEST', 'w') as f:
    json.dump(versions, f, indent=2)
"; then
  echo "Snapshot saved: $DIR"
  echo "Manifest updated: $MANIFEST"
else
  echo "ERROR: Failed to update manifest"
  exit 1
fi
