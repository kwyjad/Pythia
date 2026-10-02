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
SRC_RC_PROMPTS="$REPO_ROOT/horizon_scanner/rc_prompts.py"
SRC_RC_GROUNDING="$REPO_ROOT/horizon_scanner/rc_grounding_prompts.py"
SRC_TRIAGE_GROUNDING="$REPO_ROOT/horizon_scanner/hs_triage_grounding_prompts.py"

# Check source files exist
for f in "$SRC_FORECASTER" "$SRC_HS" "$SRC_GEMINI" "$SRC_RC_PROMPTS" "$SRC_RC_GROUNDING" "$SRC_TRIAGE_GROUNDING"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: Source file not found: $f"
    exit 1
  fi
done

# A dated snapshot is an archive. When one already exists for DATE and any
# prompt file has changed since, write DATE-2 (then DATE-3, ...) rather than
# overwrite it: a second prompt edit on the same day would otherwise replace
# the "before" copy with the "after" one, and the About page would lose a
# version.
same_as_snapshot() {
  local d="$1"
  cmp -s "$SRC_FORECASTER" "$d/forecaster_prompts.py" \
    && cmp -s "$SRC_HS" "$d/hs_prompts.py" \
    && cmp -s "$SRC_GEMINI" "$d/gemini_grounding.py" \
    && cmp -s "$SRC_RC_PROMPTS" "$d/rc_prompts.py" \
    && cmp -s "$SRC_RC_GROUNDING" "$d/rc_grounding_prompts.py" \
    && cmp -s "$SRC_TRIAGE_GROUNDING" "$d/hs_triage_grounding_prompts.py"
}
if [ -f "$DIR/forecaster_prompts.py" ] && ! same_as_snapshot "$DIR"; then
  BASE_DATE="$DATE"
  n=2
  while [ -f "$REPO_ROOT/docs/prompts/$BASE_DATE-$n/forecaster_prompts.py" ] \
        && ! same_as_snapshot "$REPO_ROOT/docs/prompts/$BASE_DATE-$n"; do
    n=$((n + 1))
  done
  DATE="$BASE_DATE-$n"
  DIR="$REPO_ROOT/docs/prompts/$DATE"
  echo "Snapshot for $BASE_DATE already holds different prompts; writing $DATE"
fi

# Create snapshot directory
mkdir -p "$DIR"

# Copy source files
cp "$SRC_FORECASTER"        "$DIR/forecaster_prompts.py"
cp "$SRC_HS"                "$DIR/hs_prompts.py"
cp "$SRC_GEMINI"            "$DIR/gemini_grounding.py"
cp "$SRC_RC_PROMPTS"        "$DIR/rc_prompts.py"
cp "$SRC_RC_GROUNDING"      "$DIR/rc_grounding_prompts.py"
cp "$SRC_TRIAGE_GROUNDING"  "$DIR/hs_triage_grounding_prompts.py"

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
