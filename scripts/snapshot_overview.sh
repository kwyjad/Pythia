#!/usr/bin/env bash
# Snapshot fred_overview.md for versioning on the About page.
# Usage: bash scripts/snapshot_overview.sh [YYYY-MM-DD] [label]
# If no date is provided, today's date is used.
# If no label is provided, "Snapshot" is used.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATE="${1:-$(date +%Y-%m-%d)}"
LABEL="${2:-Snapshot}"
DIR="$REPO_ROOT/docs/overview/$DATE"
MANIFEST="$REPO_ROOT/docs/overview/versions.json"

# Source file to archive
SRC_OVERVIEW="$REPO_ROOT/docs/fred_overview.md"

if [ ! -f "$SRC_OVERVIEW" ]; then
  echo "ERROR: Source file not found: $SRC_OVERVIEW"
  exit 1
fi

# A dated snapshot is an archive. When one already exists for DATE and the
# overview has changed since, write DATE-2 (then DATE-3, ...) rather than
# overwrite it: a second edit on the same day would otherwise replace the
# "before" copy with the "after" one, and the About page would lose a version.
if [ -f "$DIR/fred_overview.md" ] && ! cmp -s "$SRC_OVERVIEW" "$DIR/fred_overview.md"; then
  BASE_DATE="$DATE"
  n=2
  while [ -f "$REPO_ROOT/docs/overview/$BASE_DATE-$n/fred_overview.md" ] \
        && ! cmp -s "$SRC_OVERVIEW" "$REPO_ROOT/docs/overview/$BASE_DATE-$n/fred_overview.md"; do
    n=$((n + 1))
  done
  DATE="$BASE_DATE-$n"
  DIR="$REPO_ROOT/docs/overview/$DATE"
  echo "Snapshot for $BASE_DATE already holds a different overview; writing $DATE"
fi

# Create snapshot directory
mkdir -p "$DIR"

# Copy source file
cp "$SRC_OVERVIEW" "$DIR/fred_overview.md"

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
