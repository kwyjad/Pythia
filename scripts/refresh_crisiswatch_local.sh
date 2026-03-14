#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/Users/kevin/Documents/Github/Pythia"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"

cd "$REPO_DIR"

git checkout main
git pull --ff-only

$PYTHON -m scripts.refresh_crisiswatch \
    --channel chrome --auto-push --verbose \
    2>&1 | tee /tmp/crisiswatch_refresh.log
