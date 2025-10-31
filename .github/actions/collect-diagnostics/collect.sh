#!/usr/bin/env bash
set -euo pipefail

ART_DIR_ENV="${ART_DIR:-}"
TARGET_DIR=""

if [[ $# -eq 1 ]]; then
  TARGET_DIR="${1}"
else
  JOB="${1:-job}"
  # legacy arguments retained for compatibility but unused
  RUN_ID="${GITHUB_RUN_ID:-local}"
  RUN_ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"
  DIST_DIR="dist"
  TARGET_DIR="${DIST_DIR}/diagnostics-${JOB}-${RUN_ID}-${RUN_ATTEMPT}"
  mkdir -p "${DIST_DIR}"
fi

if [[ -n "$ART_DIR_ENV" ]]; then
  TARGET_DIR="$ART_DIR_ENV"
fi

mkdir -p "$TARGET_DIR"

set +e
SUMMARY_PATH=$(python scripts/ci/collect_diagnostics.py "$TARGET_DIR")
STATUS=$?
set -e

if [[ "$STATUS" -ne 0 || -z "$SUMMARY_PATH" || ! -f "$SUMMARY_PATH" ]]; then
  SUMMARY_PATH="$TARGET_DIR/summary.md"
  {
    echo "# CI Diagnostics Summary — nightly (stub)"
    echo "Collector failed to run; see raw artifacts under $TARGET_DIR."
  } >"$SUMMARY_PATH"
fi

SUMMARY_UPPER="$TARGET_DIR/SUMMARY.md"
cp "$SUMMARY_PATH" "$SUMMARY_UPPER" 2>/dev/null || true

if [[ -n "${GITHUB_STEP_SUMMARY:-}" && -f "$SUMMARY_PATH" ]]; then
  echo "" >>"$GITHUB_STEP_SUMMARY"
  cat "$SUMMARY_PATH" >>"$GITHUB_STEP_SUMMARY"
fi

if [[ -n "${GITHUB_ENV:-}" ]]; then
  SUMMARY_ABS=$(cd "$(dirname "$SUMMARY_PATH")" && pwd)/"$(basename "$SUMMARY_PATH")"
  TARGET_ABS=$(cd "$TARGET_DIR" && pwd)
  {
    echo "ARTIFACT_PATH=${TARGET_DIR}"
    echo "ARTIFACT_PATH_ABS=${TARGET_ABS}"
    echo "SUMMARY_PATH=${SUMMARY_ABS}"
  } >>"$GITHUB_ENV"
fi

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  SUMMARY_ABS=$(cd "$(dirname "$SUMMARY_PATH")" && pwd)/"$(basename "$SUMMARY_PATH")"
  TARGET_ABS=$(cd "$TARGET_DIR" && pwd)
  {
    echo "artifact_path=${TARGET_ABS}"
    echo "summary_path=${SUMMARY_ABS}"
  } >>"$GITHUB_OUTPUT"
fi

echo "Summary at: $SUMMARY_PATH"
