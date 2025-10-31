#!/usr/bin/env bash
set -euo pipefail

JOB_NAME="${1:-job}"
MODE="${2:-generic}"
SMOKE_DIR="${3:-}"
SMOKE_MIN="${4:-}"
# shellcheck disable=SC2034
PLACEHOLDER_ARGS=("${MODE}" "${SMOKE_DIR}" "${SMOKE_MIN}")

if [[ -n "${ART_DIR:-}" ]]; then
  TARGET_DIR="${ART_DIR}"
else
  RUN_ID="${GITHUB_RUN_ID:-local}"
  RUN_ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"
  SAFE_JOB="${JOB_NAME:-job}"
  TARGET_DIR="dist/diagnostics-${SAFE_JOB}-${RUN_ID}-${RUN_ATTEMPT}"
fi

mkdir -p "${TARGET_DIR}"

set +e
SUMMARY_PATH=$(python scripts/ci/collect_diagnostics.py "${TARGET_DIR}")
STATUS=$?
set -e

if [[ "${STATUS}" -ne 0 || -z "${SUMMARY_PATH}" || ! -f "${SUMMARY_PATH}" ]]; then
  SUMMARY_PATH="${TARGET_DIR}/summary.md"
  {
    echo "# CI Diagnostics Summary"
    echo
    echo "_collector failed; see artifacts under ${TARGET_DIR}._"
  } >"${SUMMARY_PATH}"
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
      echo
      cat "${SUMMARY_PATH}"
    } >>"${GITHUB_STEP_SUMMARY}"
  fi
fi

if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    echo "ARTIFACT_PATH=${TARGET_DIR}"
    echo "SUMMARY_PATH=${SUMMARY_PATH}"
  } >>"${GITHUB_ENV}"
fi

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  SUMMARY_ABS=$(cd "$(dirname "${SUMMARY_PATH}")" && pwd)/"$(basename "${SUMMARY_PATH}")"
  TARGET_ABS=$(cd "${TARGET_DIR}" && pwd)
  {
    echo "artifact_path=${TARGET_ABS}"
    echo "summary_path=${SUMMARY_ABS}"
  } >>"${GITHUB_OUTPUT}"
fi

echo "Diagnostics directory: ${TARGET_DIR}"
echo "Summary path: ${SUMMARY_PATH}"
