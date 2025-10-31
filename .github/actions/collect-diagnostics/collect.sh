#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="${1:-}"
if [[ -z "${DEST_DIR}" ]]; then
  if [[ -n "${ART_DIR:-}" ]]; then
    DEST_DIR="${ART_DIR}"
  else
    DEST_DIR=".ci/diagnostics"
  fi
fi

mkdir -p "${DEST_DIR}"

SOURCE_DIR="${ART_DIR:-}"
if [[ -n "${SOURCE_DIR}" && -d "${SOURCE_DIR}" && "${SOURCE_DIR}" != "${DEST_DIR}" ]]; then
  rsync -a "${SOURCE_DIR}/" "${DEST_DIR}/" || true
fi

copy_if_exists() {
  local path="$1"
  if [[ -n "${path}" && -f "${path}" ]]; then
    cp -f "${path}" "${DEST_DIR}/" || true
  fi
}

copy_if_exists "${DEST_DIR}/env.txt"
copy_if_exists "${DEST_DIR}/pip-freeze.txt"
copy_if_exists "${SOURCE_DIR}/env.txt"
copy_if_exists "${SOURCE_DIR}/pip-freeze.txt"
copy_if_exists "env.txt"
copy_if_exists "pip-freeze.txt"
copy_if_exists "${SOURCE_DIR}/pytest-junit.xml"
copy_if_exists "${SOURCE_DIR}/db.junit.xml"
copy_if_exists "pytest-junit.xml"
copy_if_exists "db.junit.xml"
log_candidate=".ci/pytest-${RUNNER_OS:-Linux}.out.log"
copy_if_exists "${log_candidate}"

if [[ ! -f "${DEST_DIR}/SUMMARY.md" ]]; then
  if ! python scripts/ci/generate_summary.py --out "${DEST_DIR}/SUMMARY.md"; then
    {
      echo "# CI Diagnostics Summary (fallback)"
      echo "Generator failed; see raw artifacts."
    } > "${DEST_DIR}/SUMMARY.md"
  fi
fi

if [[ -n "${GITHUB_STEP_SUMMARY:-}" && -f "${DEST_DIR}/SUMMARY.md" ]]; then
  cat "${DEST_DIR}/SUMMARY.md" >> "${GITHUB_STEP_SUMMARY}"
fi

echo "Diagnostics ready in ${DEST_DIR} (flat folder, no nested zips)."
