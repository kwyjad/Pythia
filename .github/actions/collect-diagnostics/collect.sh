#!/usr/bin/env bash
set -euo pipefail

ART_DIR="${1:-${ART_DIR:-}}"
if [[ -z "${ART_DIR}" ]]; then
  echo "Usage: collect.sh <ART_DIR>" >&2
  exit 1
fi

mkdir -p "${ART_DIR}"
DEST_DIR="${ART_DIR}/diagnostics"
mkdir -p "${DEST_DIR}"

copy_tree() {
  local src="$1"
  local dest="$2"
  if [[ -d "${src}" ]]; then
    mkdir -p "${dest}"
    cp -R "${src}/." "${dest}/" 2>/dev/null || true
  fi
}

copy_tree ".ci/diagnostics" "${DEST_DIR}/ci"
copy_tree "diagnostics" "${DEST_DIR}/ingestion"
copy_tree "${ART_DIR}/ci-diagnostics" "${DEST_DIR}/ci"
copy_tree "${ART_DIR}/diagnostics" "${DEST_DIR}/ingestion"

SUMMARY_PATH="${ART_DIR}/summary.md"
LEGACY_SUMMARY="${ART_DIR}/SUMMARY.md"
if [[ -f "${LEGACY_SUMMARY}" && ! -f "${SUMMARY_PATH}" ]]; then
  cp -f "${LEGACY_SUMMARY}" "${SUMMARY_PATH}"
fi
if [[ ! -f "${SUMMARY_PATH}" ]]; then
  {
    echo "# CI Diagnostics Summary"
    echo
    echo "(no summary generated)"
  } > "${SUMMARY_PATH}"
  cp -f "${SUMMARY_PATH}" "${LEGACY_SUMMARY}"
fi

echo "Collected diagnostics into ${ART_DIR}" 
ls -la "${ART_DIR}" || true
