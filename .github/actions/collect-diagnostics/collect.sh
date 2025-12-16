#!/usr/bin/env bash
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

set -euo pipefail

: "${ART_DIR:?ART_DIR required}"

mkdir -p "${ART_DIR}"

SUMMARY_PATH="${ART_DIR}/summary.md"
UPPER_SUMMARY="${ART_DIR}/SUMMARY.md"

if [[ -f "${UPPER_SUMMARY}" && ! -f "${SUMMARY_PATH}" ]]; then
  cp -f "${UPPER_SUMMARY}" "${SUMMARY_PATH}"
fi

if [[ -f "${SUMMARY_PATH}" && ! -f "${UPPER_SUMMARY}" ]]; then
  cp -f "${SUMMARY_PATH}" "${UPPER_SUMMARY}"
fi

if [[ ! -f "${SUMMARY_PATH}" && ! -f "${UPPER_SUMMARY}" ]]; then
  {
    echo "# CI Diagnostics Summary"
    echo
    echo "(no summary generated)"
  } > "${SUMMARY_PATH}"
  cp -f "${SUMMARY_PATH}" "${UPPER_SUMMARY}"
fi

echo "Diagnostics collected under: ${ART_DIR}"
ls -lah "${ART_DIR}" || true

# Emit step outputs for composite action
SUMMARY_PATH="${SUMMARY_PATH:-${ART_DIR}/SUMMARY.md}"
{
  echo "artifact_path=${ART_DIR}"
  echo "summary_path=${SUMMARY_PATH}"
} >> "${GITHUB_OUTPUT}"
