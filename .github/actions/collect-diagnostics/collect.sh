#!/usr/bin/env bash
set -euo pipefail

job_name=""
mode="generic"
smoke_canonical_dir=""
smoke_min_rows=""

if [[ $# -ge 4 ]]; then
  job_name="$1"
  mode="$2"
  smoke_canonical_dir="$3"
  smoke_min_rows="$4"
  ART_DIR="${ART_DIR:-${GITHUB_WORKSPACE:-.}/.ci/diagnostics}"
else
  ART_DIR="${1:-${ART_DIR:-.ci/diagnostics}}"
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

SUMMARY_PATH="${ART_DIR}/SUMMARY.md"
LEGACY_SUMMARY="${ART_DIR}/summary.md"
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

smoke_total_rows=""
if [[ "${mode}" == "smoke" && -n "${smoke_canonical_dir}" ]]; then
  if [[ -d "${smoke_canonical_dir}" ]]; then
    smoke_total_rows=$(python - <<'PY'
import pathlib
import sys
canonical = pathlib.Path(sys.argv[1])
total = 0
for path in canonical.rglob('*.csv'):
    try:
        with path.open('r', encoding='utf-8', errors='ignore') as handle:
            total += max(sum(1 for _ in handle) - 1, 0)
    except Exception:
        pass
print(total)
PY
"${smoke_canonical_dir}") || smoke_total_rows=""
  else
    smoke_total_rows="0"
  fi
fi

echo "Diagnostics ready at ${ART_DIR}"
ls -la "${ART_DIR}" || true

if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    echo "ARTIFACT_PATH=${ART_DIR}"
    if [[ -n "${job_name}" ]]; then
      echo "DIAGNOSTICS_JOB=${job_name}"
    fi
    if [[ -n "${smoke_total_rows}" ]]; then
      echo "SMOKE_TOTAL_ROWS=${smoke_total_rows}"
    fi
  } >> "${GITHUB_ENV}"
fi

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "artifact_path=${ART_DIR}"
    echo "summary_path=${SUMMARY_PATH}"
    if [[ -n "${smoke_total_rows}" ]]; then
      echo "smoke_total_rows=${smoke_total_rows}"
    fi
  } >> "${GITHUB_OUTPUT}"
fi
