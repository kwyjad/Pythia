#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${ROOT_DIR}"

OUT_DIR=".ci/diagnostics"
mkdir -p "${OUT_DIR}"

shopt -s nullglob globstar

copy_into_out() {
  local src="$1"
  local dest="${OUT_DIR}/$2"
  if [[ "$src" == "${OUT_DIR}"* || "$src" == "./${OUT_DIR}"* ]]; then
    return
  fi
  local dest_dir
  dest_dir="$(dirname "${dest}")"
  mkdir -p "${dest_dir}"
  cp -p "${src}" "${dest}"
}

# Collect junit output if present
if [ -f "pytest-junit.xml" ]; then
  copy_into_out "pytest-junit.xml" "pytest-junit.xml"
fi

# Capture environment and dependency snapshots
{
  env | sort > "${OUT_DIR}/env.txt"
} || true
{
  python -m pip freeze > "${OUT_DIR}/pip-freeze.txt"
} || true

# Gather resolver logs and generic log files
if [ -d "resolver" ]; then
  find resolver -type f -path "*/.logs/*" -size -104857600c -print0 |
    while IFS= read -r -d '' file; do
      rel="${file#*/}"
      copy_into_out "${file}" "${rel}"
    done
fi

find . -type f -name '*.log' -size -104857600c -print0 |
  while IFS= read -r -d '' file; do
    if [[ "${file}" == "${OUT_DIR}"* || "${file}" == "./${OUT_DIR}"* ]]; then
      continue
    fi
    rel="${file#./}"
    copy_into_out "${file}" "${rel}"
  done

# Collect DuckDB files from workspace and pytest temp directories
collect_duckdb_files() {
  local search_root="$1"
  if [ ! -d "${search_root}" ]; then
    return
  fi
  find "${search_root}" -maxdepth 6 -type f -name '*.duckdb' -size -314572800c -print0 |
    while IFS= read -r -d '' file; do
      if [[ "${file}" == "${OUT_DIR}"* || "${file}" == "./${OUT_DIR}"* ]]; then
        continue
      fi
      rel="${file}"
      if [[ "${file}" == "${ROOT_DIR}"* ]]; then
        rel="${file#${ROOT_DIR}/}"
      elif [[ "${file}" == /* ]]; then
        rel="${file#/}"
      fi
      copy_into_out "${file}" "duckdb/${rel}"
    done
}

collect_duckdb_files "${ROOT_DIR}"
collect_duckdb_files "/tmp/pytest-of-$(whoami 2>/dev/null || echo runner)"
collect_duckdb_files "/tmp"

safe_job=${JOB_NAME// /_}
safe_label=${DUCKDB_LABEL// /_}
ARCHIVE_NAME="diagnostics_${safe_job:-job}_${safe_label:-default}.tar.gz"
# Ensure predictable archive contents even when empty
if [ -z "$(find "${OUT_DIR}" -mindepth 1 -maxdepth 1 | head -n1)" ]; then
  touch "${OUT_DIR}/.keep"
fi

tar -czf "${ARCHIVE_NAME}" -C "${OUT_DIR}" .

echo "Created diagnostics archive: ${ARCHIVE_NAME}" >&2
