#!/usr/bin/env bash
set -euo pipefail

JOB="${1:-job}"
RUN_ID="${GITHUB_RUN_ID:-local}"
RUN_ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"
DIST_DIR="dist"
BASE_DIR="${DIST_DIR}/diag-${JOB}"
SUMMARY_MD="${BASE_DIR}/SUMMARY.md"
ZIP_NAME="diagnostics-${JOB}-${RUN_ID}-${RUN_ATTEMPT}.zip"
ZIP_PATH="${DIST_DIR}/${ZIP_NAME}"

mkdir -p "${BASE_DIR}" "${DIST_DIR}" \
  ".ci" \
  ".ci/diagnostics" \
  ".ci/exitcodes"

append_section() {
  printf '\n## %s\n\n' "$1" >> "${SUMMARY_MD}"
}

append_code_block() {
  echo '```' >> "${SUMMARY_MD}"
  if [ -n "${1:-}" ] && [ -f "$1" ]; then
    cat "$1" >> "${SUMMARY_MD}"
  elif [ -n "${2:-}" ]; then
    echo "$2" >> "${SUMMARY_MD}"
  fi
  echo '```' >> "${SUMMARY_MD}"
}

versions_file="${BASE_DIR}/versions.txt"
{
  echo "== VERSIONS =="
  (python -V 2>&1 || true)
  (pip --version 2>&1 || true)
  if command -v pip >/dev/null 2>&1; then
    (pip freeze 2>/dev/null | sort || true)
  else
    echo "pip not available"
  fi
  if command -v duckdb >/dev/null 2>&1; then
    (duckdb --version 2>&1 || true)
  else
    echo "duckdb binary not available"
  fi
  (uname -a 2>&1 || true)
} >"${versions_file}" || true

env_file="${BASE_DIR}/env.txt"
python - <<'PY' >"${env_file}" 2>&1 || true
import os

SAFE_PREFIXES = (
    "RESOLVER_",
    "PERIOD_LABEL",
    "GITHUB_",
    "RUNNER_",
    "CI",
    "PYTHON",
)
SENSITIVE_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "KEY", "ACCESS", "CREDENTIAL", "PWD", "PASS")

print("== IMPORTANT ENVS ==")
for name in sorted(os.environ):
    upper = name.upper()
    if not any(name.startswith(prefix) for prefix in SAFE_PREFIXES):
        continue
    value = os.environ.get(name, "")
    if any(marker in upper for marker in SENSITIVE_MARKERS) and value:
        value = "***REDACTED***"
    print(f"{name}={value}")
PY

git_file="${BASE_DIR}/git.txt"
{
  echo "== GIT STATE =="
  (git rev-parse HEAD 2>&1 || true)
  (git status --porcelain 2>&1 || true)
  (git --no-pager log -1 --oneline 2>&1 || true)
} >"${git_file}" || true

duckdb_file="${BASE_DIR}/duckdb.txt"
python - <<'PY' >"${duckdb_file}" 2>&1 || true
import os
from pathlib import Path

def main() -> None:
    try:
        import duckdb  # type: ignore
    except ModuleNotFoundError:
        print("duckdb module not available")
        return

    db_path = Path(os.environ.get("RESOLVER_DB_PATH", "data/resolver.duckdb")).expanduser()
    if not db_path.exists():
        print(f"database not found: {db_path}")
        return

    con = duckdb.connect(str(db_path))
    try:
        print(f"database: {db_path}")
        try:
            df = con.sql("PRAGMA database_list").fetchdf()
            print(df)
        except Exception as exc:  # pragma: no cover - best effort diagnostics
            print(f"failed PRAGMA database_list: {exc}")
        try:
            tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
        except Exception as exc:
            print(f"failed to enumerate tables: {exc}")
            tables = []
        if not tables:
            print("no tables present")
        for table in tables:
            try:
                count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"{table}: {count} rows")
            except Exception as exc:  # pragma: no cover - best effort diagnostics
                print(f"{table}: failed to count ({exc})")
    finally:
        con.close()

if __name__ == "__main__":
    main()
PY

staging_listing_file="${BASE_DIR}/staging.txt"
{
  echo "== STAGING =="
  if [ -d data/staging ]; then
    (ls -R data/staging 2>&1 || true)
  else
    echo "data/staging missing"
  fi
} >"${staging_listing_file}" || true

staging_stats_file="${BASE_DIR}/staging_stats.md"
: >"${staging_stats_file}"
if [ -d data/staging ]; then
  found="false"
  while IFS= read -r csv_file; do
    [ -n "${csv_file}" ] || continue
    found="true"
    echo "### ${csv_file}" >> "${staging_stats_file}"
    echo '```' >> "${staging_stats_file}"
    echo "path: ${csv_file}" >> "${staging_stats_file}"
    if rows=$(wc -l <"${csv_file}" 2>/dev/null); then
      echo "rows: ${rows}" >> "${staging_stats_file}"
    else
      echo "rows: n/a" >> "${staging_stats_file}"
    fi
    header="$(head -n 1 "${csv_file}" 2>/dev/null | tr -d '\r')"
    if [ -n "${header}" ]; then
      echo "header: ${header}" >> "${staging_stats_file}"
    else
      echo "header: <empty>" >> "${staging_stats_file}"
    fi
    echo "sample:" >> "${staging_stats_file}"
    tail -n +2 "${csv_file}" 2>/dev/null | head -n 3 | tr -d '\r' >> "${staging_stats_file}"
    echo '```' >> "${staging_stats_file}"
    echo >> "${staging_stats_file}"
  done < <(find data/staging -type f -name '*.csv' 2>/dev/null | sort)
  if [ "${found}" = "false" ]; then
    echo "No CSV files located under data/staging." >> "${staging_stats_file}"
  fi
else
  echo "data/staging missing" >> "${staging_stats_file}"
fi

snapshots_file="${BASE_DIR}/snapshots.txt"
{
  echo "== SNAPSHOTS =="
  if [ -d data/snapshots ]; then
    (ls -R data/snapshots 2>&1 || true)
  else
    echo "data/snapshots missing"
  fi
} >"${snapshots_file}" || true

logs_file="${BASE_DIR}/logs.txt"
{
  echo "== RESOLVER LOGS =="
  handled="false"
  for candidate in resolver/logs data/logs; do
    if [ -d "${candidate}" ]; then
      handled="true"
      while IFS= read -r log_path; do
        [ -f "${log_path}" ] || continue
        echo "--- ${log_path}" | sed 's#//\+#/#g'
        (tail -n 200 "${log_path}" 2>&1 || true)
      done < <(find "${candidate}" -type f -name '*.log' 2>/dev/null | sort | head -n 10)
    fi
  done
  if [ "${handled}" = "false" ]; then
    echo "resolver/logs missing"
    echo "data/logs missing"
  fi
} >"${logs_file}" || true

pytest_file="${BASE_DIR}/pytest.txt"
if [ -f pytest-junit.xml ]; then
  python - <<'PY' >"${pytest_file}" 2>&1 || true
import xml.etree.ElementTree as ET
from pathlib import Path

def summarize(path: Path) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    tests = int(root.attrib.get("tests", 0))
    failures = int(root.attrib.get("failures", 0))
    errors = int(root.attrib.get("errors", 0))
    skipped = int(root.attrib.get("skipped", 0))
    print(f"tests={tests} failures={failures} errors={errors} skipped={skipped}")
    for testcase in root.iter("testcase"):
        for tag in ("failure", "error"):
            elem = testcase.find(tag)
            if elem is not None:
                name = testcase.attrib.get("name")
                classname = testcase.attrib.get("classname")
                print(f"{tag.upper()}: {classname}::{name}")
                text = (elem.text or "").strip()
                if text:
                    print(text[:2000])
                    print("---")

summarize(Path("pytest-junit.xml"))
PY
else
  echo "pytest-junit.xml missing" >"${pytest_file}"
fi

exitcodes_file="${BASE_DIR}/exitcodes.txt"
if ls .ci/exitcodes/* >/dev/null 2>&1; then
  {
    for breadcrumb in .ci/exitcodes/*; do
      [ -f "${breadcrumb}" ] || continue
      printf '%s: %s\n' "$(basename "${breadcrumb}")" "$(cat "${breadcrumb}" 2>/dev/null || echo n/a)"
    done
  } >"${exitcodes_file}" || true
else
  echo "no exitcode breadcrumbs found" >"${exitcodes_file}"
fi

usage_file="${BASE_DIR}/disk_usage.txt"
{
  echo "== DISK USAGE =="
  (du -sh data resolver 2>/dev/null || true)
  (df -h . 2>&1 || true)
} >"${usage_file}" || true

: >"${SUMMARY_MD}"
{
  echo "# Diagnostics Summary — ${JOB}"
  echo
  echo "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Run ID: ${RUN_ID} (attempt ${RUN_ATTEMPT})"
} >> "${SUMMARY_MD}"

append_section "Git"
append_code_block "${git_file}" "git information unavailable"

append_section "Versions"
append_code_block "${versions_file}" "version details unavailable"

append_section "Environment"
append_code_block "${env_file}" "no environment snapshot captured"

append_section "DuckDB"
append_code_block "${duckdb_file}" "duckdb not inspected"

append_section "Staging directory"
append_code_block "${staging_listing_file}" "data/staging missing"

append_section "Staging quick stats"
if [ -s "${staging_stats_file}" ]; then
  cat "${staging_stats_file}" >> "${SUMMARY_MD}"
else
  echo "No staging files discovered." >> "${SUMMARY_MD}"
fi

append_section "Snapshots directory"
append_code_block "${snapshots_file}" "data/snapshots missing"

append_section "Resolver logs (tail)"
append_code_block "${logs_file}" "resolver/data logs missing"

append_section "Pytest summary"
append_code_block "${pytest_file}" "pytest did not emit junit XML"

append_section "Step exit codes"
append_code_block "${exitcodes_file}" "no exitcode breadcrumbs found"

gate_rows_breadcrumb=".ci/exitcodes/gate_rows"
if [ -f "${gate_rows_breadcrumb}" ]; then
  rows_line=$(grep -Eo 'rows=[0-9]+' "${gate_rows_breadcrumb}" | head -n 1 || true)
  if [ -n "${rows_line}" ]; then
    rows_value=${rows_line#rows=}
    printf '\nCanonical row total (gate_rows): %s\n' "${rows_value}" >> "${SUMMARY_MD}"
  fi
fi

append_section "Smoke assertion"
smoke_assert_source=".ci/diagnostics/smoke-assert.json"
smoke_assert_dest="${BASE_DIR}/smoke-assert.json"
if [ -f "${smoke_assert_source}" ]; then
  cp "${smoke_assert_source}" "${smoke_assert_dest}" 2>/dev/null || true
  append_code_block "${smoke_assert_dest}" "smoke assertion report unavailable"
else
  append_code_block "" "smoke-assert.json not generated"
fi

append_section "Disk usage snapshot"
append_code_block "${usage_file}" "disk usage unavailable"

cp "${SUMMARY_MD}" ".ci/diagnostics/${JOB}.SUMMARY.md" 2>/dev/null || true

pushd "${DIST_DIR}" >/dev/null
rm -f "${ZIP_NAME}"
zip -qr "${ZIP_NAME}" "${BASE_DIR##${DIST_DIR}/}" || true
popd >/dev/null

if [ -n "${GITHUB_ENV:-}" ]; then
  {
    echo "ARTIFACT_PATH=${ZIP_PATH}"
    echo "SUMMARY_PATH=${SUMMARY_MD}"
  } >> "${GITHUB_ENV}"
fi

exit 0
