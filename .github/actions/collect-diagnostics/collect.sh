#!/usr/bin/env bash
set -euo pipefail

JOB="${1:-job}"
MODE="${2:-generic}"
SMOKE_CANONICAL_DIR="${3:-data/staging/ci-smoke/canonical}"
SMOKE_MIN_ROWS="${4:-1}"
RUN_ID="${GITHUB_RUN_ID:-local}"
RUN_ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"
DIST_DIR="dist"
BASE_DIR="${DIST_DIR}/diagnostics-${JOB}-${RUN_ID}-${RUN_ATTEMPT}"
SUMMARY_MD="${BASE_DIR}/SUMMARY.md"

rm -rf "${BASE_DIR}"
mkdir -p "${BASE_DIR}" "${DIST_DIR}" \
  ".ci" \
  ".ci/diagnostics" \
  ".ci/exitcodes"

# Pick the toolchain python installed by actions/setup-python if present
if command -v python >/dev/null 2>&1; then
  PYTHON=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

SMOKE_ASSERT_SOURCE=".ci/diagnostics/smoke-assert.json"
SMOKE_TOTAL_ROWS_VALUE=""

read_smoke_total() {
  local value
  local status
  if [ ! -f "${SMOKE_ASSERT_SOURCE}" ]; then
    echo "n/a"
    return
  fi
  set +e
  value=$("$PYTHON" - "${SMOKE_ASSERT_SOURCE}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    sys.exit(1)

print(payload.get("total_rows", "n/a"))
PY
  )
  status=$?
  set -euo pipefail
  if [ "${status}" -ne 0 ] || [ -z "${value}" ]; then
    echo "n/a"
  else
    echo "${value}" | tr -d '\n'
  fi
}

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

if [ "${MODE}" = "smoke" ]; then
  if [ -z "${SMOKE_TOTAL_ROWS_VALUE}" ] && [ -f "${SMOKE_ASSERT_SOURCE}" ]; then
    SMOKE_TOTAL_ROWS_VALUE=$(read_smoke_total)
  fi

  if [ -z "${SMOKE_TOTAL_ROWS_VALUE}" ]; then
    SMOKE_TOTAL_ROWS_VALUE="n/a"
  fi
fi

versions_file="${BASE_DIR}/versions.txt"
{
  echo "== VERSIONS =="
  "$PYTHON" - <<'PY'
import sys, subprocess, shutil, platform
print(f"Python {sys.version.split()[0]}")
pip = shutil.which("pip") or ""
print(subprocess.getoutput(f"{pip} --version" if pip else "pip not found"))
print(subprocess.getoutput("pip freeze || true"))
print(subprocess.getoutput("uname -a || true"))
PY
  echo ""
  echo "== PYTHON PATHS =="
  which -a python || true
  which -a python3 || true
  echo ""
  echo "== DUCKDB =="
  if command -v duckdb >/dev/null 2>&1; then
    duckdb --version 2>&1 || true
  else
    echo "duckdb binary not available"
  fi
} >"${versions_file}" || true

env_file="${BASE_DIR}/env.txt"
"$PYTHON" - <<'PY' >"${env_file}" 2>&1 || true
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
"$PYTHON" - <<'PY' >"${duckdb_file}" 2>&1 || true
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
"$PYTHON" - <<'PY' >"${staging_stats_file}" 2>/dev/null || true
from __future__ import annotations

import csv
import json
from pathlib import Path


def rows_from_report() -> dict[Path, int]:
    report_path = Path(".ci/diagnostics/smoke-assert.json")
    if not report_path.is_file():
        return {}
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    mapping: dict[Path, int] = {}
    for entry in payload.get("files", []):
        try:
            entry_path = Path(entry["path"]).resolve()
        except Exception:
            continue
        try:
            rows = int(entry.get("rows", 0))
        except Exception:
            rows = 0
        mapping[entry_path] = rows
    return mapping


def count_rows(csv_path: Path) -> int:
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            rows = 0
            for record in reader:
                if not any(cell.strip() for cell in record):
                    continue
                rows += 1
            return rows
    except Exception:
        return 0


def header_and_sample(csv_path: Path) -> tuple[str, list[str]]:
    header = ""
    sample: list[str] = []
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header_row = next(reader, [])
            header = ",".join(header_row)
            for record in reader:
                if not any(cell.strip() for cell in record):
                    continue
                sample.append(",".join(record))
                if len(sample) >= 3:
                    break
    except Exception:
        header = ""
        sample = []
    return header, sample


def main() -> None:
    staging_root = Path("data/staging")
    if not staging_root.exists():
        print("data/staging missing")
        return

    files = sorted(path.resolve() for path in staging_root.rglob("*.csv"))
    if not files:
        print("No CSV files located under data/staging.")
        return

    row_report = rows_from_report()
    cwd = Path.cwd()
    for file_path in files:
        rel_path = file_path
        try:
            rel_path = file_path.relative_to(cwd)
        except ValueError:
            pass

        rows = row_report.get(file_path, count_rows(file_path))
        header, sample = header_and_sample(file_path)

        print(f"### {rel_path}")
        print("```")
        print(f"path: {rel_path}")
        print(f"rows: {rows}")
        print(f"header: {header if header else '<empty>'}")
        print("sample:")
        if sample:
            for line in sample:
                print(line)
        else:
            print("<no sample>")
        print("```")
        print()


if __name__ == "__main__":
    main()
PY

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
pytest_source=""
if [ -f ".ci/diagnostics/pytest-junit.xml" ]; then
  pytest_source=".ci/diagnostics/pytest-junit.xml"
elif [ -f "pytest-junit.xml" ]; then
  pytest_source="pytest-junit.xml"
fi

if [ -n "${pytest_source}" ]; then
  PYTEST_JUNIT="${pytest_source}" "$PYTHON" - <<'PY' >"${pytest_file}" 2>&1 || true
import os
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

summarize(Path(os.environ["PYTEST_JUNIT"]))
PY
else
  {
    echo "pytest-junit.xml missing"
    if [ -f ".ci/diagnostics/pytest-collect.tail.txt" ]; then
      echo ""
      echo "== Pytest (collect-only) tail =="
      tail -n 120 ".ci/diagnostics/pytest-collect.tail.txt" 2>/dev/null || true
    fi
  } >"${pytest_file}"
fi

exitcodes_file="${BASE_DIR}/exitcodes.txt"
if compgen -G ".ci/exitcodes/*" >/dev/null; then
  {
    for breadcrumb in .ci/exitcodes/*; do
      [ -f "${breadcrumb}" ] || continue
      printf '%s: %s\n' "$(basename "${breadcrumb}")" "$(cat "${breadcrumb}" 2>/dev/null || echo n/a)"
    done
  } >"${exitcodes_file}" || true
else
  echo "no exitcode breadcrumbs found" >"${exitcodes_file}"
fi

pip_freeze_tail="${BASE_DIR}/pip-freeze-tail.txt"
if [ -f ".ci/diagnostics/pip-freeze.txt" ]; then
  cp ".ci/diagnostics/pip-freeze.txt" "${BASE_DIR}/pip-freeze.txt" 2>/dev/null || true
  {
    echo "Last 40 lines of .ci/diagnostics/pip-freeze.txt"
    echo ""
    tail -n 40 ".ci/diagnostics/pip-freeze.txt" 2>/dev/null || true
    echo ""
    echo "(Full pip freeze captured at .ci/diagnostics/pip-freeze.txt in the diagnostics artifact.)"
  } >"${pip_freeze_tail}" || true
else
  echo "pip freeze output not captured" >"${pip_freeze_tail}"
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

append_section "Pip freeze snapshot"
append_code_block "${pip_freeze_tail}" "pip freeze output unavailable"

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

if compgen -G ".ci/diagnostics/*.tail.txt" >/dev/null; then
  append_section "Command tails (pip-freeze/import-probes/pytest, last 120 lines)"
  for f in .ci/diagnostics/*.tail.txt; do
    printf '```text (%s)\n' "$f" >> "${SUMMARY_MD}"
    cat "$f" >> "${SUMMARY_MD}"
    printf '```\n' >> "${SUMMARY_MD}"
  done
fi

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
  if [ "${MODE}" = "smoke" ]; then
    if printf '%s' "${SMOKE_TOTAL_ROWS_VALUE}" | grep -Eq '^[0-9]+$'; then
      if [ "${SMOKE_TOTAL_ROWS_VALUE}" -ge "${SMOKE_MIN_ROWS}" ] 2>/dev/null; then
        printf 'Smoke assertion: PASS (rows=%s, min=%s)\n\n' "${SMOKE_TOTAL_ROWS_VALUE}" "${SMOKE_MIN_ROWS}" >> "${SUMMARY_MD}"
      else
        printf 'Smoke assertion: FAIL (rows=%s, min=%s)\n\n' "${SMOKE_TOTAL_ROWS_VALUE}" "${SMOKE_MIN_ROWS}" >> "${SUMMARY_MD}"
      fi
    else
      printf 'Smoke assertion: rows=%s, min=%s\n\n' "${SMOKE_TOTAL_ROWS_VALUE}" "${SMOKE_MIN_ROWS}" >> "${SUMMARY_MD}"
    fi
  fi
  cp "${smoke_assert_source}" "${smoke_assert_dest}" 2>/dev/null || true
  append_code_block "${smoke_assert_dest}" "smoke assertion report unavailable"
else
  append_code_block "" "smoke-assert.json not generated"
fi

append_section "Disk usage snapshot"
append_code_block "${usage_file}" "disk usage unavailable"

mkdir -p "${BASE_DIR}/.ci/diagnostics" 2>/dev/null || true
cp "${SUMMARY_MD}" ".ci/diagnostics/${JOB}.SUMMARY.md" 2>/dev/null || true
cp "${SUMMARY_MD}" ".ci/diagnostics/SUMMARY.md" 2>/dev/null || true
cp "${SUMMARY_MD}" "${BASE_DIR}/.ci/diagnostics/SUMMARY.md" 2>/dev/null || true

if [ -n "${GITHUB_ENV:-}" ]; then
  SUMMARY_ABS_PATH="$(pwd)/${SUMMARY_MD}"
  ARTIFACT_ABS_PATH="$(pwd)/${BASE_DIR}"
  {
    echo "ARTIFACT_PATH=${BASE_DIR}"
    echo "ARTIFACT_PATH_ABS=${ARTIFACT_ABS_PATH}"
    echo "SUMMARY_PATH=${SUMMARY_ABS_PATH}"
  } >> "${GITHUB_ENV}"
fi

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  {
    echo "artifact_path=$(pwd)/${BASE_DIR}"
    echo "summary_path=$(pwd)/${SUMMARY_MD}"
    if [ "${MODE}" = "smoke" ]; then
      echo "smoke_total_rows=${SMOKE_TOTAL_ROWS_VALUE}"
    fi
  } >> "${GITHUB_OUTPUT}"
fi

exit 0
