#!/usr/bin/env bash
set -euo pipefail

JOB="${1:-job}"
RUN_ID="${GITHUB_RUN_ID:-local}"
RUN_ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"
DIST_DIR="dist"
BASE_DIR="${DIST_DIR}/diag-${JOB}"
ZIP_NAME="diagnostics-${JOB}-${RUN_ID}-${RUN_ATTEMPT}.zip"
ZIP_PATH="${DIST_DIR}/${ZIP_NAME}"
SUMMARY_MD="${BASE_DIR}/SUMMARY.md"

mkdir -p "${BASE_DIR}"

versions_file="${BASE_DIR}/versions.txt"
env_file="${BASE_DIR}/env.txt"
git_file="${BASE_DIR}/git.txt"
staging_file="${BASE_DIR}/staging.txt"
snapshots_file="${BASE_DIR}/snapshots.txt"
logs_file="${BASE_DIR}/logs.txt"
duckdb_file="${BASE_DIR}/duckdb.txt"
pytest_file="${BASE_DIR}/pytest.txt"

{
  echo "== VERSIONS =="
  (python -V 2>&1 || true)
  (pip --version 2>&1 || true)
  (pip freeze 2>&1 || true)
  if command -v duckdb >/dev/null 2>&1; then
    (duckdb --version 2>&1 || true)
  else
    echo "duckdb binary not available"
  fi
  (uname -a 2>&1 || true)
} >"${versions_file}" || true

python - <<'PY' >"${env_file}" 2>&1 || true
import os
from typing import Iterable

SAFE_PREFIXES = (
    "RESOLVER_",
    "PERIOD_LABEL",
    "GITHUB_",
    "RUNNER_",
    "CI",
    "PYTHON",
)
SENSITIVE_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "KEY", "ACCESS", "CREDENTIAL")

def include_var(name: str) -> bool:
    for prefix in SAFE_PREFIXES:
        if name.startswith(prefix):
            return True
    return False

def redact(value: str) -> str:
    if not value:
        return value
    return "***" if len(value) > 4 else "***"

def is_sensitive(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in SENSITIVE_MARKERS)

def iter_vars(env: dict) -> Iterable[tuple[str, str]]:
    for name in sorted(env):
        if include_var(name):
            value = env[name]
            if is_sensitive(name):
                value = redact(value)
            yield name, value

print("== IMPORTANT ENVS ==")
for key, value in iter_vars(os.environ):
    print(f"{key}={value}")
PY

{
  echo "== GIT STATE =="
  (git rev-parse HEAD 2>&1 || true)
  (git status --porcelain 2>&1 || true)
  (git log -1 --oneline 2>&1 || true)
} >"${git_file}" || true

{
  echo "== STAGING =="
  if [ -d data/staging ]; then
    (ls -R data/staging 2>&1 || true)
  else
    echo "data/staging missing"
  fi
} >"${staging_file}" || true

{
  echo "== SNAPSHOTS =="
  if [ -d data/snapshots ]; then
    (ls -R data/snapshots 2>&1 || true)
  else
    echo "data/snapshots missing"
  fi
} >"${snapshots_file}" || true

{
  echo "== RESOLVER LOGS =="
  handled=false
  for candidate in resolver/logs data/logs; do
    if [ -d "${candidate}" ]; then
      handled=true
      while IFS= read -r -d '' log_path; do
        echo "--- ${log_path}"
        (tail -n 200 "${log_path}" 2>&1 || true)
      done < <(find "${candidate}" -type f -print0)
    fi
  done
  if [ "${handled}" = false ]; then
    echo "resolver/logs missing"
    echo "data/logs missing"
  fi
} >"${logs_file}" || true

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

python - <<'PY' >"${duckdb_file}" 2>&1 || true
import os
from pathlib import Path

def main() -> None:
    db_path = Path(os.environ.get("RESOLVER_DB_PATH", "data/resolver.duckdb"))
    try:
        import duckdb  # type: ignore
    except ModuleNotFoundError:
        print("duckdb module not available")
        return
    if not db_path.exists():
        print(f"database not found: {db_path}")
        return
    con = duckdb.connect(str(db_path))
    try:
        tables = [row[0] for row in con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()]
    except Exception as exc:
        print(f"failed to enumerate tables: {exc}")
        return
    if not tables:
        print("no tables present")
        return
    for table in tables:
        try:
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except Exception as exc:
            print(f"{table}: failed to count ({exc})")
        else:
            print(f"{table}: {count}")

if __name__ == "__main__":
    main()
PY

# Build SUMMARY.md
{
  echo "# Diagnostics Summary — ${JOB}"
  echo ""
  echo "Generated: $(date -u '+%Y-%m-%d %H:%M:%SZ')"
  echo "Run ID: ${RUN_ID} (attempt ${RUN_ATTEMPT})"
  echo ""
  echo "## Git"
  echo '```'
  cat "${git_file}" 2>/dev/null || true
  echo '```'
  echo ""
  echo "## Versions"
  echo '```'
  cat "${versions_file}" 2>/dev/null || true
  echo '```'
  echo ""
  echo "## Environment"
  echo '```'
  cat "${env_file}" 2>/dev/null || true
  echo '```'
  echo ""
  echo "## DuckDB"
  echo '```'
  cat "${duckdb_file}" 2>/dev/null || true
  echo '```'
  echo ""
  echo "## Staging directory"
  echo '```'
  cat "${staging_file}" 2>/dev/null | head -n 200 || true
  echo '```'
  echo ""
  echo "## Snapshots directory"
  echo '```'
  cat "${snapshots_file}" 2>/dev/null | head -n 200 || true
  echo '```'
  echo ""
  echo "## Pytest summary"
  echo '```'
  cat "${pytest_file}" 2>/dev/null || true
  echo '```'
  echo ""
  echo "## Resolver logs (tail)"
  echo '```'
  cat "${logs_file}" 2>/dev/null | tail -n 200 || true
  echo '```'
} >"${SUMMARY_MD}" || true

mkdir -p "${DIST_DIR}"
pushd "${DIST_DIR}" >/dev/null
rm -f "${ZIP_NAME}"
zip -rq "${ZIP_NAME}" "${BASE_DIR#${DIST_DIR}/}" || true
popd >/dev/null

if [ -n "${GITHUB_ENV:-}" ]; then
  {
    echo "ARTIFACT_PATH=${ZIP_PATH}"
    echo "SUMMARY_PATH=${SUMMARY_MD}"
  } >> "${GITHUB_ENV}"
fi
