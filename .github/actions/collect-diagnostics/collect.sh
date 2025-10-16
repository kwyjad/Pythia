#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[collect-diagnostics] %s\n' "$*" >&2
}

JOB_NAME="${INPUT_JOB_NAME:-}"
if [[ -z "$JOB_NAME" ]]; then
  log "INPUT_JOB_NAME is required"
  exit 1
fi

RUN_ID="${GITHUB_RUN_ID:-local}"
RUN_ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"
DEFAULT_ARTIFACT="diagnostics-${JOB_NAME}-${RUN_ID}-${RUN_ATTEMPT}"
ARTIFACT_NAME="${INPUT_ARTIFACT_NAME:-$DEFAULT_ARTIFACT}"

WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"
DIST_DIR="${WORKSPACE}/dist"
PAYLOAD_ROOT="${DIST_DIR}/diagnostics"
PAYLOAD_DIR="${PAYLOAD_ROOT}/${JOB_NAME}"

# Ensure predictable permissions
umask 077

mkdir -p "$PAYLOAD_DIR"

# Utility helpers -----------------------------------------------------------

capture_cmd() {
  local file="$1"
  shift
  mkdir -p "$(dirname "$file")"
  (
    set +e
    echo "\$ $*"
    "$@"
    local status=$?
    if [[ $status -ne 0 ]]; then
      echo
      echo "Command exited with status $status"
    fi
    exit 0
  ) &>"$file"
}

write_text() {
  local file="$1"
  shift
  mkdir -p "$(dirname "$file")"
  printf '%s\n' "$@" >"$file"
}

redact_env() {
  local file="$1"
  mkdir -p "$(dirname "$file")"
  env | sort | while IFS= read -r line; do
    if [[ -z "$line" ]]; then
      echo
      continue
    fi
    local key="${line%%=*}"
    local value="${line#*=}"
    if [[ "$key" =~ (TOKEN|PASSWORD|SECRET|KEY|COOKIE|PWD|PASS|CREDENTIAL|PRIVATE|CERT|ACCESS) ]]; then
      printf '%s=%s\n' "$key" "***REDACTED***"
    else
      printf '%s=%s\n' "$key" "$value"
    fi
  done >"$file"
}

copy_if_exists() {
  local source="$1"
  local dest="$2"
  if [[ -f "$source" ]]; then
    mkdir -p "$(dirname "$dest")"
    cp "$source" "$dest" 2>/dev/null || true
  fi
}

collect_tree() {
  local source="$1"
  local label="$2"
  if [[ ! -d "$source" ]]; then
    return
  fi
  local dest="$PAYLOAD_DIR/$label"
  mkdir -p "$dest/files"

  # Capture directory listing (depth limited for readability)
  (
    set +e
    find "$source" -maxdepth 5 -print
    exit 0
  ) >"$dest/listing.txt" 2>&1 || true

  # Copy individual files up to 50 MiB each for quick inspection
  (
    set +e
    find "$source" -type f -size -52428800c -print0
    exit 0
  ) | while IFS= read -r -d '' file; do
    local rel
    rel="${file#$source/}"
    if [[ "$rel" == "$file" ]]; then
      rel="$(basename "$file")"
    fi
    local dest_file="$dest/files/$rel"
    mkdir -p "$(dirname "$dest_file")"
    cp "$file" "$dest_file" 2>/dev/null || true
  done

  (
    set +e
    find "$source" -type f -size +52428800c -print
    exit 0
  ) >"$dest/skipped-large-files.txt" 2>&1 || true
}

# Summary ------------------------------------------------------------------
write_text "$PAYLOAD_DIR/SUMMARY.md" \
  "Diagnostics bundle for job: $JOB_NAME" \
  "Generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
  "Workspace: $WORKSPACE" \
  "Run URL: ${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-unknown}/actions/runs/${RUN_ID}" \
  "Artifact name: $ARTIFACT_NAME"

# System information -------------------------------------------------------
SYSTEM_DIR="$PAYLOAD_DIR/system"
if command -v python >/dev/null 2>&1; then
  capture_cmd "$SYSTEM_DIR/python-version.txt" python --version
  capture_cmd "$SYSTEM_DIR/pip-freeze.txt" python -m pip freeze
else
  write_text "$SYSTEM_DIR/python-version.txt" "python command not available"
fi

if command -v duckdb >/dev/null 2>&1; then
  capture_cmd "$SYSTEM_DIR/duckdb-version.txt" duckdb --version
else
  write_text "$SYSTEM_DIR/duckdb-version.txt" "duckdb CLI not available"
fi

if command -v uname >/dev/null 2>&1; then
  capture_cmd "$SYSTEM_DIR/uname.txt" uname -a
fi

# Environment snapshots ----------------------------------------------------
ENV_DIR="$PAYLOAD_DIR/env"
redact_env "$ENV_DIR/environment.txt"
(
  set +e
  env | sort | grep '^RESOLVER_' || true
  exit 0
) >"$ENV_DIR/resolver-env.txt"

# Git repository state -----------------------------------------------------
GIT_DIR="$PAYLOAD_DIR/git"
if command -v git >/dev/null 2>&1; then
  capture_cmd "$GIT_DIR/head.txt" git rev-parse HEAD
  capture_cmd "$GIT_DIR/status.txt" git status --short --branch
  capture_cmd "$GIT_DIR/diffstat.txt" git diff --stat
  capture_cmd "$GIT_DIR/diff.patch" git diff
else
  write_text "$GIT_DIR/info.txt" "git command not available"
fi

# Test outputs -------------------------------------------------------------
TEST_DIR="$PAYLOAD_DIR/tests"
copy_if_exists "pytest-junit.xml" "$TEST_DIR/pytest-junit.xml"
copy_if_exists "coverage.xml" "$TEST_DIR/coverage.xml"
if [[ -d "htmlcov" ]]; then
  collect_tree "htmlcov" "tests/htmlcov"
fi
if [[ -d ".pytest_cache" ]]; then
  collect_tree ".pytest_cache" "tests/pytest_cache"
fi

# Pipeline artifacts -------------------------------------------------------
collect_tree "resolver/logs" "resolver/logs"
collect_tree "data/staging" "data/staging"
collect_tree "data/snapshots" "data/snapshots"
collect_tree "resolver/output" "resolver/output"
collect_tree "logs" "logs"

# DuckDB inspection --------------------------------------------------------
DUCKDB_REPORT="$PAYLOAD_DIR/duckdb/summary.txt"
mkdir -p "$(dirname "$DUCKDB_REPORT")"
if command -v python >/dev/null 2>&1; then
  set +e
  python - <<'PY' >"$DUCKDB_REPORT" 2>&1
import os
import pathlib
import sys
from datetime import datetime

try:
    import duckdb
except Exception as exc:  # pragma: no cover - environment dependent
    print(f"duckdb module unavailable: {exc}")
    raise SystemExit(0)

print(f"duckdb.__version__ = {duckdb.__version__}")
print(f"generated at {datetime.utcnow().isoformat()}Z")

workspace = pathlib.Path(os.environ.get("GITHUB_WORKSPACE", os.getcwd()))

candidates = []
for pattern in ("*.duckdb", "*.db", "*.ddb"):
    candidates.extend(workspace.rglob(pattern))

if not candidates:
    snapshots = workspace.joinpath("data", "snapshots")
    if snapshots.exists():
        candidates.extend(p for p in snapshots.rglob("*.duckdb"))

if not candidates:
    print("No duckdb-like files discovered")
    raise SystemExit(0)

for path in sorted(set(candidates)):
    if not path.is_file():
        continue
    print(f"Inspecting {path.relative_to(workspace)}")
    try:
        con = duckdb.connect(str(path))
    except Exception as exc:  # pragma: no cover - best effort
        print(f"  failed to connect: {exc}")
        continue
    try:
        print("  PRAGMA database_list")
        for row in con.execute("PRAGMA database_list").fetchall():
            print(f"    {row}")
        print("  SHOW TABLES")
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        if not tables:
            print("    (no tables)")
            continue
        for table in tables:
            print(f"    table {table}")
            try:
                result = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                print(f"      row_count={result[0] if result else 'unknown'}")
            except Exception as exc:  # pragma: no cover - defensive
                print(f"      failed to count rows: {exc}")
    finally:
        con.close()
PY
  status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    echo >>"$DUCKDB_REPORT"
    echo "DuckDB introspection exited with status $status (non-fatal)" >>"$DUCKDB_REPORT"
  fi
fi

# Final manifest -----------------------------------------------------------
MANIFEST="$PAYLOAD_DIR/manifest.txt"
(
  set +e
  find "$PAYLOAD_DIR" -maxdepth 6 -type f | sort
  exit 0
) >"$MANIFEST" 2>&1 || true

mkdir -p "$DIST_DIR"
ARCHIVE_PATH="$DIST_DIR/${ARTIFACT_NAME}.zip"
PAYLOAD_BASENAME="$(basename "$PAYLOAD_DIR")"

set +e
(
  cd "$DIST_DIR"
  python - "$PAYLOAD_BASENAME" "$ARCHIVE_PATH" <<'PY'
import sys
import zipfile
from pathlib import Path

payload_name, archive_path = sys.argv[1:3]
payload_dir = Path('diagnostics') / payload_name
archive = Path(archive_path)
if not payload_dir.exists():
    raise SystemExit(f"payload directory missing: {payload_dir}")
if archive.exists():
    archive.unlink()
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for path in payload_dir.rglob('*'):
        if path.is_file():
            arcname = Path('diagnostics') / path.relative_to(payload_dir.parent)
            zf.write(path, arcname)
        elif path.is_dir() and not any(path.iterdir()):
            arcname = Path('diagnostics') / path.relative_to(payload_dir.parent)
            zf.writestr(str(arcname) + '/', '')
print(f"Wrote {archive}")
PY
)
status=$?
set -e
if [[ $status -ne 0 ]]; then
  log "Python zip creation failed (status $status); falling back to zip CLI"
  (
    cd "$DIST_DIR"
    set +e
    zip -r "${ARTIFACT_NAME}.zip" "diagnostics/${PAYLOAD_BASENAME}"
    status=$?
    set -e
    if [[ $status -ne 0 ]]; then
      log "Failed to create diagnostics archive"
      exit $status
    fi
  )
fi

log "Diagnostics archive created at $ARCHIVE_PATH"
