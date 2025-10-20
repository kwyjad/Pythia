#!/usr/bin/env bash
set -euo pipefail

JOB="${1:-job}"
RUN_ID="${GITHUB_RUN_ID:-local}"
RUN_ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"
BASE_DIR="dist/diag-${JOB}"
ZIP_PATH="dist/diagnostics-${JOB}-${RUN_ID}-${RUN_ATTEMPT}.zip"

mkdir -p "$BASE_DIR"

versions_file="$BASE_DIR/versions.txt"
env_file="$BASE_DIR/env.txt"
git_file="$BASE_DIR/git.txt"
staging_file="$BASE_DIR/staging.txt"
snapshots_file="$BASE_DIR/snapshots.txt"
logs_file="$BASE_DIR/logs.txt"
duckdb_file="$BASE_DIR/duckdb.txt"
summary_file="$BASE_DIR/summary.txt"

{
  echo "== VERSIONS =="
  python -V || true
  pip --version || true
  pip freeze || true
  if command -v duckdb >/dev/null 2>&1; then
    duckdb --version || true
  fi
  uname -a || true
} >"$versions_file" 2>&1 || true

{
  echo "== IMPORTANT ENVS =="
  env | grep -E '^(RESOLVER_|PERIOD_LABEL|GITHUB_|RUNNER_OS)' || true
} >"$env_file" 2>&1 || true

{
  echo "== GIT STATE =="
  git rev-parse HEAD || true
  git status --porcelain || true
  git diff --stat || true
} >"$git_file" 2>&1 || true

{
  echo "== STAGING =="
  ls -R data/staging || true
} >"$staging_file" 2>&1 || true

{
  echo "== SNAPSHOTS =="
  ls -R data/snapshots || true
} >"$snapshots_file" 2>&1 || true

{
  echo "== RESOLVER LOGS =="
  ls -R resolver/logs || true
} >"$logs_file" 2>&1 || true

python - <<'PY' >"$duckdb_file" 2>&1 || true
import os
from pathlib import Path

def describe_db(path: Path) -> None:
    try:
        import duckdb  # type: ignore
    except ModuleNotFoundError:
        print("duckdb not available")
        return
    if not path.exists():
        print(f"database not found: {path}")
        return
    try:
        con = duckdb.connect(str(path))
    except Exception as exc:  # pragma: no cover - best effort
        print(f"failed to open {path}: {exc}")
        return
    try:
        tables = con.sql("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
    except Exception as exc:
        print(f"failed to enumerate tables: {exc}")
        return
    if not tables:
        print("no tables present")
        return
    for (table_name,) in tables:
        try:
            count = con.sql(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"{table_name}: {count}")
        except Exception as exc:
            print(f"{table_name}: failed to count ({exc})")

def main() -> None:
    db_path = Path(os.environ.get("RESOLVER_DB_PATH", "data/resolver.duckdb"))
    describe_db(db_path)

if __name__ == "__main__":
    main()
PY

if [ -f SUMMARY.md ]; then
  cp SUMMARY.md "$summary_file"
fi

pushd dist >/dev/null
rm -f "${ZIP_PATH#dist/}"
zip -rq "${ZIP_PATH#dist/}" "${BASE_DIR#dist/}" || true
popd >/dev/null

echo "ARTIFACT_PATH=${ZIP_PATH}" >> "$GITHUB_ENV"
