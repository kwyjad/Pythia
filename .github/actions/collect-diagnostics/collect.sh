#!/usr/bin/env bash
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

{
  echo "## Resolver API-related versions (if present)"
  python - <<'PY' || true
import importlib
for m in ("fastapi","starlette","uvicorn","websockets"):
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, '__version__', 'n/a')
        print(f"{m}=={ver}")
    except Exception as e:
        print(f"{m}: not installed ({e.__class__.__name__})")
PY
} >> "$GITHUB_STEP_SUMMARY" 2>/dev/null || true
# Emit step outputs for composite action
SUMMARY_PATH="${SUMMARY_PATH:-${ART_DIR}/SUMMARY.md}"
{
  echo "artifact_path=${ART_DIR}"
  echo "summary_path=${SUMMARY_PATH}"
} >> "${GITHUB_OUTPUT}"
