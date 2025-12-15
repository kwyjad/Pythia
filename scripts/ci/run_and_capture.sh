#!/usr/bin/env bash
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

# Run a command, teeing its stdout/stderr to diagnostics logs and recording exit codes.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <step_name> <command...>" >&2
  exit 2
fi

step_name="$1"
shift
if [ -z "$step_name" ]; then
  echo "step_name must be non-empty" >&2
  exit 2
fi

sanitized_step=$(printf '%s' "$step_name" | tr ' /:\\' '____')
diag_dir=".ci/diagnostics"
exit_dir=".ci/exitcodes"
mkdir -p "$diag_dir" "$exit_dir"

log_base="${diag_dir}/${sanitized_step}"
stdout_log="${log_base}.out.log"
stderr_log="${log_base}.err.log"
combined_log="${log_base}.log"
tail_log="${log_base}.tail.txt"

cmd="$*"

# Provide a defensive fallback for pytest parallel flags when pytest-xdist is
# unavailable. This keeps the fast CI job running even if the plugin was not
# installed (for example, during local smoke tests) while still honoring the
# caller's request whenever xdist is present.
if [[ "$cmd" == pytest* ]] && [[ "$cmd" == *"-n auto --dist=worksteal"* ]]; then
  if ! pytest --help 2>/dev/null | grep -q "-n NUM"; then
    echo "pytest-xdist not detected; running serial" >&2
    export RUN_AND_CAPTURE_COMMAND="$cmd"
    cmd=$(python <<'PY'
import os
import shlex

original = os.environ["RUN_AND_CAPTURE_COMMAND"]
parts = shlex.split(original)
cleaned = []
i = 0
while i < len(parts):
    part = parts[i]
    if part == "-n" and i + 1 < len(parts) and parts[i + 1] == "auto":
        i += 2
        continue
    if part == "--dist=worksteal":
        i += 1
        continue
    if part == "--dist" and i + 1 < len(parts) and parts[i + 1] == "worksteal":
        i += 2
        continue
    cleaned.append(part)
    i += 1

print(shlex.join(cleaned))
PY
)
    unset RUN_AND_CAPTURE_COMMAND
  fi
fi

status=0
set +e
bash -o pipefail -c "$cmd" \
  > >(tee "$stdout_log") \
  2> >(tee "$stderr_log" >&2)
status=$?
set -euo pipefail

# Combine logs (stderr first so errors float to the top).
: > "$combined_log"
[ -f "$stderr_log" ] && cat "$stderr_log" >> "$combined_log"
[ -f "$stdout_log" ] && cat "$stdout_log" >> "$combined_log"

# Tail snapshot
{
  [ -f "$stderr_log" ] && cat "$stderr_log"
  [ -f "$stdout_log" ] && cat "$stdout_log"
} | tail -n 120 > "$tail_log" || true

printf 'exit=%s\n' "$status" > "${exit_dir}/${sanitized_step}"
exit "$status"
