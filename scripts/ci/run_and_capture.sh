#!/usr/bin/env bash
# Run a command, teeing its output to a diagnostics log and recording the exit code.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <step_name> <command...>" >&2
  exit 2
fi

step_name="$1"
shift
sanitized_step=$(printf '%s' "$step_name" | tr ' /:\\' '____')
diag_dir=".ci/diagnostics"
exit_dir=".ci/exitcodes"
mkdir -p "$diag_dir" "$exit_dir"
log_file="${diag_dir}/${sanitized_step}.log"

cmd="$*"
status=0
set +e
bash -lc "$cmd" |& tee "$log_file"
status=${PIPESTATUS[0]}
set -euo pipefail
printf 'exit=%s\n' "$status" > "${exit_dir}/${sanitized_step}"
exit "$status"
