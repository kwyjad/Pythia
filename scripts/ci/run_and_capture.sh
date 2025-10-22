#!/usr/bin/env bash
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
