#!/usr/bin/env bash
# Run a command, teeing stdout/stderr to diagnostics and recording exit codes.
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

sanitized_step="$(printf '%s' "$step_name" | tr ' /:\\' '____')"
diag_dir=".ci/diagnostics"
exit_dir=".ci/exitcodes"
mkdir -p "$diag_dir" "$exit_dir"

stdout_log="${diag_dir}/${sanitized_step}.out.log"
stderr_log="${diag_dir}/${sanitized_step}.err.log"
combined_log="${diag_dir}/${sanitized_step}.log"
tail_log="${diag_dir}/${sanitized_step}.tail.txt"

cmd="$*"
status=0
set +e
# Capture both streams separately; preserve pipefail for the inner command.
bash -o pipefail -c "$cmd" \
  > >(tee "$stdout_log") \
  2> >(tee "$stderr_log" >&2)
status=$?
set -euo pipefail

# Combine for legacy readers and create a short tail.
: > "$combined_log"
[ -f "$stderr_log" ] && cat "$stderr_log" >> "$combined_log"
[ -f "$stdout_log" ] && cat "$stdout_log" >> "$combined_log"
{
  [ -f "$stderr_log" ] && cat "$stderr_log"
  [ -f "$stdout_log" ] && cat "$stdout_log"
} | tail -n 120 > "$tail_log" || true

printf 'exit=%s\n' "$status" > "${exit_dir}/${sanitized_step}"
exit "$status"
