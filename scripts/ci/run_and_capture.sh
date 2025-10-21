#!/usr/bin/env bash
# Run a command, teeing its output to diagnostics logs and capturing exit codes.
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

stdout_log="${diag_dir}/${sanitized_step}.out.log"
stderr_log="${diag_dir}/${sanitized_step}.err.log"
combined_log="${diag_dir}/${sanitized_step}.log"
tail_log="${diag_dir}/${sanitized_step}.tail.txt"

cmd="$*"
status=0
set +e
bash -o pipefail -c "$cmd" \
  > >(tee "$stdout_log") \
  2> >(tee "$stderr_log" >&2)
status=$?
set -euo pipefail

# Combine logs for compatibility with older tooling.
if [ -f "$stderr_log" ] || [ -f "$stdout_log" ]; then
  : > "$combined_log"
  if [ -f "$stderr_log" ]; then
    cat "$stderr_log" >> "$combined_log"
  fi
  if [ -f "$stdout_log" ]; then
    cat "$stdout_log" >> "$combined_log"
  fi
  # Produce a tail snapshot (stderr first, then stdout) limited to the last 120 lines.
  {
    [ -f "$stderr_log" ] && cat "$stderr_log"
    [ -f "$stdout_log" ] && cat "$stdout_log"
  } | tail -n 120 > "$tail_log" || true
else
  : > "$combined_log"
  : > "$tail_log"
fi

printf 'exit=%s\n' "$status" > "${exit_dir}/${sanitized_step}"

exit "$status"
