#!/bin/sh
set -eu
USER_HOME_DIR=$(python3 -c 'import os; print(os.path.expanduser("~"))')
STATE_ROOT=${DEEPUPLIFT_RUNNER_ROOT:-"$USER_HOME_DIR/Library/Application Support/DeepUplift/runner"}
LOG_FILE="$STATE_ROOT/logs/runner.log"
if [ ! -f "$LOG_FILE" ]; then
  echo "runner.log is not used by direct launchd mode; showing launchd logs"
  for file in "$STATE_ROOT/logs/launchd.stdout.log" "$STATE_ROOT/logs/launchd.stderr.log"; do
    if [ -f "$file" ]; then
      echo "--- $file"
      tail -n "${1:-100}" "$file"
    fi
  done
  exit 0
fi
if [ "${1:-}" = "--follow" ]; then
  tail -f "$LOG_FILE"
else
  tail -n "${1:-100}" "$LOG_FILE"
fi
