#!/bin/sh
set -eu
STATE_ROOT=${DEEPUPLIFT_RUNNER_STATE:-"${TMPDIR:-/tmp}/deepuplift-autonomous-runner"}
PID_FILE="$STATE_ROOT/runner.pid"
if [ ! -f "$PID_FILE" ]; then
  echo "not running (no pid file)"
  exit 0
fi
PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
  kill -TERM "$PID"
  echo "stop requested pid=$PID"
else
  echo "stale pid file pid=$PID"
fi
rm -f "$PID_FILE"

