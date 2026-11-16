#!/bin/sh
set -eu
USER_HOME_DIR=$(python3 -c 'import os; print(os.path.expanduser("~"))')
STATE_ROOT=${DEEPUPLIFT_RUNNER_ROOT:-"$USER_HOME_DIR/Library/Application Support/DeepUplift/runner"}
PID_FILE="$STATE_ROOT/runner.pid"
PAUSE_FILE="$STATE_ROOT/PAUSED"
touch "$PAUSE_FILE"
LABEL=com.deepuplift.autonomous-runner
if launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  echo "launchd unloaded label=$LABEL"
fi
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

