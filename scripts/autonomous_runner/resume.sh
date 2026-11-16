#!/bin/sh
set -eu
USER_HOME_DIR=$(python3 -c 'import os; print(os.path.expanduser("~"))')
STATE_ROOT=${DEEPUPLIFT_RUNNER_ROOT:-"$USER_HOME_DIR/Library/Application Support/DeepUplift/runner"}
rm -f "$STATE_ROOT/PAUSED"
LABEL=com.deepuplift.autonomous-runner
PLIST="$USER_HOME_DIR/Library/LaunchAgents/$LABEL.plist"
if [ -f "$PLIST" ]; then
  launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null || launchctl kickstart -k "gui/$(id -u)/$LABEL"
else
  "$(dirname -- "$0")/start.sh"
fi
"$(dirname -- "$0")/status.sh"

