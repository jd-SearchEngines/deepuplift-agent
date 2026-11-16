#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
USER_HOME_DIR=$(python3 -c 'import os; print(os.path.expanduser("~"))')
STATE_ROOT=${DEEPUPLIFT_RUNNER_ROOT:-"$USER_HOME_DIR/Library/Application Support/DeepUplift/runner"}
PYTHON_BIN=${CODEX_PYTHON:-python3}
LABEL=com.deepuplift.autonomous-runner
LAUNCH_AGENTS="$USER_HOME_DIR/Library/LaunchAgents"
PLIST="$LAUNCH_AGENTS/$LABEL.plist"
TEMPLATE="$ROOT/scripts/autonomous_runner/$LABEL.plist.template"

if ! "$PYTHON_BIN" -c 'import openai_codex' >/dev/null 2>&1; then
  echo "official openai-codex is unavailable in CODEX_PYTHON=$PYTHON_BIN" >&2
  exit 2
fi
mkdir -p "$LAUNCH_AGENTS" "$STATE_ROOT/logs"
rm -f "$STATE_ROOT/PAUSED"
if [ -f "$PLIST" ]; then
  cp "$PLIST" "$PLIST.previous.$(date +%Y%m%d_%H%M%S)"
fi
sed \
  -e "s#REPLACE_WITH_REPO#$ROOT#g" \
  -e "s#REPLACE_WITH_STATE#$STATE_ROOT#g" \
  -e "s#REPLACE_WITH_PYTHON#$PYTHON_BIN#g" \
  "$TEMPLATE" > "$PLIST"
plutil -lint "$PLIST"
launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null || launchctl kickstart -k "gui/$(id -u)/$LABEL"
launchctl enable "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl print "gui/$(id -u)/$LABEL"
echo "installed $PLIST"
