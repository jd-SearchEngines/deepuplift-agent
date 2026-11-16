#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
USER_HOME_DIR=$(python3 -c 'import os; print(os.path.expanduser("~"))')
STATE_ROOT=${DEEPUPLIFT_RUNNER_ROOT:-"$USER_HOME_DIR/Library/Application Support/DeepUplift/runner"}
LEDGER=${DEEPUPLIFT_RUNNER_LEDGER:-"$STATE_ROOT/runner.sqlite3"}
BOARD=${DEEPUPLIFT_RUNNER_BOARD:-deepuplift-longterm}
RUN_ID=${DEEPUPLIFT_RUNNER_ID:-deepuplift-longterm}
PYTHON_BIN=${CODEX_PYTHON:-python3}
PID_FILE="$STATE_ROOT/runner.pid"
LOG_DIR="$STATE_ROOT/logs"

mkdir -p "$STATE_ROOT" "$LOG_DIR"
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "already running pid=$(cat "$PID_FILE")"
  exit 0
fi

if ! "$PYTHON_BIN" -c 'import openai_codex' >/dev/null 2>&1; then
  echo "official openai-codex is unavailable in CODEX_PYTHON=$PYTHON_BIN" >&2
  echo "set CODEX_PYTHON to the validated Python 3.10+ SDK environment" >&2
  exit 2
fi

nohup env PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PATH="/Users/jingdong/.local/bin:/opt/homebrew/bin:/usr/bin:/bin${PATH:+:$PATH}" "$PYTHON_BIN" "$ROOT/scripts/run_autonomous_runner.py" \
  --repo-root "$ROOT" --run-id "$RUN_ID" --board "$BOARD" --ledger "$LEDGER" \
  --state-root "$STATE_ROOT" --cycles 0 \
  >>"$LOG_DIR/runner.log" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
echo "started pid=$(cat "$PID_FILE") state=$STATE_ROOT"

