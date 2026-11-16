#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
USER_HOME_DIR=$(python3 -c 'import os; print(os.path.expanduser("~"))')
STATE_ROOT=${DEEPUPLIFT_RUNNER_ROOT:-"$USER_HOME_DIR/Library/Application Support/DeepUplift/runner"}
LEDGER=${DEEPUPLIFT_RUNNER_LEDGER:-"$STATE_ROOT/runner.sqlite3"}
BOARD=${DEEPUPLIFT_RUNNER_BOARD:-deepuplift-longterm}
RUN_ID=${DEEPUPLIFT_RUNNER_ID:-deepuplift-longterm}
PYTHON_BIN=${CODEX_PYTHON:-python3}
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$ROOT/scripts/run_autonomous_runner.py" \
  --repo-root "$ROOT" --run-id "$RUN_ID" --board "$BOARD" --ledger "$LEDGER" \
  --state-root "$STATE_ROOT" --plan-only

