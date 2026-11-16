#!/bin/sh
set -eu
STATE_ROOT=${DEEPUPLIFT_RUNNER_STATE:-"${TMPDIR:-/tmp}/deepuplift-autonomous-runner"}
PID_FILE="$STATE_ROOT/runner.pid"
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "RUNNING pid=$(cat "$PID_FILE") state=$STATE_ROOT"
else
  echo "STOPPED state=$STATE_ROOT"
fi
if [ -f "$STATE_ROOT/runner.sqlite3" ]; then
  python3 - "$STATE_ROOT/runner.sqlite3" <<'PY'
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
rows = db.execute("select status, count(*) from stages group by status order by status").fetchall()
print("LEDGER", ", ".join(f"{status}={count}" for status, count in rows) or "empty")
PY
fi

