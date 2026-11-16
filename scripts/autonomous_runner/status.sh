#!/bin/sh
set -eu
USER_HOME_DIR=$(python3 -c 'import os; print(os.path.expanduser("~"))')
STATE_ROOT=${DEEPUPLIFT_RUNNER_ROOT:-"$USER_HOME_DIR/Library/Application Support/DeepUplift/runner"}
PID_FILE="$STATE_ROOT/runner.pid"
LABEL=com.deepuplift.autonomous-runner
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "RUNNING pid=$(cat "$PID_FILE") state=$STATE_ROOT"
else
  echo "NO_MANUAL_PID state=$STATE_ROOT"
fi
if launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; then
  echo "LAUNCHD loaded label=$LABEL"
  launchctl print "gui/$(id -u)/$LABEL" | awk '/^[[:space:]]*pid = / {print "LAUNCHD_PID", $3; found=1} END {if (!found) print "LAUNCHD_PID none"}'
else
  echo "LAUNCHD not_loaded label=$LABEL"
fi
if [ -f "$STATE_ROOT/runner.sqlite3" ]; then
  python3 - "$STATE_ROOT/runner.sqlite3" <<'PY'
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
db.row_factory = sqlite3.Row
rows = db.execute("select status, count(*) as n from stages group by status order by status").fetchall()
print("LEDGER", ", ".join(f"{row['status']}={row['n']}" for row in rows) or "empty")
active = db.execute("select phase_id, heartbeat_at, worker_pid from stages where status='RUNNING' order by updated_at desc limit 1").fetchone()
ready = [row[0] for row in db.execute("select phase_id from stages where status in ('READY','REPAIR_READY') order by updated_at")]
accepted = db.execute("select phase_id from stages where status='ACCEPTED' order by updated_at desc limit 1").fetchone()
blocked = [row[0] for row in db.execute("select phase_id from stages where status='BLOCKED' order by updated_at")]
print("ACTIVE", dict(active) if active else None)
print("READY", ready)
print("LAST_ACCEPTED", accepted[0] if accepted else None)
print("BLOCKED", blocked)
PY
fi

