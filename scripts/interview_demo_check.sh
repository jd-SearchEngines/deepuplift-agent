#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_PYTHON="python3"
if [[ -z "${PYTHON_BIN:-}" && -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON_BIN="${DEFAULT_PYTHON}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

APP_URL="${APP_URL:-http://localhost:8501}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
ROWS="${ROWS:-180}"
SENSITIVITY_SAMPLES="${SENSITIVITY_SAMPLES:-1}"

wait_for_ui() {
  local attempt
  for attempt in $(seq 1 45); do
    if curl -fsS "${APP_URL}/_stcore/health" >/dev/null 2>&1 || curl -fsS "${APP_URL}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

start_ui_if_needed() {
  if curl -fsS "${APP_URL}/_stcore/health" >/dev/null 2>&1 || curl -fsS "${APP_URL}" >/dev/null 2>&1; then
    echo "UI already reachable at ${APP_URL}"
    return 0
  fi

  echo "Starting Streamlit at ${APP_URL}"
  screen -dmS deepuplift_agent bash -lc "cd '${ROOT_DIR}' && '${PYTHON_BIN}' -m streamlit run app.py --server.headless true --server.port '${STREAMLIT_PORT}' --server.fileWatcherType none > /tmp/deepuplift-streamlit.log 2>&1"
  if ! wait_for_ui; then
    echo "Streamlit did not become reachable. Latest log:"
    tail -n 80 /tmp/deepuplift-streamlit.log || true
    exit 1
  fi
}

print_demo_summary() {
  "${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import glob
import json
from pathlib import Path


def latest(pattern: str) -> Path | None:
    matches = [Path(path) for path in glob.glob(pattern)]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


audit_path = latest("reports/model_catalog_audit_*.json")
validation_path = latest("reports/agent_regression_*.json")
snapshot_path = Path("docs/RESUME_DEEPUplift_AGENT_SNAPSHOT.md")
evidence_index_path = Path("docs/DEEPUplift_AGENT_EVIDENCE_INDEX.md")
story_paths = sorted(
    Path("reports").glob("ui_screenshots_*/story.png"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)

audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path else {}
regression = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path else {}
results = regression.get("results") or []

print("\nDeepUplift Agent interview demo proof")
print("-------------------------------------")
print(f"Registered models: {audit.get('registered_models', 'NA')}")
print(f"Runnable models:   {audit.get('ready_models', 'NA')}")
print(f"Ready backends:    {audit.get('ready_source_counts', {})}")
print(f"Regression runs:   {len(results)}")
if results:
    best = max(results, key=lambda row: row.get("qini") if row.get("qini") is not None else float("-inf"))
    print(f"Example winner:    {best.get('model')} qini={best.get('qini'):.4f} auuc={best.get('auuc'):.4f}")
if story_paths:
    print(f"Story screenshot:  {story_paths[0]}")
if audit_path:
    print(f"Model audit JSON:  {audit_path}")
if snapshot_path.exists():
    print(f"Resume snapshot:   {snapshot_path}")
if evidence_index_path.exists():
    print(f"Evidence index:    {evidence_index_path}")
print("App URL:           http://localhost:8501")
PY
}

start_ui_if_needed

echo "[1/3] Running compact full regression gate"
PYTHON_BIN="${PYTHON_BIN}" ROWS="${ROWS}" SENSITIVITY_SAMPLES="${SENSITIVITY_SAMPLES}" APP_URL="${APP_URL}" scripts/full_regression_check.sh

echo "[2/3] Capturing resume-facing Story page"
"${PYTHON_BIN}" scripts/smoke_ui_screenshots.py --url "${APP_URL}" --tabs Story --min-bytes 20000

echo "[3/3] Re-validating latest artifacts"
"${PYTHON_BIN}" scripts/validate_latest_artifacts.py --require-ui
"${PYTHON_BIN}" scripts/generate_resume_snapshot.py --check
"${PYTHON_BIN}" scripts/generate_cn_project_onepager.py
"${PYTHON_BIN}" scripts/generate_interview_evidence_pack.py
"${PYTHON_BIN}" scripts/generate_model_readiness_trend.py
"${PYTHON_BIN}" scripts/build_interview_pack_zip.py
"${PYTHON_BIN}" scripts/validate_interview_pack.py
"${PYTHON_BIN}" scripts/generate_evidence_index.py

print_demo_summary
