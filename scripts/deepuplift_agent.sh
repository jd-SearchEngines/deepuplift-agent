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

usage() {
  cat <<'EOF'
DeepUplift Agent launcher

Usage:
  scripts/deepuplift_agent.sh start       Start or reuse Streamlit on localhost:8501
  scripts/deepuplift_agent.sh status      Check UI health and latest artifact pointers
  scripts/deepuplift_agent.sh story       Capture the Story page screenshot
  scripts/deepuplift_agent.sh regression  Run the compact full regression gate
  scripts/deepuplift_agent.sh snapshot    Refresh the live resume snapshot
  scripts/deepuplift_agent.sh evidence    Refresh the evidence index
  scripts/deepuplift_agent.sh pack        Build and validate interview handoff ZIP
  scripts/deepuplift_agent.sh demo-lite   Refresh interview docs/screenshots without training smoke
  scripts/deepuplift_agent.sh demo        Run the one-command interview demo
EOF
}

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

start_app() {
  if curl -fsS "${APP_URL}/_stcore/health" >/dev/null 2>&1 || curl -fsS "${APP_URL}" >/dev/null 2>&1; then
    echo "UI already reachable at ${APP_URL}"
    return 0
  fi
  echo "Starting Streamlit at ${APP_URL}"
  screen -dmS deepuplift_agent bash -lc "cd '${ROOT_DIR}' && '${PYTHON_BIN}' -m streamlit run app.py --server.headless true --server.port '${STREAMLIT_PORT}' --server.fileWatcherType none > /tmp/deepuplift-streamlit.log 2>&1"
  wait_for_ui
}

status() {
  if curl -fsS "${APP_URL}/_stcore/health" >/dev/null 2>&1 || curl -fsS "${APP_URL}" >/dev/null 2>&1; then
    echo "UI: ok (${APP_URL})"
  else
    echo "UI: not reachable (${APP_URL})"
  fi
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
patterns = {
    "model_audit": "reports/model_catalog_audit_*.json",
    "agent_regression": "reports/agent_regression_*.json",
    "no_ui_compare": "reports/no_ui_compare_manifest_*.json",
    "screenshots": "reports/ui_screenshots_*",
    "resume_snapshot": "docs/RESUME_DEEPUplift_AGENT_SNAPSHOT.md",
    "agent_transcript_sample": "docs/AGENT_INTERVIEW_TRANSCRIPT_SAMPLE.md",
    "industry_playbook": "docs/INDUSTRIAL_UPLIFT_CASE_STUDIES.md",
    "industry_source_audit": "docs/INDUSTRY_SOURCE_QUALITY_AUDIT.md",
    "industry_source_backlog": "docs/UPLIFT_INDUSTRY_SOURCE_ADOPTION_BACKLOG.md",
    "scenario_model_matrix": "docs/UPLIFT_SCENARIO_MODEL_MATRIX.md",
    "scenario_model_matrix_audit": "reports/scenario_model_matrix_audit_latest.json",
    "evidence_index": "docs/DEEPUplift_AGENT_EVIDENCE_INDEX.md",
    "interview_pack": "reports/deepuplift_agent_interview_pack_latest.zip",
}
for label, pattern in patterns.items():
    paths = sorted(Path(".").glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    print(f"{label}: {paths[0] if paths else 'missing'}")
story_paths = sorted(Path(".").glob("reports/ui_screenshots_*/story.png"), key=lambda path: path.stat().st_mtime, reverse=True)
refs = [
    Path("docs/DEEPUplift_AGENT_ONEPAGER_CN.md"),
    Path("docs/DEEPUplift_AGENT_INTERVIEW_EVIDENCE_PACK.md"),
    Path("docs/DEEPUplift_AGENT_EVIDENCE_INDEX.md"),
    Path("reports/deepuplift_agent_interview_pack_latest.zip"),
]
existing_refs = [path for path in refs if path.exists()]
if story_paths and existing_refs:
    newest_ref = max(existing_refs, key=lambda path: path.stat().st_mtime)
    freshness = "fresh" if story_paths[0].stat().st_mtime >= newest_ref.stat().st_mtime else "stale"
    print(f"story_freshness: {freshness} ({story_paths[0]} vs {newest_ref})")
else:
    print("story_freshness: unknown")
PY
}

command="${1:-help}"
case "${command}" in
  start)
    start_app
    ;;
  status)
    status
    ;;
  story)
    start_app
    "${PYTHON_BIN}" scripts/smoke_ui_screenshots.py --url "${APP_URL}" --tabs Story --min-bytes 20000
    ;;
  regression)
    start_app
    PYTHON_BIN="${PYTHON_BIN}" ROWS="${ROWS}" SENSITIVITY_SAMPLES="${SENSITIVITY_SAMPLES}" APP_URL="${APP_URL}" scripts/full_regression_check.sh
    ;;
  snapshot)
    "${PYTHON_BIN}" scripts/generate_resume_snapshot.py --check
    "${PYTHON_BIN}" scripts/generate_industry_playbook_docs.py
    "${PYTHON_BIN}" scripts/generate_industry_source_backlog.py
    "${PYTHON_BIN}" scripts/audit_industry_sources.py --json reports/industry_source_audit_latest.json
    "${PYTHON_BIN}" scripts/generate_cn_project_onepager.py
    "${PYTHON_BIN}" scripts/generate_agent_interview_transcript_sample.py
    "${PYTHON_BIN}" scripts/generate_scenario_model_matrix.py
    "${PYTHON_BIN}" scripts/audit_scenario_model_matrix.py --json reports/scenario_model_matrix_audit_latest.json
    ;;
  evidence)
    "${PYTHON_BIN}" scripts/generate_cn_project_onepager.py
    "${PYTHON_BIN}" scripts/generate_industry_playbook_docs.py
    "${PYTHON_BIN}" scripts/generate_industry_source_backlog.py
    "${PYTHON_BIN}" scripts/audit_industry_sources.py --json reports/industry_source_audit_latest.json
    "${PYTHON_BIN}" scripts/generate_agent_interview_transcript_sample.py
    "${PYTHON_BIN}" scripts/generate_interview_evidence_pack.py
    "${PYTHON_BIN}" scripts/generate_model_readiness_trend.py
    "${PYTHON_BIN}" scripts/generate_scenario_model_matrix.py
    "${PYTHON_BIN}" scripts/audit_scenario_model_matrix.py --json reports/scenario_model_matrix_audit_latest.json
    "${PYTHON_BIN}" scripts/build_interview_pack_zip.py
    "${PYTHON_BIN}" scripts/validate_interview_pack.py
    "${PYTHON_BIN}" scripts/generate_evidence_index.py
    ;;
  pack)
    "${PYTHON_BIN}" scripts/generate_industry_playbook_docs.py
    "${PYTHON_BIN}" scripts/generate_industry_source_backlog.py
    "${PYTHON_BIN}" scripts/audit_industry_sources.py --json reports/industry_source_audit_latest.json
    "${PYTHON_BIN}" scripts/generate_agent_interview_transcript_sample.py
    "${PYTHON_BIN}" scripts/generate_scenario_model_matrix.py
    "${PYTHON_BIN}" scripts/audit_scenario_model_matrix.py --json reports/scenario_model_matrix_audit_latest.json
    "${PYTHON_BIN}" scripts/generate_interview_evidence_pack.py
    "${PYTHON_BIN}" scripts/build_interview_pack_zip.py
    "${PYTHON_BIN}" scripts/validate_interview_pack.py
    ;;
  demo-lite)
    PYTHON_BIN="${PYTHON_BIN}" APP_URL="${APP_URL}" STREAMLIT_PORT="${STREAMLIT_PORT}" scripts/demo_lite_check.sh
    ;;
  demo)
    PYTHON_BIN="${PYTHON_BIN}" ROWS="${ROWS}" SENSITIVITY_SAMPLES="${SENSITIVITY_SAMPLES}" APP_URL="${APP_URL}" STREAMLIT_PORT="${STREAMLIT_PORT}" scripts/interview_demo_check.sh
    ;;
  help|--help|-h)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
