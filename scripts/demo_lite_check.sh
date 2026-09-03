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
PAPER_BENCH_PRESET="${PAPER_BENCH_PRESET:-paper-lite}"
PAPER_BENCH_ROWS="${PAPER_BENCH_ROWS:-220}"
DEEP_TUNING_PRESET="${DEEP_TUNING_PRESET:-tuned-smoke}"
DEEP_TUNING_ROWS="${DEEP_TUNING_ROWS:-160}"

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

start_ui_if_needed

echo "[1/5] Compile app and demo scripts"
"${PYTHON_BIN}" -m py_compile \
  app.py \
  deepuplift/core/frontier_models.py \
  deepuplift/core/deep_uplift_designs.py \
  deepuplift/core/industrial_scenario_lab.py \
  deepuplift/core/resume_scenario_policy.py \
  deepuplift/core/resume_scenario_workbench.py \
  deepuplift/core/industry_playbook.py \
  deepuplift/core/open_source_frameworks.py \
  deepuplift/core/evaluation_formulas.py \
  deepuplift/core/failure_modes.py \
  deepuplift/core/model_deconstruction.py \
  deepuplift/core/interview_agent.py \
  deepuplift/models/deep_losses.py \
  deepuplift/models/ContrastiveUpliftNet.py \
  scripts/audit_interview_materials.py \
  scripts/audit_industry_playbook.py \
  scripts/audit_industry_sources.py \
  scripts/audit_model_catalog.py \
  scripts/audit_scenario_model_matrix.py \
  scripts/build_interview_pack_zip.py \
  scripts/generate_agent_interview_transcript_sample.py \
  scripts/generate_industry_playbook_docs.py \
  scripts/generate_industry_source_backlog.py \
  scripts/generate_industrial_scenario_datasets.py \
  scripts/generate_cn_project_onepager.py \
  scripts/generate_deep_loss_architecture_docs.py \
  scripts/generate_evaluation_framework_docs.py \
  scripts/generate_failure_mode_docs.py \
  scripts/generate_uplift_diagnostic_cases.py \
  scripts/generate_evidence_index.py \
  scripts/generate_frontier_synthetic_datasets.py \
  scripts/generate_glossary_doc.py \
  scripts/generate_interview_evidence_pack.py \
  scripts/generate_model_readiness_trend.py \
  scripts/generate_model_deconstruction_docs.py \
  scripts/generate_paper_benchmark_datasets.py \
  scripts/generate_paper_benchmark_interpretation.py \
  scripts/import_official_acic_dataset.py \
  scripts/generate_scenario_model_matrix.py \
  scripts/probe_optional_uplift_backends.py \
  scripts/smoke_demo_preset_source_trace.py \
  scripts/smoke_cfrnet_differentiable_balance.py \
  scripts/smoke_cfrnet_training_balance.py \
  scripts/smoke_contrastive_uplift_training.py \
  scripts/smoke_deep_neural_ablation.py \
  scripts/smoke_deep_model_training_evidence.py \
  scripts/smoke_deep_loss_architecture_catalog.py \
  scripts/smoke_deep_loss_functions.py \
  scripts/smoke_evaluation_formula_catalog.py \
  scripts/smoke_uplift_failure_modes.py \
  scripts/smoke_open_source_framework_audit.py \
  scripts/smoke_utboost_guarded_adapter.py \
  scripts/smoke_industry_agent_answers.py \
  scripts/smoke_llm_routing_dataset.py \
  scripts/smoke_agent_interview_answers.py \
  scripts/smoke_model_deconstruction_agent.py \
  scripts/run_deep_model_tuned_benchmark.py \
  scripts/run_paper_benchmark_suite.py \
  scripts/smoke_frontier_evaluator_metrics.py \
  scripts/smoke_frontier_model_capabilities.py \
  scripts/smoke_frontier_synthetic_datasets.py \
  scripts/smoke_industrial_scenario_datasets.py \
  scripts/smoke_industrial_scenario_training.py \
  scripts/smoke_industrial_policy_metrics.py \
  scripts/smoke_industrial_scenario_compare.py \
  scripts/smoke_resume_scenario_policy.py \
  scripts/smoke_resume_scenario_workbench.py \
  scripts/smoke_agent_interview_ui.py \
  scripts/smoke_ui_screenshots.py \
  scripts/summarize_frontier_training_evidence.py \
  scripts/validate_interview_pack.py \
  scripts/validate_story_freshness.py
bash -n scripts/demo_lite_check.sh

echo "[2/5] Refresh model catalog audit and docs without training"
"${PYTHON_BIN}" scripts/audit_model_catalog.py \
  --strict \
  --min-registered 90 \
  --min-ready 50 \
  --min-ready-source DeepUplift=20 \
  --min-ready-source LightGBM=8 \
  --min-ready-source EconML=8 \
  --min-ready-source scikit-uplift=8
"${PYTHON_BIN}" scripts/generate_resume_snapshot.py --check
"${PYTHON_BIN}" scripts/generate_industry_playbook_docs.py
"${PYTHON_BIN}" scripts/generate_industry_source_backlog.py
"${PYTHON_BIN}" scripts/generate_cn_project_onepager.py
"${PYTHON_BIN}" scripts/generate_agent_interview_transcript_sample.py
"${PYTHON_BIN}" scripts/generate_glossary_doc.py
"${PYTHON_BIN}" scripts/generate_model_readiness_trend.py
"${PYTHON_BIN}" scripts/generate_model_deconstruction_docs.py
"${PYTHON_BIN}" scripts/generate_paper_benchmark_datasets.py
"${PYTHON_BIN}" scripts/generate_scenario_model_matrix.py
"${PYTHON_BIN}" scripts/audit_scenario_model_matrix.py --json reports/scenario_model_matrix_audit_latest.json
"${PYTHON_BIN}" scripts/generate_deep_loss_architecture_docs.py
"${PYTHON_BIN}" scripts/generate_evaluation_framework_docs.py
"${PYTHON_BIN}" scripts/generate_uplift_diagnostic_cases.py
"${PYTHON_BIN}" scripts/smoke_uplift_failure_modes.py
"${PYTHON_BIN}" scripts/generate_failure_mode_docs.py
"${PYTHON_BIN}" scripts/probe_optional_uplift_backends.py
"${PYTHON_BIN}" scripts/smoke_cfrnet_differentiable_balance.py
"${PYTHON_BIN}" scripts/smoke_cfrnet_training_balance.py
"${PYTHON_BIN}" scripts/smoke_contrastive_uplift_training.py
"${PYTHON_BIN}" scripts/smoke_deep_neural_ablation.py
"${PYTHON_BIN}" scripts/smoke_deep_model_training_evidence.py
"${PYTHON_BIN}" scripts/run_deep_model_tuned_benchmark.py --preset "${DEEP_TUNING_PRESET}" --rows "${DEEP_TUNING_ROWS}" --known-effect-bootstrap-samples 4
"${PYTHON_BIN}" scripts/run_paper_benchmark_suite.py --preset "${PAPER_BENCH_PRESET}" --rows "${PAPER_BENCH_ROWS}" --known-effect-bootstrap-samples 8
"${PYTHON_BIN}" scripts/generate_paper_benchmark_interpretation.py
"${PYTHON_BIN}" scripts/smoke_deep_loss_architecture_catalog.py
"${PYTHON_BIN}" scripts/smoke_deep_loss_functions.py
"${PYTHON_BIN}" scripts/smoke_utboost_guarded_adapter.py
"${PYTHON_BIN}" scripts/generate_interview_evidence_pack.py
"${PYTHON_BIN}" scripts/build_interview_pack_zip.py

echo "[3/5] Validate interview materials and pack"
"${PYTHON_BIN}" scripts/audit_interview_materials.py --json reports/interview_materials_audit_latest.json
"${PYTHON_BIN}" scripts/audit_industry_playbook.py --json reports/industry_playbook_audit_latest.json
"${PYTHON_BIN}" scripts/audit_industry_sources.py --json reports/industry_source_audit_latest.json
"${PYTHON_BIN}" scripts/smoke_industry_agent_answers.py
"${PYTHON_BIN}" scripts/smoke_demo_preset_source_trace.py
"${PYTHON_BIN}" scripts/smoke_frontier_model_capabilities.py
"${PYTHON_BIN}" scripts/smoke_deep_loss_architecture_catalog.py
"${PYTHON_BIN}" scripts/smoke_deep_loss_functions.py
"${PYTHON_BIN}" scripts/smoke_evaluation_formula_catalog.py
"${PYTHON_BIN}" scripts/smoke_uplift_failure_modes.py
"${PYTHON_BIN}" scripts/smoke_open_source_framework_audit.py
"${PYTHON_BIN}" scripts/generate_frontier_synthetic_datasets.py >/dev/null
"${PYTHON_BIN}" scripts/smoke_frontier_synthetic_datasets.py
"${PYTHON_BIN}" scripts/smoke_frontier_evaluator_metrics.py
"${PYTHON_BIN}" scripts/generate_industrial_scenario_datasets.py >/dev/null
"${PYTHON_BIN}" scripts/generate_paper_benchmark_datasets.py >/dev/null
"${PYTHON_BIN}" scripts/smoke_industrial_scenario_datasets.py
"${PYTHON_BIN}" scripts/smoke_industrial_policy_metrics.py
"${PYTHON_BIN}" scripts/smoke_industrial_scenario_training.py --rows 1200
"${PYTHON_BIN}" scripts/smoke_industrial_scenario_compare.py --all --rows 700
"${PYTHON_BIN}" scripts/smoke_resume_scenario_policy.py
"${PYTHON_BIN}" scripts/smoke_resume_scenario_workbench.py
"${PYTHON_BIN}" scripts/summarize_frontier_training_evidence.py --allow-missing
"${PYTHON_BIN}" scripts/smoke_llm_routing_dataset.py
"${PYTHON_BIN}" scripts/smoke_agent_interview_answers.py
"${PYTHON_BIN}" scripts/smoke_model_deconstruction_agent.py
"${PYTHON_BIN}" scripts/generate_evidence_index.py

echo "[4/5] Capture demo-lite screenshots"
"${PYTHON_BIN}" scripts/smoke_ui_screenshots.py --url "${APP_URL}" --tabs Workflow Scenario Evaluation Story Evidence Agent Models Model-Deconstruction Policy --min-bytes 20000
"${PYTHON_BIN}" scripts/smoke_agent_interview_ui.py --url "${APP_URL}"
"${PYTHON_BIN}" scripts/generate_evidence_index.py
"${PYTHON_BIN}" scripts/build_interview_pack_zip.py
"${PYTHON_BIN}" scripts/validate_interview_pack.py
"${PYTHON_BIN}" scripts/smoke_ui_screenshots.py --url "${APP_URL}" --tabs Story --min-bytes 20000
"${PYTHON_BIN}" scripts/validate_story_freshness.py

echo "[5/5] Demo-lite summary"
"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import json
from pathlib import Path


def latest(pattern: str) -> Path | None:
    matches = sorted(Path(".").glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


audit_path = latest("reports/model_catalog_audit_*.json")
pack_path = Path("reports/deepuplift_agent_interview_pack_latest.zip")
story_path = latest("reports/ui_screenshots_*/story.png")
scenario_matrix_path = Path("reports/scenario_model_matrix_latest.csv")
audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path else {}
print("DeepUplift Agent demo-lite proof")
print("--------------------------------")
print(f"Registered models: {audit.get('registered_models', 'NA')}")
print(f"Runnable models:   {audit.get('ready_models', 'NA')}")
print(f"Ready backends:    {audit.get('ready_source_counts', {})}")
print(f"Scenario matrix:   {scenario_matrix_path} ({scenario_matrix_path.stat().st_size if scenario_matrix_path.exists() else 'missing'} bytes)")
print(f"Story screenshot:  {story_path or 'missing'}")
print(f"Interview ZIP:     {pack_path} ({pack_path.stat().st_size if pack_path.exists() else 'missing'} bytes)")
print("Skipped training:  yes")
PY
