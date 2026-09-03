#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
ROWS="${ROWS:-220}"
SENSITIVITY_SAMPLES="${SENSITIVITY_SAMPLES:-1}"
APP_URL="${APP_URL:-http://localhost:8501}"
SKIP_UI="${SKIP_UI:-0}"
RUN_AGENT_UI_SMOKE="${RUN_AGENT_UI_SMOKE:-0}"
RUN_COMPARE_UI_SMOKE="${RUN_COMPARE_UI_SMOKE:-0}"
PAPER_BENCH_PRESET="${PAPER_BENCH_PRESET:-paper-lite}"
PAPER_BENCH_ROWS="${PAPER_BENCH_ROWS:-220}"
DEEP_TUNING_PRESET="${DEEP_TUNING_PRESET:-tuned-smoke}"
DEEP_TUNING_ROWS="${DEEP_TUNING_ROWS:-180}"
DEEP_ABLATION_RECOMMENDED_SMOKE_LIMIT="${DEEP_ABLATION_RECOMMENDED_SMOKE_LIMIT:-2}"
DEEP_ABLATION_RECOMMENDED_SMOKE_ALL="${DEEP_ABLATION_RECOMMENDED_SMOKE_ALL:-1}"

echo "[1/8] Compile app, core, and scripts"
"${PYTHON_BIN}" -m py_compile \
  app.py \
  deepuplift/core/artifacts.py \
  deepuplift/core/agent_v2.py \
  deepuplift/core/dataset_registry.py \
  deepuplift/core/evaluator.py \
  deepuplift/core/evidence_store.py \
  deepuplift/core/frontier_models.py \
  deepuplift/core/deep_uplift_designs.py \
  deepuplift/core/industrial_scenario_lab.py \
  deepuplift/core/resume_scenario_policy.py \
  deepuplift/core/resume_scenario_workbench.py \
  deepuplift/core/interview_agent.py \
  deepuplift/core/industry_playbook.py \
  deepuplift/core/knowledge.py \
  deepuplift/core/model_cards.py \
  deepuplift/core/model_deconstruction.py \
  deepuplift/core/ope.py \
  deepuplift/core/promotion.py \
  deepuplift/core/readiness.py \
  deepuplift/core/registry.py \
  deepuplift/core/sklearn_models.py \
  deepuplift/core/trainer.py \
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
  scripts/generate_deep_loss_architecture_docs.py \
  scripts/generate_dataset_cards.py \
  scripts/generate_evidence_index.py \
  scripts/generate_evidence_store.py \
  scripts/generate_frontier_synthetic_datasets.py \
  scripts/generate_glossary_doc.py \
  scripts/generate_cn_project_onepager.py \
  scripts/generate_interview_evidence_pack.py \
  scripts/generate_model_readiness_trend.py \
  scripts/generate_model_deconstruction_docs.py \
  scripts/generate_algorithm_claim_ledger.py \
  scripts/generate_paper_reproduction_gap_ledger.py \
  scripts/generate_model_evidence_promotion_matrix.py \
  scripts/generate_model_upgrade_recipe_cards.py \
  scripts/generate_deep_model_telemetry_gap_matrix.py \
  scripts/generate_deep_model_forward_hook_contracts.py \
  scripts/generate_deep_model_hook_training_bridge.py \
  scripts/generate_deep_model_telemetry_benchmark_linkage.py \
  scripts/generate_deep_model_telemetry_ablation_gate.py \
  scripts/generate_deep_model_ablation_interpretation.py \
  scripts/generate_deep_model_ablation_promotion_matrix.py \
  scripts/generate_deep_model_ablation_next_experiment_plan.py \
  scripts/generate_deep_model_ablation_command_contract.py \
  scripts/generate_deep_model_ablation_command_smoke_coverage.py \
  scripts/generate_deep_model_ablation_command_smoke_coverage_diff.py \
  scripts/smoke_deep_model_ablation_command_contract.py \
  scripts/smoke_deep_model_ablation_recommended_commands.py \
  scripts/generate_deep_model_objective_diagnostics.py \
  scripts/generate_paper_benchmark_datasets.py \
  scripts/generate_paper_benchmark_interpretation.py \
  scripts/generate_regression_warning_audit.py \
  scripts/generate_promotion_launch_cards.py \
  scripts/generate_regression_warning_triage.py \
  scripts/generate_pilot_readiness_scorecard.py \
  scripts/generate_pilot_experiment_plan.py \
  scripts/smoke_pilot_experiment_commands.py \
  scripts/generate_pilot_experiment_command_coverage.py \
  scripts/import_official_acic_dataset.py \
  scripts/generate_resume_snapshot.py \
  scripts/generate_scenario_model_matrix.py \
  scripts/probe_optional_uplift_backends.py \
  scripts/run_benchmark_suite.py \
  scripts/run_deep_model_tuned_benchmark.py \
  scripts/run_paper_benchmark_suite.py \
  scripts/smoke_demo_preset_source_trace.py \
  scripts/smoke_utboost_guarded_adapter.py \
  scripts/smoke_industry_agent_answers.py \
  scripts/smoke_llm_routing_dataset.py \
  scripts/smoke_llm_routing_policy.py \
  scripts/smoke_multi_treatment_benchmark.py \
  scripts/smoke_nuisance_diagnostics.py \
  scripts/smoke_ope_engine.py \
  scripts/smoke_continuous_treatment_policy.py \
  scripts/smoke_agent_interview_answers.py \
  scripts/smoke_model_deconstruction_agent.py \
  scripts/smoke_agent_interview_ui.py \
  scripts/smoke_agent_workflow_ui.py \
  scripts/smoke_budget_policy_optimizer.py \
  scripts/smoke_compare_ui.py \
  scripts/smoke_compare_manifest.py \
  scripts/smoke_cfrnet_differentiable_balance.py \
  scripts/smoke_cfrnet_training_balance.py \
  scripts/smoke_contrastive_uplift_training.py \
  scripts/smoke_deep_neural_ablation.py \
  scripts/smoke_deep_model_forward_hooks.py \
  scripts/smoke_deep_model_trainer_collectors.py \
  scripts/smoke_deep_model_telemetry.py \
  scripts/smoke_deep_model_training_evidence.py \
  scripts/smoke_deep_loss_architecture_catalog.py \
  scripts/smoke_deep_loss_functions.py \
  scripts/smoke_frontier_evaluator_metrics.py \
  scripts/smoke_frontier_model_capabilities.py \
  scripts/smoke_frontier_synthetic_datasets.py \
  scripts/smoke_industrial_scenario_datasets.py \
  scripts/smoke_industrial_scenario_training.py \
  scripts/smoke_industrial_policy_metrics.py \
  scripts/smoke_industrial_scenario_compare.py \
  scripts/smoke_resume_scenario_policy.py \
  scripts/smoke_resume_scenario_workbench.py \
  scripts/smoke_ui_screenshots.py \
  scripts/summarize_frontier_training_evidence.py \
  scripts/test_model_presets.py \
  scripts/validate_interview_pack.py \
  scripts/validate_latest_artifacts.py \
  scripts/validate_story_freshness.py
bash -n scripts/deepuplift_agent.sh
bash -n scripts/demo_lite_check.sh
bash -n scripts/full_regression_check.sh
bash -n scripts/interview_demo_check.sh
bash -n scripts/setup_causalml_py311.sh
bash -n scripts/smoke_test_agent.sh

echo "[2/8] Validate knowledge base JSON"
"${PYTHON_BIN}" -m json.tool deepuplift/knowledge/causal_agent_kb.json >/dev/null
"${PYTHON_BIN}" scripts/generate_industry_playbook_docs.py
"${PYTHON_BIN}" scripts/generate_industry_source_backlog.py
"${PYTHON_BIN}" scripts/generate_cn_project_onepager.py
"${PYTHON_BIN}" scripts/generate_agent_interview_transcript_sample.py
"${PYTHON_BIN}" scripts/generate_glossary_doc.py
"${PYTHON_BIN}" scripts/generate_interview_evidence_pack.py
"${PYTHON_BIN}" scripts/generate_model_readiness_trend.py
"${PYTHON_BIN}" scripts/generate_model_deconstruction_docs.py
"${PYTHON_BIN}" scripts/generate_paper_benchmark_datasets.py
"${PYTHON_BIN}" scripts/generate_dataset_cards.py
"${PYTHON_BIN}" scripts/smoke_ope_engine.py
"${PYTHON_BIN}" scripts/smoke_nuisance_diagnostics.py --rows "${ROWS}"
"${PYTHON_BIN}" scripts/smoke_budget_policy_optimizer.py
"${PYTHON_BIN}" scripts/smoke_multi_treatment_benchmark.py --rows "${ROWS}"
"${PYTHON_BIN}" scripts/smoke_continuous_treatment_policy.py
"${PYTHON_BIN}" scripts/generate_scenario_model_matrix.py
"${PYTHON_BIN}" scripts/audit_scenario_model_matrix.py --json reports/scenario_model_matrix_audit_latest.json
"${PYTHON_BIN}" scripts/generate_deep_loss_architecture_docs.py
"${PYTHON_BIN}" scripts/probe_optional_uplift_backends.py
"${PYTHON_BIN}" scripts/smoke_cfrnet_differentiable_balance.py
"${PYTHON_BIN}" scripts/smoke_cfrnet_training_balance.py
"${PYTHON_BIN}" scripts/smoke_contrastive_uplift_training.py
"${PYTHON_BIN}" scripts/smoke_deep_neural_ablation.py
"${PYTHON_BIN}" scripts/smoke_deep_model_training_evidence.py
"${PYTHON_BIN}" scripts/smoke_deep_model_telemetry.py
"${PYTHON_BIN}" scripts/run_deep_model_tuned_benchmark.py --preset "${DEEP_TUNING_PRESET}" --rows "${DEEP_TUNING_ROWS}" --known-effect-bootstrap-samples 4
"${PYTHON_BIN}" scripts/generate_deep_model_objective_diagnostics.py
"${PYTHON_BIN}" scripts/generate_algorithm_claim_ledger.py
"${PYTHON_BIN}" scripts/generate_paper_reproduction_gap_ledger.py
"${PYTHON_BIN}" scripts/generate_model_evidence_promotion_matrix.py
"${PYTHON_BIN}" scripts/generate_model_upgrade_recipe_cards.py
"${PYTHON_BIN}" scripts/generate_deep_model_telemetry_gap_matrix.py
"${PYTHON_BIN}" scripts/generate_deep_model_forward_hook_contracts.py
"${PYTHON_BIN}" scripts/smoke_deep_model_forward_hooks.py
"${PYTHON_BIN}" scripts/generate_deep_model_hook_training_bridge.py
"${PYTHON_BIN}" scripts/smoke_deep_model_trainer_collectors.py
"${PYTHON_BIN}" scripts/run_paper_benchmark_suite.py --preset "${PAPER_BENCH_PRESET}" --rows "${PAPER_BENCH_ROWS}" --known-effect-bootstrap-samples 8
"${PYTHON_BIN}" scripts/generate_paper_benchmark_interpretation.py
"${PYTHON_BIN}" scripts/generate_deep_model_telemetry_benchmark_linkage.py
"${PYTHON_BIN}" scripts/generate_deep_model_telemetry_ablation_gate.py
"${PYTHON_BIN}" scripts/generate_deep_model_ablation_interpretation.py
"${PYTHON_BIN}" scripts/generate_deep_model_ablation_promotion_matrix.py
"${PYTHON_BIN}" scripts/generate_deep_model_ablation_next_experiment_plan.py
"${PYTHON_BIN}" scripts/generate_deep_model_ablation_command_contract.py
"${PYTHON_BIN}" scripts/generate_algorithm_claim_ledger.py
"${PYTHON_BIN}" scripts/generate_paper_reproduction_gap_ledger.py
"${PYTHON_BIN}" scripts/generate_model_evidence_promotion_matrix.py
"${PYTHON_BIN}" scripts/generate_model_upgrade_recipe_cards.py
"${PYTHON_BIN}" scripts/generate_deep_model_telemetry_gap_matrix.py
"${PYTHON_BIN}" scripts/generate_deep_model_forward_hook_contracts.py
"${PYTHON_BIN}" scripts/smoke_deep_model_forward_hooks.py
"${PYTHON_BIN}" scripts/generate_deep_model_hook_training_bridge.py
"${PYTHON_BIN}" scripts/smoke_deep_model_trainer_collectors.py
"${PYTHON_BIN}" scripts/generate_deep_model_telemetry_benchmark_linkage.py
"${PYTHON_BIN}" scripts/generate_deep_model_telemetry_ablation_gate.py
"${PYTHON_BIN}" scripts/generate_deep_model_ablation_interpretation.py
"${PYTHON_BIN}" scripts/generate_deep_model_ablation_promotion_matrix.py
"${PYTHON_BIN}" scripts/generate_deep_model_ablation_next_experiment_plan.py
"${PYTHON_BIN}" scripts/generate_deep_model_ablation_command_contract.py
"${PYTHON_BIN}" scripts/smoke_deep_model_ablation_command_contract.py --limit 1
if [[ "${DEEP_ABLATION_RECOMMENDED_SMOKE_ALL}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/smoke_deep_model_ablation_recommended_commands.py --all
else
  "${PYTHON_BIN}" scripts/smoke_deep_model_ablation_recommended_commands.py --limit "${DEEP_ABLATION_RECOMMENDED_SMOKE_LIMIT}"
fi
"${PYTHON_BIN}" scripts/generate_deep_model_ablation_command_smoke_coverage.py
"${PYTHON_BIN}" scripts/generate_deep_model_ablation_command_smoke_coverage_diff.py
"${PYTHON_BIN}" scripts/smoke_deep_loss_architecture_catalog.py
"${PYTHON_BIN}" scripts/smoke_deep_loss_functions.py
"${PYTHON_BIN}" scripts/smoke_utboost_guarded_adapter.py
"${PYTHON_BIN}" scripts/generate_interview_evidence_pack.py
"${PYTHON_BIN}" scripts/build_interview_pack_zip.py
"${PYTHON_BIN}" scripts/audit_interview_materials.py --json reports/interview_materials_audit_latest.json
"${PYTHON_BIN}" scripts/audit_industry_playbook.py --json reports/industry_playbook_audit_latest.json
"${PYTHON_BIN}" scripts/audit_industry_sources.py --json reports/industry_source_audit_latest.json
"${PYTHON_BIN}" scripts/smoke_industry_agent_answers.py
"${PYTHON_BIN}" scripts/smoke_demo_preset_source_trace.py
"${PYTHON_BIN}" scripts/smoke_frontier_model_capabilities.py
"${PYTHON_BIN}" scripts/smoke_deep_loss_architecture_catalog.py
"${PYTHON_BIN}" scripts/smoke_deep_loss_functions.py
"${PYTHON_BIN}" scripts/generate_frontier_synthetic_datasets.py >/dev/null
"${PYTHON_BIN}" scripts/smoke_frontier_synthetic_datasets.py
"${PYTHON_BIN}" scripts/smoke_frontier_evaluator_metrics.py
"${PYTHON_BIN}" scripts/generate_industrial_scenario_datasets.py >/dev/null
"${PYTHON_BIN}" scripts/generate_paper_benchmark_datasets.py >/dev/null
"${PYTHON_BIN}" scripts/smoke_industrial_scenario_datasets.py
"${PYTHON_BIN}" scripts/smoke_industrial_policy_metrics.py
"${PYTHON_BIN}" scripts/smoke_industrial_scenario_training.py --rows "${ROWS}"
"${PYTHON_BIN}" scripts/smoke_industrial_scenario_compare.py --rows "${ROWS}" --all
"${PYTHON_BIN}" scripts/smoke_resume_scenario_policy.py
"${PYTHON_BIN}" scripts/smoke_resume_scenario_workbench.py
"${PYTHON_BIN}" scripts/summarize_frontier_training_evidence.py --allow-missing
"${PYTHON_BIN}" scripts/smoke_llm_routing_dataset.py
"${PYTHON_BIN}" scripts/smoke_llm_routing_policy.py
"${PYTHON_BIN}" scripts/smoke_agent_interview_answers.py
"${PYTHON_BIN}" scripts/smoke_model_deconstruction_agent.py
"${PYTHON_BIN}" scripts/generate_regression_warning_audit.py
"${PYTHON_BIN}" scripts/generate_promotion_launch_cards.py
"${PYTHON_BIN}" scripts/generate_regression_warning_triage.py
"${PYTHON_BIN}" scripts/generate_pilot_readiness_scorecard.py
"${PYTHON_BIN}" scripts/generate_pilot_experiment_plan.py
"${PYTHON_BIN}" scripts/smoke_pilot_experiment_commands.py --limit 3
"${PYTHON_BIN}" scripts/generate_pilot_experiment_command_coverage.py
"${PYTHON_BIN}" scripts/generate_interview_evidence_pack.py
"${PYTHON_BIN}" scripts/build_interview_pack_zip.py

echo "[3/8] Audit model catalog, model cards, and presets"
"${PYTHON_BIN}" scripts/audit_model_catalog.py \
  --strict \
  --min-registered 90 \
  --min-ready 35 \
  --min-ready-source DeepUplift=20
"${PYTHON_BIN}" scripts/test_model_presets.py
"${PYTHON_BIN}" scripts/generate_model_readiness_trend.py

echo "[4/8] Run lightweight training smoke"
PYTHON_BIN="${PYTHON_BIN}" ROWS="${ROWS}" SENSITIVITY_SAMPLES="${SENSITIVITY_SAMPLES}" scripts/smoke_test_agent.sh

echo "[5/8] Run no-UI compare smoke"
"${PYTHON_BIN}" scripts/smoke_compare_manifest.py \
  --rows "${ROWS}" \
  --models TLearnerGBM DRLearnerGBM \
  --sensitivity-samples "${SENSITIVITY_SAMPLES}"

echo "[6/8] Check running UI and capture screenshots"
REQUIRE_UI=0
if [[ "${SKIP_UI}" == "1" ]]; then
  echo "UI screenshot smoke skipped because SKIP_UI=1"
elif curl -fsS --max-time 5 "${APP_URL}" >/dev/null; then
  REQUIRE_UI=1
  "${PYTHON_BIN}" scripts/smoke_ui_screenshots.py --url "${APP_URL}" --tabs Workflow Scenario Data Models Model-Deconstruction Evaluation Evidence Agent Knowledge Story
  "${PYTHON_BIN}" scripts/smoke_agent_interview_ui.py --url "${APP_URL}"
  if [[ "${RUN_AGENT_UI_SMOKE}" == "1" ]]; then
    echo "Running Agent workflow browser smoke"
    "${PYTHON_BIN}" scripts/smoke_agent_workflow_ui.py --url "${APP_URL}" --rows "${ROWS}" --reports-dir reports
  fi
  if [[ "${RUN_COMPARE_UI_SMOKE}" == "1" ]]; then
    echo "Running Compare browser smoke"
    "${PYTHON_BIN}" scripts/smoke_compare_ui.py --url "${APP_URL}" --rows "${ROWS}" --reports-dir reports
  fi
else
  echo "UI at ${APP_URL} is not reachable; skipping screenshots."
fi

echo "[7/8] Validate latest artifacts"
if [[ "${REQUIRE_UI}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/validate_latest_artifacts.py --require-ui
else
  "${PYTHON_BIN}" scripts/validate_latest_artifacts.py
fi

echo "[8/8] Refresh resume snapshot"
"${PYTHON_BIN}" scripts/generate_resume_snapshot.py --check
"${PYTHON_BIN}" scripts/generate_industry_playbook_docs.py
"${PYTHON_BIN}" scripts/generate_industry_source_backlog.py
"${PYTHON_BIN}" scripts/generate_cn_project_onepager.py
"${PYTHON_BIN}" scripts/generate_agent_interview_transcript_sample.py
"${PYTHON_BIN}" scripts/generate_glossary_doc.py
"${PYTHON_BIN}" scripts/generate_interview_evidence_pack.py
"${PYTHON_BIN}" scripts/generate_model_readiness_trend.py
"${PYTHON_BIN}" scripts/generate_model_deconstruction_docs.py
"${PYTHON_BIN}" scripts/generate_deep_model_objective_diagnostics.py
"${PYTHON_BIN}" scripts/smoke_deep_model_telemetry.py
"${PYTHON_BIN}" scripts/generate_algorithm_claim_ledger.py
"${PYTHON_BIN}" scripts/generate_paper_reproduction_gap_ledger.py
"${PYTHON_BIN}" scripts/generate_model_evidence_promotion_matrix.py
"${PYTHON_BIN}" scripts/generate_model_upgrade_recipe_cards.py
"${PYTHON_BIN}" scripts/generate_deep_model_telemetry_gap_matrix.py
"${PYTHON_BIN}" scripts/generate_deep_model_forward_hook_contracts.py
"${PYTHON_BIN}" scripts/smoke_deep_model_forward_hooks.py
"${PYTHON_BIN}" scripts/generate_deep_model_hook_training_bridge.py
"${PYTHON_BIN}" scripts/smoke_deep_model_trainer_collectors.py
"${PYTHON_BIN}" scripts/generate_scenario_model_matrix.py
"${PYTHON_BIN}" scripts/generate_paper_benchmark_datasets.py
"${PYTHON_BIN}" scripts/generate_paper_benchmark_interpretation.py
"${PYTHON_BIN}" scripts/generate_deep_model_telemetry_benchmark_linkage.py
"${PYTHON_BIN}" scripts/generate_deep_model_telemetry_ablation_gate.py
"${PYTHON_BIN}" scripts/generate_deep_model_ablation_interpretation.py
"${PYTHON_BIN}" scripts/generate_deep_model_ablation_promotion_matrix.py
"${PYTHON_BIN}" scripts/generate_deep_model_ablation_next_experiment_plan.py
"${PYTHON_BIN}" scripts/generate_deep_model_ablation_command_contract.py
"${PYTHON_BIN}" scripts/generate_deep_model_ablation_command_smoke_coverage.py
"${PYTHON_BIN}" scripts/generate_deep_model_ablation_command_smoke_coverage_diff.py
"${PYTHON_BIN}" scripts/generate_dataset_cards.py
"${PYTHON_BIN}" scripts/generate_interview_evidence_pack.py
"${PYTHON_BIN}" scripts/audit_scenario_model_matrix.py --json reports/scenario_model_matrix_audit_latest.json
"${PYTHON_BIN}" scripts/build_interview_pack_zip.py
"${PYTHON_BIN}" scripts/validate_interview_pack.py
"${PYTHON_BIN}" scripts/generate_evidence_store.py --limit 200
"${PYTHON_BIN}" scripts/generate_regression_warning_audit.py
"${PYTHON_BIN}" scripts/generate_promotion_launch_cards.py
"${PYTHON_BIN}" scripts/generate_regression_warning_triage.py
"${PYTHON_BIN}" scripts/generate_pilot_readiness_scorecard.py
"${PYTHON_BIN}" scripts/generate_pilot_experiment_plan.py
"${PYTHON_BIN}" scripts/smoke_pilot_experiment_commands.py --limit 3
"${PYTHON_BIN}" scripts/generate_pilot_experiment_command_coverage.py
"${PYTHON_BIN}" scripts/generate_algorithm_claim_ledger.py
"${PYTHON_BIN}" scripts/generate_paper_reproduction_gap_ledger.py
"${PYTHON_BIN}" scripts/generate_model_evidence_promotion_matrix.py
"${PYTHON_BIN}" scripts/generate_model_upgrade_recipe_cards.py
"${PYTHON_BIN}" scripts/generate_deep_model_telemetry_gap_matrix.py
"${PYTHON_BIN}" scripts/generate_deep_model_forward_hook_contracts.py
"${PYTHON_BIN}" scripts/smoke_deep_model_forward_hooks.py
"${PYTHON_BIN}" scripts/generate_deep_model_hook_training_bridge.py
"${PYTHON_BIN}" scripts/smoke_deep_model_trainer_collectors.py
"${PYTHON_BIN}" scripts/generate_deep_model_telemetry_benchmark_linkage.py
"${PYTHON_BIN}" scripts/generate_deep_model_telemetry_ablation_gate.py
"${PYTHON_BIN}" scripts/generate_deep_model_ablation_interpretation.py
"${PYTHON_BIN}" scripts/generate_deep_model_ablation_promotion_matrix.py
"${PYTHON_BIN}" scripts/generate_deep_model_ablation_next_experiment_plan.py
"${PYTHON_BIN}" scripts/generate_deep_model_ablation_command_contract.py
"${PYTHON_BIN}" scripts/generate_deep_model_ablation_command_smoke_coverage.py
"${PYTHON_BIN}" scripts/generate_deep_model_ablation_command_smoke_coverage_diff.py
"${PYTHON_BIN}" scripts/generate_interview_evidence_pack.py
"${PYTHON_BIN}" scripts/build_interview_pack_zip.py
"${PYTHON_BIN}" scripts/validate_interview_pack.py
"${PYTHON_BIN}" scripts/generate_evidence_index.py
if [[ "${SKIP_UI}" == "1" ]]; then
  echo "Final Story freshness screenshot skipped because SKIP_UI=1"
elif curl -fsS --max-time 5 "${APP_URL}" >/dev/null; then
  "${PYTHON_BIN}" scripts/smoke_ui_screenshots.py --url "${APP_URL}" --tabs Workflow Story --min-bytes 20000
  "${PYTHON_BIN}" scripts/validate_story_freshness.py
else
  echo "UI at ${APP_URL} is not reachable; skipping final Story freshness check."
fi

echo "Full regression check passed."
