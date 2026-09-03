from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def latest(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def read_json(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def latest_run_evidence(runs_dir: Path) -> Path | None:
    run_dirs = [
        path
        for path in runs_dir.glob("*")
        if path.is_dir()
        and (path / "metrics.json").exists()
        and (path / "config.json").exists()
        and (path / "evidence_manifest.json").exists()
    ]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def file_row(label: str, path: Path | None) -> str:
    if path is None:
        return f"| {label} | missing |  |  |"
    if path.is_dir():
        size = sum(item.stat().st_size for item in path.glob("*") if item.is_file())
        return f"| {label} | `{path}` | directory | {size} |"
    if path.is_file():
        return f"| {label} | `{path}` | {path.suffix.lstrip('.') or 'file'} | {path.stat().st_size} |"
    return f"| {label} | `{path}` | missing |  |"


def build_index(reports_dir: Path, runs_dir: Path) -> str:
    audit_path = latest(reports_dir, "model_catalog_audit_*.json")
    regression_path = latest(reports_dir, "agent_regression_*.json")
    agent_v2_regression_path = latest(reports_dir, "agent_v2_regression_latest.json")
    model_deconstruction_catalog_path = latest(reports_dir, "model_deconstruction_catalog_latest.json")
    model_deconstruction_agent_path = latest(reports_dir, "model_deconstruction_agent_latest.json")
    compare_path = latest(reports_dir, "no_ui_compare_manifest_*.json")
    compare_bundle_path = latest(reports_dir, "*compare_bundle_*.json")
    screenshots_dir = latest(reports_dir, "ui_screenshots_*")
    source_csv_path = latest(reports_dir, "model_source_readiness_*.csv")
    family_csv_path = latest(reports_dir, "model_family_readiness_*.csv")
    interview_audit_path = latest(reports_dir, "interview_materials_audit*.json")
    industry_audit_path = latest(reports_dir, "industry_playbook_audit*.json")
    industry_source_audit_path = latest(reports_dir, "industry_source_audit*.json")
    industry_source_backlog_path = latest(reports_dir, "industry_source_backlog*.json")
    industry_json_path = latest(reports_dir, "industry_playbook*.json")
    optional_backend_probe_path = latest(reports_dir, "optional_backend_probe_latest.json")
    utboost_guarded_smoke_path = latest(reports_dir, "utboost_guarded_adapter_smoke_latest.json")
    open_source_framework_audit_path = latest(reports_dir, "open_source_framework_audit_latest.json")
    open_source_framework_smoke_path = latest(reports_dir, "open_source_framework_audit_smoke_latest.json")
    evaluation_formula_smoke_path = latest(reports_dir, "evaluation_formula_catalog_smoke_latest.json")
    interview_pack_json_path = latest(reports_dir, "interview_evidence_pack*.json")
    agent_transcript_json_path = latest(reports_dir, "agent_interview_transcript_sample*.json")
    interview_zip_path = latest(reports_dir, "deepuplift_agent_interview_pack*.zip")
    interview_zip_manifest_path = latest(reports_dir, "deepuplift_agent_interview_pack_manifest*.json")
    readiness_trend_csv_path = latest(reports_dir, "model_readiness_trend*.csv")
    readiness_trend_json_path = latest(reports_dir, "model_readiness_trend*.json")
    scenario_matrix_csv_path = reports_dir / "scenario_model_matrix_latest.csv"
    if not scenario_matrix_csv_path.exists():
        scenario_matrix_csv_path = latest(reports_dir, "scenario_model_matrix*.csv")
    scenario_matrix_json_path = reports_dir / "scenario_model_matrix_latest.json"
    if not scenario_matrix_json_path.exists():
        scenario_matrix_json_path = latest(reports_dir, "scenario_model_matrix*.json")
    scenario_matrix_audit_path = latest(reports_dir, "scenario_model_matrix_audit*.json")
    github_framework_audit_path = latest(reports_dir, "github_framework_audit_latest.json")
    github_framework_audit_csv_path = latest(reports_dir, "github_framework_audit_latest.csv")
    frontier_dataset_smoke_path = latest(reports_dir, "frontier_synthetic_datasets_smoke_latest.json")
    frontier_evaluator_smoke_path = latest(reports_dir, "frontier_evaluator_metrics_smoke_latest.json")
    industrial_dataset_smoke_path = latest(reports_dir, "industrial_scenario_datasets_smoke_latest.json")
    industrial_training_smoke_path = latest(reports_dir, "industrial_scenario_training_smoke_latest.json")
    industrial_policy_smoke_path = latest(reports_dir, "industrial_policy_metrics_smoke_latest.json")
    resume_workbench_smoke_path = latest(reports_dir, "resume_scenario_workbench_smoke_latest.json")
    industrial_compare_smoke_path = latest(reports_dir, "industrial_scenario_compare_smoke_latest.json")
    industrial_compare_csv_path = latest(reports_dir, "industrial_scenario_compare_latest.csv")
    demo_preset_source_trace_path = latest(reports_dir, "demo_preset_source_trace_latest.json")
    frontier_training_evidence_path = latest(reports_dir, "frontier_training_evidence_latest.json")
    deep_loss_architecture_smoke_path = latest(reports_dir, "deep_loss_architecture_catalog_latest.json")
    deep_loss_functions_smoke_path = latest(reports_dir, "deep_loss_functions_smoke_latest.json")
    cfrnet_balance_smoke_path = latest(reports_dir, "cfrnet_differentiable_balance_smoke_latest.json")
    cfrnet_training_smoke_path = latest(reports_dir, "cfrnet_training_balance_smoke_latest.json")
    contrastive_training_smoke_path = latest(reports_dir, "contrastive_uplift_training_smoke_latest.json")
    deep_neural_ablation_path = latest(reports_dir, "deep_neural_ablation_latest.json")
    deep_model_training_evidence_path = latest(reports_dir, "deep_model_training_evidence_latest.json")
    deep_model_training_leaderboard_path = latest(reports_dir, "deep_model_training_leaderboard_latest.csv")
    deep_model_tuned_benchmark_path = latest(reports_dir, "deep_model_tuned_benchmark_latest.json")
    deep_model_tuned_leaderboard_path = latest(reports_dir, "deep_model_tuned_benchmark_leaderboard_latest.csv")
    deep_model_tuned_battle_path = latest(reports_dir, "deep_model_tuned_benchmark_battle_cards_latest.csv")
    deep_model_tuned_run_diff_path = latest(reports_dir, "deep_model_tuned_benchmark_run_diff_latest.json")
    deep_model_tuned_manifest_path = latest(reports_dir, "deep_model_tuned_benchmark_manifest_latest.json")
    deep_model_objective_diagnostics_path = latest(reports_dir, "deep_model_objective_diagnostics_latest.json")
    deep_model_objective_diagnostics_csv_path = latest(reports_dir, "deep_model_objective_diagnostics_latest.csv")
    algorithm_claim_ledger_path = latest(reports_dir, "algorithm_claim_ledger_latest.json")
    algorithm_claim_ledger_csv_path = latest(reports_dir, "algorithm_claim_ledger_latest.csv")
    paper_reproduction_gap_ledger_path = latest(reports_dir, "paper_reproduction_gap_ledger_latest.json")
    paper_reproduction_gap_ledger_csv_path = latest(reports_dir, "paper_reproduction_gap_ledger_latest.csv")
    model_evidence_promotion_matrix_path = latest(reports_dir, "model_evidence_promotion_matrix_latest.json")
    model_evidence_promotion_matrix_csv_path = latest(reports_dir, "model_evidence_promotion_matrix_latest.csv")
    model_upgrade_recipe_cards_path = latest(reports_dir, "model_upgrade_recipe_cards_latest.json")
    model_upgrade_recipe_cards_csv_path = latest(reports_dir, "model_upgrade_recipe_cards_latest.csv")
    deep_model_telemetry_gap_matrix_path = latest(reports_dir, "deep_model_telemetry_gap_matrix_latest.json")
    deep_model_telemetry_gap_matrix_csv_path = latest(reports_dir, "deep_model_telemetry_gap_matrix_latest.csv")
    deep_model_telemetry_smoke_path = latest(reports_dir, "deep_model_telemetry_smoke_latest.json")
    deep_model_telemetry_smoke_csv_path = latest(reports_dir, "deep_model_telemetry_smoke_latest.csv")
    deep_model_forward_hook_contracts_path = latest(reports_dir, "deep_model_forward_hook_contracts_latest.json")
    deep_model_forward_hook_contracts_csv_path = latest(reports_dir, "deep_model_forward_hook_contracts_latest.csv")
    deep_model_forward_hook_smoke_path = latest(reports_dir, "deep_model_forward_hook_smoke_latest.json")
    deep_model_forward_hook_smoke_csv_path = latest(reports_dir, "deep_model_forward_hook_smoke_latest.csv")
    deep_model_hook_training_bridge_path = latest(reports_dir, "deep_model_hook_training_bridge_latest.json")
    deep_model_hook_training_bridge_csv_path = latest(reports_dir, "deep_model_hook_training_bridge_latest.csv")
    deep_model_trainer_collector_smoke_path = latest(reports_dir, "deep_model_trainer_collector_smoke_latest.json")
    deep_model_trainer_collector_models_csv_path = latest(reports_dir, "deep_model_trainer_collector_models_latest.csv")
    deep_model_trainer_collector_epochs_csv_path = latest(reports_dir, "deep_model_trainer_collector_epochs_latest.csv")
    deep_model_telemetry_benchmark_linkage_path = latest(reports_dir, "deep_model_telemetry_benchmark_linkage_latest.json")
    deep_model_telemetry_benchmark_linkage_csv_path = latest(reports_dir, "deep_model_telemetry_benchmark_linkage_latest.csv")
    deep_model_telemetry_ablation_gate_path = latest(reports_dir, "deep_model_telemetry_ablation_gate_latest.json")
    deep_model_telemetry_ablation_gate_csv_path = latest(reports_dir, "deep_model_telemetry_ablation_gate_latest.csv")
    deep_model_ablation_interpretation_path = latest(reports_dir, "deep_model_ablation_interpretation_latest.json")
    deep_model_ablation_interpretation_csv_path = latest(reports_dir, "deep_model_ablation_interpretation_latest.csv")
    deep_model_ablation_promotion_matrix_path = latest(reports_dir, "deep_model_ablation_promotion_matrix_latest.json")
    deep_model_ablation_promotion_matrix_csv_path = latest(reports_dir, "deep_model_ablation_promotion_matrix_latest.csv")
    deep_model_ablation_next_experiment_plan_path = latest(reports_dir, "deep_model_ablation_next_experiment_plan_latest.json")
    deep_model_ablation_next_experiment_plan_csv_path = latest(reports_dir, "deep_model_ablation_next_experiment_plan_latest.csv")
    deep_model_ablation_command_contract_path = latest(reports_dir, "deep_model_ablation_command_contract_latest.json")
    deep_model_ablation_command_contract_csv_path = latest(reports_dir, "deep_model_ablation_command_contract_latest.csv")
    deep_model_ablation_command_contract_smoke_path = latest(reports_dir, "deep_model_ablation_command_contract_smoke_latest.json")
    deep_model_ablation_command_contract_smoke_csv_path = latest(reports_dir, "deep_model_ablation_command_contract_smoke_latest.csv")
    deep_model_ablation_recommended_command_smoke_path = latest(reports_dir, "deep_model_ablation_recommended_command_smoke_latest.json")
    deep_model_ablation_recommended_command_smoke_csv_path = latest(reports_dir, "deep_model_ablation_recommended_command_smoke_latest.csv")
    deep_model_ablation_command_smoke_coverage_path = latest(reports_dir, "deep_model_ablation_command_smoke_coverage_latest.json")
    deep_model_ablation_command_smoke_coverage_csv_path = latest(reports_dir, "deep_model_ablation_command_smoke_coverage_latest.csv")
    deep_model_ablation_command_smoke_coverage_diff_path = latest(reports_dir, "deep_model_ablation_command_smoke_coverage_diff_latest.json")
    deep_model_ablation_command_smoke_coverage_diff_csv_path = latest(reports_dir, "deep_model_ablation_command_smoke_coverage_diff_latest.csv")
    paper_benchmark_path = latest(reports_dir, "paper_benchmark_latest.json")
    paper_benchmark_leaderboard_path = latest(reports_dir, "paper_benchmark_leaderboard_latest.csv")
    paper_benchmark_run_diff_path = latest(reports_dir, "paper_benchmark_run_diff_latest.json")
    paper_benchmark_manifest_path = latest(reports_dir, "paper_benchmark_manifest_latest.json")
    paper_benchmark_interpretation_path = latest(reports_dir, "paper_benchmark_interpretation_latest.json")
    glossary_json_path = latest(reports_dir, "glossary_latest.json")
    uplift_diagnostic_cases_path = latest(reports_dir, "uplift_diagnostic_cases_latest.json")
    uplift_diagnostic_cases_csv_path = latest(reports_dir, "uplift_diagnostic_cases_latest.csv")
    uplift_failure_modes_smoke_path = latest(reports_dir, "uplift_failure_modes_smoke_latest.json")
    llm_routing_ope_smoke_path = latest(reports_dir, "llm_routing_ope_smoke_latest.json")
    ope_engine_smoke_path = latest(reports_dir, "ope_engine_smoke_latest.json")
    nuisance_diagnostics_smoke_path = latest(reports_dir, "nuisance_diagnostics_smoke_latest.json")
    budget_policy_optimizer_smoke_path = latest(reports_dir, "budget_policy_optimizer_smoke_latest.json")
    multi_treatment_benchmark_path = latest(reports_dir, "multi_treatment_benchmark_latest.json")
    multi_treatment_benchmark_csv_path = latest(reports_dir, "multi_treatment_benchmark_latest.csv")
    llm_routing_bandit_drift_path = latest(reports_dir, "llm_routing_bandit_drift_latest.csv")
    continuous_treatment_policy_smoke_path = latest(reports_dir, "continuous_treatment_policy_smoke_latest.json")
    continuous_treatment_policy_csv_path = latest(reports_dir, "continuous_treatment_policy_rows_latest.csv")
    benchmark_suite_path = latest(reports_dir, "benchmark_suite_latest.json")
    benchmark_leaderboard_path = latest(reports_dir, "benchmark_leaderboard_latest.csv")
    benchmark_run_diff_path = latest(reports_dir, "benchmark_run_diff_latest.json")
    dataset_registry_json_path = latest(reports_dir, "dataset_registry_latest.json")
    dataset_registry_csv_path = latest(reports_dir, "dataset_registry_latest.csv")
    evidence_store_json_path = latest(reports_dir, "evidence_store_latest.json")
    evidence_store_csv_path = latest(reports_dir, "evidence_store_latest.csv")
    evidence_run_diff_path = latest(reports_dir, "evidence_run_diff_latest.json")
    regression_warning_audit_json_path = latest(reports_dir, "regression_warning_audit_latest.json")
    regression_warning_audit_csv_path = latest(reports_dir, "regression_warning_audit_latest.csv")
    promotion_launch_cards_json_path = latest(reports_dir, "promotion_launch_cards_latest.json")
    promotion_launch_cards_csv_path = latest(reports_dir, "promotion_launch_cards_latest.csv")
    regression_warning_triage_json_path = latest(reports_dir, "regression_warning_triage_latest.json")
    regression_warning_triage_csv_path = latest(reports_dir, "regression_warning_triage_latest.csv")
    pilot_readiness_scorecard_json_path = latest(reports_dir, "pilot_readiness_scorecard_latest.json")
    pilot_readiness_scorecard_csv_path = latest(reports_dir, "pilot_readiness_scorecard_latest.csv")
    pilot_experiment_plan_json_path = latest(reports_dir, "pilot_experiment_plan_latest.json")
    pilot_experiment_plan_csv_path = latest(reports_dir, "pilot_experiment_plan_latest.csv")
    pilot_experiment_command_smoke_json_path = latest(reports_dir, "pilot_experiment_command_smoke_latest.json")
    pilot_experiment_command_smoke_csv_path = latest(reports_dir, "pilot_experiment_command_smoke_latest.csv")
    pilot_experiment_command_coverage_json_path = latest(reports_dir, "pilot_experiment_command_coverage_latest.json")
    pilot_experiment_command_coverage_csv_path = latest(reports_dir, "pilot_experiment_command_coverage_latest.csv")
    resume_snapshot_path = Path("docs/RESUME_DEEPUplift_AGENT_SNAPSHOT.md")
    cn_onepager_path = Path("docs/DEEPUplift_AGENT_ONEPAGER_CN.md")
    interview_difficulties_path = Path("docs/INTERVIEW_DIFFICULTIES_AND_SOLUTIONS.md")
    interview_deep_dive_path = Path("docs/INTERVIEW_DEEPUplift_AGENT_DEEP_DIVE.md")
    agent_transcript_path = Path("docs/AGENT_INTERVIEW_TRANSCRIPT_SAMPLE.md")
    agent_v2_architecture_path = Path("docs/DEEPUplift_AGENT_2_0_ARCHITECTURE.md")
    model_deconstruction_index_path = Path("docs/DEEPUplift_MODEL_DECONSTRUCTION_INDEX.md")
    interview_pack_path = Path("docs/DEEPUplift_AGENT_INTERVIEW_EVIDENCE_PACK.md")
    interview_qa_path = Path("docs/INTERVIEW_QA_REHEARSAL_CN.md")
    interview_checklist_path = Path("docs/INTERVIEW_DEMO_CHECKLIST_CN.md")
    production_path = Path("docs/DEEPUplift_AGENT_PRODUCTION_ROADMAP.md")
    backend_strategy_path = Path("docs/DEEPUplift_AGENT_MODEL_BACKEND_STRATEGY.md")
    readiness_trend_path = Path("docs/DEEPUplift_AGENT_MODEL_READINESS_TREND.md")
    offline_online_path = Path("docs/DEEPUplift_AGENT_OFFLINE_ONLINE_PLAYBOOK.md")
    deep_loss_architecture_path = Path("docs/UPLIFT_DEEP_LOSS_AND_ARCHITECTURE.md")
    deep_model_interview_path = Path("docs/UPLIFT_DEEP_MODEL_INTERVIEW_DEEP_DIVE.md")
    deep_model_training_evidence_doc_path = Path("docs/UPLIFT_DEEP_MODEL_TRAINING_EVIDENCE.md")
    deep_model_tuned_benchmark_doc_path = Path("docs/DEEPUplift_DEEP_MODEL_TUNING_BENCHMARK.md")
    deep_model_objective_diagnostics_doc_path = Path("docs/DEEPUplift_DEEP_MODEL_OBJECTIVE_DIAGNOSTICS.md")
    algorithm_claim_ledger_doc_path = Path("docs/DEEPUplift_ALGORITHM_CLAIM_LEDGER.md")
    paper_reproduction_gap_ledger_doc_path = Path("docs/DEEPUplift_PAPER_REPRODUCTION_GAP_LEDGER.md")
    model_evidence_promotion_matrix_doc_path = Path("docs/DEEPUplift_MODEL_EVIDENCE_PROMOTION_MATRIX.md")
    model_upgrade_recipe_cards_doc_path = Path("docs/DEEPUplift_MODEL_UPGRADE_RECIPE_CARDS.md")
    deep_model_telemetry_gap_matrix_doc_path = Path("docs/DEEPUplift_DEEP_MODEL_TELEMETRY_GAP_MATRIX.md")
    deep_model_telemetry_smoke_doc_path = Path("docs/DEEPUplift_DEEP_MODEL_TELEMETRY_SMOKE.md")
    deep_model_forward_hook_contracts_doc_path = Path("docs/DEEPUplift_DEEP_MODEL_FORWARD_HOOK_CONTRACTS.md")
    deep_model_forward_hook_smoke_doc_path = Path("docs/DEEPUplift_DEEP_MODEL_FORWARD_HOOK_SMOKE.md")
    deep_model_hook_training_bridge_doc_path = Path("docs/DEEPUplift_DEEP_MODEL_HOOK_TRAINING_BRIDGE.md")
    deep_model_trainer_collector_smoke_doc_path = Path("docs/DEEPUplift_DEEP_MODEL_TRAINER_COLLECTOR_SMOKE.md")
    deep_model_telemetry_benchmark_linkage_doc_path = Path("docs/DEEPUplift_DEEP_MODEL_TELEMETRY_BENCHMARK_LINKAGE.md")
    deep_model_telemetry_ablation_gate_doc_path = Path("docs/DEEPUplift_DEEP_MODEL_TELEMETRY_ABLATION_GATE.md")
    deep_model_ablation_interpretation_doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_INTERPRETATION.md")
    deep_model_ablation_promotion_matrix_doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_PROMOTION_MATRIX.md")
    deep_model_ablation_next_experiment_plan_doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_NEXT_EXPERIMENT_PLAN.md")
    deep_model_ablation_command_contract_doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_CONTRACT.md")
    deep_model_ablation_command_contract_smoke_doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_CONTRACT_SMOKE.md")
    deep_model_ablation_recommended_command_smoke_doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_RECOMMENDED_COMMAND_SMOKE.md")
    deep_model_ablation_command_smoke_coverage_doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_SMOKE_COVERAGE.md")
    deep_model_ablation_command_smoke_coverage_diff_doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_SMOKE_COVERAGE_DIFF.md")
    paper_benchmark_doc_path = Path("docs/DEEPUplift_PAPER_LEVEL_BENCHMARK.md")
    paper_benchmark_interpretation_doc_path = Path("docs/DEEPUplift_BENCHMARK_V3_INTERPRETATION.md")
    glossary_doc_path = Path("docs/DEEPUplift_GLOSSARY.md")
    contrastive_llm_path = Path("docs/UPLIFT_CONTRASTIVE_AND_LLM_UPLIFT.md")
    industrial_scenario_lab_path = Path("docs/UPLIFT_INDUSTRIAL_SCENARIO_LAB.md")
    evaluation_formula_path = Path("docs/UPLIFT_EVALUATION_METRICS_FORMULAS.md")
    policy_derivation_path = Path("docs/UPLIFT_POLICY_VALUE_AND_ROI_DERIVATION.md")
    optimization_path = Path("docs/UPLIFT_MODEL_OPTIMIZATION_PATH.md")
    open_source_framework_path = Path("docs/UPLIFT_OPEN_SOURCE_FRAMEWORK_AUDIT.md")
    industrial_expansion_path = Path("docs/UPLIFT_INDUSTRIAL_CASE_EXPANSION.md")
    model_failure_modes_path = Path("docs/UPLIFT_MODEL_FAILURE_MODES.md")
    scenario_pitfalls_path = Path("docs/UPLIFT_SCENARIO_PITFALLS.md")
    diagnostic_cases_path = Path("docs/UPLIFT_DIAGNOSTIC_CASES.md")
    metric_misleading_cases_path = Path("docs/UPLIFT_METRIC_MISLEADING_CASES.md")
    research_evidence_gate_path = Path("docs/DEEPUplift_RESEARCH_EVIDENCE_GATE.md")
    benchmark_suite_doc_path = Path("docs/DEEPUplift_BENCHMARK_SUITE.md")
    dataset_registry_doc_path = Path("docs/DEEPUplift_DATASET_REGISTRY.md")
    evidence_store_doc_path = Path("docs/DEEPUplift_EXPERIMENT_EVIDENCE_STORE.md")
    regression_warning_audit_doc_path = Path("docs/DEEPUplift_REGRESSION_WARNING_AUDIT.md")
    promotion_launch_cards_doc_path = Path("docs/DEEPUplift_PROMOTION_LAUNCH_CARDS.md")
    regression_warning_triage_doc_path = Path("docs/DEEPUplift_REGRESSION_WARNING_TRIAGE.md")
    pilot_readiness_scorecard_doc_path = Path("docs/DEEPUplift_PILOT_READINESS_SCORECARD.md")
    pilot_experiment_plan_doc_path = Path("docs/DEEPUplift_PILOT_EXPERIMENT_PLAN.md")
    pilot_experiment_command_smoke_doc_path = Path("docs/DEEPUplift_PILOT_EXPERIMENT_COMMAND_SMOKE.md")
    pilot_experiment_command_coverage_doc_path = Path("docs/DEEPUplift_PILOT_EXPERIMENT_COMMAND_COVERAGE.md")
    nuisance_diagnostics_doc_path = Path("docs/DEEPUplift_NUISANCE_DIAGNOSTICS.md")
    long_term_plan_path = Path("docs/DEEPUplift_LONG_TERM_INDUSTRIAL_OPTIMIZATION_PLAN.md")
    industry_doc_paths = [
        Path("docs/INDUSTRIAL_UPLIFT_CASE_STUDIES.md"),
        Path("docs/UPLIFT_BUSINESS_SCENARIOS.md"),
        Path("docs/UPLIFT_INTERVIEW_INDUSTRY_DEEP_DIVE.md"),
        Path("docs/UPLIFT_MODEL_SELECTION_PLAYBOOK.md"),
        Path("docs/UPLIFT_COUPON_ADS_GROWTH_PLAYBOOK.md"),
        Path("docs/UPLIFT_LLM_CAUSAL_FRONTIER.md"),
        Path("docs/LLM_ROUTING_POLICY_PLAYBOOK.md"),
        Path("docs/UPLIFT_LLM_INTERVIEW_QA.md"),
        Path("docs/UPLIFT_ONLINE_EXPERIMENT_AND_INCREMENTALITY.md"),
        Path("docs/UPLIFT_INDUSTRY_REFERENCES.md"),
        Path("docs/INDUSTRY_SOURCE_QUALITY_AUDIT.md"),
        Path("docs/UPLIFT_INDUSTRY_SOURCE_ADOPTION_BACKLOG.md"),
        Path("docs/UPLIFT_SCENARIO_MODEL_MATRIX.md"),
        Path("docs/UPLIFT_SCENARIO_MODEL_MATRIX_INTERVIEW_QA.md"),
        Path("docs/UPLIFT_AGENT_INTERVIEW_RUNBOOK.md"),
    ]
    run_dir = latest_run_evidence(runs_dir)

    audit = read_json(audit_path)
    regression = read_json(regression_path)
    compare = read_json(compare_path)
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S %z")
    screenshot_files = sorted(path.name for path in screenshots_dir.glob("*.png")) if screenshots_dir and screenshots_dir.is_dir() else []
    run_files = sorted(path.name for path in run_dir.glob("*") if path.is_file()) if run_dir and run_dir.is_dir() else []

    rows = "\n".join(
        [
            file_row("Model catalog audit", audit_path),
            file_row("Source readiness CSV", source_csv_path),
            file_row("Family readiness CSV", family_csv_path),
            file_row("Interview materials audit", interview_audit_path),
            file_row("Industry playbook audit", industry_audit_path),
            file_row("Industry source quality audit", industry_source_audit_path),
            file_row("Industry source adoption backlog", industry_source_backlog_path),
            file_row("Industry playbook JSON", industry_json_path),
            file_row("Optional backend probe", optional_backend_probe_path),
            file_row("UTBoost guarded adapter smoke", utboost_guarded_smoke_path),
            file_row("Open source framework audit", open_source_framework_audit_path),
            file_row("Open source framework smoke", open_source_framework_smoke_path),
            file_row("Evaluation formula catalog smoke", evaluation_formula_smoke_path),
            file_row("Interview evidence pack JSON", interview_pack_json_path),
            file_row("Agent interview transcript sample JSON", agent_transcript_json_path),
            file_row("Interview pack ZIP", interview_zip_path),
            file_row("Interview pack ZIP manifest", interview_zip_manifest_path),
            file_row("Model readiness trend CSV", readiness_trend_csv_path),
            file_row("Model readiness trend JSON", readiness_trend_json_path),
            file_row("Scenario model matrix CSV", scenario_matrix_csv_path),
            file_row("Scenario model matrix JSON", scenario_matrix_json_path),
            file_row("Scenario model matrix audit", scenario_matrix_audit_path),
            file_row("GitHub framework audit JSON", github_framework_audit_path),
            file_row("GitHub framework audit CSV", github_framework_audit_csv_path),
            file_row("Frontier synthetic dataset smoke", frontier_dataset_smoke_path),
            file_row("Frontier evaluator metrics smoke", frontier_evaluator_smoke_path),
            file_row("Industrial scenario dataset smoke", industrial_dataset_smoke_path),
            file_row("Industrial scenario training smoke", industrial_training_smoke_path),
            file_row("Industrial policy metrics smoke", industrial_policy_smoke_path),
            file_row("Resume scenario workbench smoke", resume_workbench_smoke_path),
            file_row("Industrial scenario compare smoke", industrial_compare_smoke_path),
            file_row("Industrial scenario compare CSV", industrial_compare_csv_path),
            file_row("Demo preset source trace smoke", demo_preset_source_trace_path),
            file_row("Frontier training evidence", frontier_training_evidence_path),
            file_row("Deep loss architecture catalog smoke", deep_loss_architecture_smoke_path),
            file_row("Deep loss functions gradient smoke", deep_loss_functions_smoke_path),
            file_row("CFRNet differentiable balance smoke", cfrnet_balance_smoke_path),
            file_row("CFRNet training balance smoke", cfrnet_training_smoke_path),
            file_row("Contrastive uplift training smoke", contrastive_training_smoke_path),
            file_row("Deep neural ablation", deep_neural_ablation_path),
            file_row("Deep model training evidence", deep_model_training_evidence_path),
            file_row("Deep model training leaderboard", deep_model_training_leaderboard_path),
            file_row("Deep model tuned benchmark", deep_model_tuned_benchmark_path),
            file_row("Deep model tuned leaderboard", deep_model_tuned_leaderboard_path),
            file_row("Deep model tuned battle cards", deep_model_tuned_battle_path),
            file_row("Deep model tuned run diff", deep_model_tuned_run_diff_path),
            file_row("Deep model tuned manifest", deep_model_tuned_manifest_path),
            file_row("Deep model objective diagnostics", deep_model_objective_diagnostics_path),
            file_row("Deep model objective diagnostics CSV", deep_model_objective_diagnostics_csv_path),
            file_row("Algorithm claim ledger", algorithm_claim_ledger_path),
            file_row("Algorithm claim ledger CSV", algorithm_claim_ledger_csv_path),
            file_row("Paper reproduction gap ledger", paper_reproduction_gap_ledger_path),
            file_row("Paper reproduction gap ledger CSV", paper_reproduction_gap_ledger_csv_path),
            file_row("Model evidence promotion matrix", model_evidence_promotion_matrix_path),
            file_row("Model evidence promotion matrix CSV", model_evidence_promotion_matrix_csv_path),
            file_row("Model upgrade recipe cards", model_upgrade_recipe_cards_path),
            file_row("Model upgrade recipe cards CSV", model_upgrade_recipe_cards_csv_path),
            file_row("Deep model telemetry gap matrix", deep_model_telemetry_gap_matrix_path),
            file_row("Deep model telemetry gap matrix CSV", deep_model_telemetry_gap_matrix_csv_path),
            file_row("Deep model telemetry smoke", deep_model_telemetry_smoke_path),
            file_row("Deep model telemetry smoke CSV", deep_model_telemetry_smoke_csv_path),
            file_row("Deep model forward hook contracts", deep_model_forward_hook_contracts_path),
            file_row("Deep model forward hook contracts CSV", deep_model_forward_hook_contracts_csv_path),
            file_row("Deep model forward hook smoke", deep_model_forward_hook_smoke_path),
            file_row("Deep model forward hook smoke CSV", deep_model_forward_hook_smoke_csv_path),
            file_row("Deep model hook training bridge", deep_model_hook_training_bridge_path),
            file_row("Deep model hook training bridge CSV", deep_model_hook_training_bridge_csv_path),
            file_row("Deep model trainer collector smoke", deep_model_trainer_collector_smoke_path),
            file_row("Deep model trainer collector models CSV", deep_model_trainer_collector_models_csv_path),
            file_row("Deep model trainer collector epochs CSV", deep_model_trainer_collector_epochs_csv_path),
            file_row("Deep model telemetry-benchmark linkage", deep_model_telemetry_benchmark_linkage_path),
            file_row("Deep model telemetry-benchmark linkage CSV", deep_model_telemetry_benchmark_linkage_csv_path),
            file_row("Deep model telemetry ablation gate", deep_model_telemetry_ablation_gate_path),
            file_row("Deep model telemetry ablation gate CSV", deep_model_telemetry_ablation_gate_csv_path),
            file_row("Deep model ablation interpretation", deep_model_ablation_interpretation_path),
            file_row("Deep model ablation interpretation CSV", deep_model_ablation_interpretation_csv_path),
            file_row("Deep model ablation promotion matrix", deep_model_ablation_promotion_matrix_path),
            file_row("Deep model ablation promotion matrix CSV", deep_model_ablation_promotion_matrix_csv_path),
            file_row("Deep model ablation next experiment plan", deep_model_ablation_next_experiment_plan_path),
            file_row("Deep model ablation next experiment plan CSV", deep_model_ablation_next_experiment_plan_csv_path),
            file_row("Deep model ablation command contract", deep_model_ablation_command_contract_path),
            file_row("Deep model ablation command contract CSV", deep_model_ablation_command_contract_csv_path),
            file_row("Deep model ablation command contract smoke", deep_model_ablation_command_contract_smoke_path),
            file_row("Deep model ablation command contract smoke CSV", deep_model_ablation_command_contract_smoke_csv_path),
            file_row("Deep model ablation recommended command smoke", deep_model_ablation_recommended_command_smoke_path),
            file_row("Deep model ablation recommended command smoke CSV", deep_model_ablation_recommended_command_smoke_csv_path),
            file_row("Deep model ablation command smoke coverage", deep_model_ablation_command_smoke_coverage_path),
            file_row("Deep model ablation command smoke coverage CSV", deep_model_ablation_command_smoke_coverage_csv_path),
            file_row("Deep model ablation command smoke coverage diff", deep_model_ablation_command_smoke_coverage_diff_path),
            file_row("Deep model ablation command smoke coverage diff CSV", deep_model_ablation_command_smoke_coverage_diff_csv_path),
            file_row("Paper-level benchmark", paper_benchmark_path),
            file_row("Paper-level benchmark leaderboard", paper_benchmark_leaderboard_path),
            file_row("Paper-level benchmark run diff", paper_benchmark_run_diff_path),
            file_row("Paper-level benchmark manifest", paper_benchmark_manifest_path),
            file_row("Paper-level benchmark interpretation", paper_benchmark_interpretation_path),
            file_row("Glossary JSON", glossary_json_path),
            file_row("Uplift diagnostic cases JSON", uplift_diagnostic_cases_path),
            file_row("Uplift diagnostic cases CSV", uplift_diagnostic_cases_csv_path),
            file_row("Uplift failure modes smoke", uplift_failure_modes_smoke_path),
            file_row("LLM routing OPE smoke", llm_routing_ope_smoke_path),
            file_row("General OPE engine smoke", ope_engine_smoke_path),
            file_row("Cross-fitted nuisance diagnostics smoke", nuisance_diagnostics_smoke_path),
            file_row("Budget policy optimizer smoke", budget_policy_optimizer_smoke_path),
            file_row("Multi-treatment benchmark", multi_treatment_benchmark_path),
            file_row("Multi-treatment benchmark CSV", multi_treatment_benchmark_csv_path),
            file_row("LLM routing bandit drift CSV", llm_routing_bandit_drift_path),
            file_row("Continuous treatment policy smoke", continuous_treatment_policy_smoke_path),
            file_row("Continuous treatment policy CSV", continuous_treatment_policy_csv_path),
            file_row("Benchmark suite latest", benchmark_suite_path),
            file_row("Benchmark leaderboard latest", benchmark_leaderboard_path),
            file_row("Benchmark run diff latest", benchmark_run_diff_path),
            file_row("Dataset registry JSON", dataset_registry_json_path),
            file_row("Dataset registry CSV", dataset_registry_csv_path),
            file_row("Evidence store JSON", evidence_store_json_path),
            file_row("Evidence store CSV", evidence_store_csv_path),
            file_row("Evidence run diff JSON", evidence_run_diff_path),
            file_row("Regression warning audit JSON", regression_warning_audit_json_path),
            file_row("Regression warning audit CSV", regression_warning_audit_csv_path),
            file_row("Promotion launch cards JSON", promotion_launch_cards_json_path),
            file_row("Promotion launch cards CSV", promotion_launch_cards_csv_path),
            file_row("Regression warning triage JSON", regression_warning_triage_json_path),
            file_row("Regression warning triage CSV", regression_warning_triage_csv_path),
            file_row("Pilot readiness scorecard JSON", pilot_readiness_scorecard_json_path),
            file_row("Pilot readiness scorecard CSV", pilot_readiness_scorecard_csv_path),
            file_row("Pilot experiment plan JSON", pilot_experiment_plan_json_path),
            file_row("Pilot experiment plan CSV", pilot_experiment_plan_csv_path),
            file_row("Pilot experiment command smoke JSON", pilot_experiment_command_smoke_json_path),
            file_row("Pilot experiment command smoke CSV", pilot_experiment_command_smoke_csv_path),
            file_row("Pilot experiment command coverage JSON", pilot_experiment_command_coverage_json_path),
            file_row("Pilot experiment command coverage CSV", pilot_experiment_command_coverage_csv_path),
            file_row("Agent regression", regression_path),
            file_row("Agent 2.0 regression", agent_v2_regression_path),
            file_row("Model deconstruction catalog", model_deconstruction_catalog_path),
            file_row("Model deconstruction agent regression", model_deconstruction_agent_path),
            file_row("No-UI compare manifest", compare_path),
            file_row("UI compare bundle", compare_bundle_path),
            file_row("UI screenshot directory", screenshots_dir),
            file_row("Latest run evidence directory", run_dir),
            file_row("Live resume snapshot", resume_snapshot_path if resume_snapshot_path.exists() else None),
            file_row("Chinese project one-pager", cn_onepager_path if cn_onepager_path.exists() else None),
            file_row("Interview difficulties", interview_difficulties_path if interview_difficulties_path.exists() else None),
            file_row("Interview deep dive", interview_deep_dive_path if interview_deep_dive_path.exists() else None),
            file_row("Agent interview transcript sample", agent_transcript_path if agent_transcript_path.exists() else None),
            file_row("Agent 2.0 architecture", agent_v2_architecture_path if agent_v2_architecture_path.exists() else None),
            file_row("Model deconstruction index", model_deconstruction_index_path if model_deconstruction_index_path.exists() else None),
            file_row("Interview evidence pack", interview_pack_path if interview_pack_path.exists() else None),
            file_row("Chinese interview Q&A", interview_qa_path if interview_qa_path.exists() else None),
            file_row("Chinese interview demo checklist", interview_checklist_path if interview_checklist_path.exists() else None),
            file_row("Productionization roadmap", production_path if production_path.exists() else None),
            file_row("Model backend strategy", backend_strategy_path if backend_strategy_path.exists() else None),
            file_row("Model readiness trend", readiness_trend_path if readiness_trend_path.exists() else None),
            file_row("Offline-online playbook", offline_online_path if offline_online_path.exists() else None),
            file_row("Deep loss and architecture", deep_loss_architecture_path if deep_loss_architecture_path.exists() else None),
            file_row("Deep model interview deep dive", deep_model_interview_path if deep_model_interview_path.exists() else None),
            file_row("Deep model training evidence doc", deep_model_training_evidence_doc_path if deep_model_training_evidence_doc_path.exists() else None),
            file_row("Deep model tuned benchmark doc", deep_model_tuned_benchmark_doc_path if deep_model_tuned_benchmark_doc_path.exists() else None),
            file_row(
                "Deep model objective diagnostics doc",
                deep_model_objective_diagnostics_doc_path if deep_model_objective_diagnostics_doc_path.exists() else None,
            ),
            file_row("Algorithm claim ledger doc", algorithm_claim_ledger_doc_path if algorithm_claim_ledger_doc_path.exists() else None),
            file_row(
                "Paper reproduction gap ledger doc",
                paper_reproduction_gap_ledger_doc_path if paper_reproduction_gap_ledger_doc_path.exists() else None,
            ),
            file_row(
                "Model evidence promotion matrix doc",
                model_evidence_promotion_matrix_doc_path if model_evidence_promotion_matrix_doc_path.exists() else None,
            ),
            file_row(
                "Model upgrade recipe cards doc",
                model_upgrade_recipe_cards_doc_path if model_upgrade_recipe_cards_doc_path.exists() else None,
            ),
            file_row(
                "Deep model telemetry gap matrix doc",
                deep_model_telemetry_gap_matrix_doc_path if deep_model_telemetry_gap_matrix_doc_path.exists() else None,
            ),
            file_row(
                "Deep model telemetry smoke doc",
                deep_model_telemetry_smoke_doc_path if deep_model_telemetry_smoke_doc_path.exists() else None,
            ),
            file_row(
                "Deep model forward hook contracts doc",
                deep_model_forward_hook_contracts_doc_path if deep_model_forward_hook_contracts_doc_path.exists() else None,
            ),
            file_row(
                "Deep model forward hook smoke doc",
                deep_model_forward_hook_smoke_doc_path if deep_model_forward_hook_smoke_doc_path.exists() else None,
            ),
            file_row(
                "Deep model hook training bridge doc",
                deep_model_hook_training_bridge_doc_path if deep_model_hook_training_bridge_doc_path.exists() else None,
            ),
            file_row(
                "Deep model trainer collector smoke doc",
                deep_model_trainer_collector_smoke_doc_path if deep_model_trainer_collector_smoke_doc_path.exists() else None,
            ),
            file_row(
                "Deep model telemetry-benchmark linkage doc",
                deep_model_telemetry_benchmark_linkage_doc_path
                if deep_model_telemetry_benchmark_linkage_doc_path.exists()
                else None,
            ),
            file_row(
                "Deep model telemetry ablation gate doc",
                deep_model_telemetry_ablation_gate_doc_path
                if deep_model_telemetry_ablation_gate_doc_path.exists()
                else None,
            ),
            file_row(
                "Deep model ablation interpretation doc",
                deep_model_ablation_interpretation_doc_path
                if deep_model_ablation_interpretation_doc_path.exists()
                else None,
            ),
            file_row(
                "Deep model ablation promotion matrix doc",
                deep_model_ablation_promotion_matrix_doc_path
                if deep_model_ablation_promotion_matrix_doc_path.exists()
                else None,
            ),
            file_row(
                "Deep model ablation next experiment plan doc",
                deep_model_ablation_next_experiment_plan_doc_path
                if deep_model_ablation_next_experiment_plan_doc_path.exists()
                else None,
            ),
            file_row(
                "Deep model ablation command contract doc",
                deep_model_ablation_command_contract_doc_path
                if deep_model_ablation_command_contract_doc_path.exists()
                else None,
            ),
            file_row(
                "Deep model ablation command contract smoke doc",
                deep_model_ablation_command_contract_smoke_doc_path
                if deep_model_ablation_command_contract_smoke_doc_path.exists()
                else None,
            ),
            file_row(
                "Deep model ablation recommended command smoke doc",
                deep_model_ablation_recommended_command_smoke_doc_path
                if deep_model_ablation_recommended_command_smoke_doc_path.exists()
                else None,
            ),
            file_row(
                "Deep model ablation command smoke coverage doc",
                deep_model_ablation_command_smoke_coverage_doc_path
                if deep_model_ablation_command_smoke_coverage_doc_path.exists()
                else None,
            ),
            file_row(
                "Deep model ablation command smoke coverage diff doc",
                deep_model_ablation_command_smoke_coverage_diff_doc_path
                if deep_model_ablation_command_smoke_coverage_diff_doc_path.exists()
                else None,
            ),
            file_row("Paper-level benchmark doc", paper_benchmark_doc_path if paper_benchmark_doc_path.exists() else None),
            file_row(
                "Paper-level benchmark interpretation doc",
                paper_benchmark_interpretation_doc_path if paper_benchmark_interpretation_doc_path.exists() else None,
            ),
            file_row("Glossary doc", glossary_doc_path if glossary_doc_path.exists() else None),
            file_row("Contrastive and LLM uplift", contrastive_llm_path if contrastive_llm_path.exists() else None),
            file_row("Industrial scenario lab", industrial_scenario_lab_path if industrial_scenario_lab_path.exists() else None),
            file_row("Evaluation metrics formulas", evaluation_formula_path if evaluation_formula_path.exists() else None),
            file_row("Policy value ROI derivation", policy_derivation_path if policy_derivation_path.exists() else None),
            file_row("Uplift optimization path", optimization_path if optimization_path.exists() else None),
            file_row("Open source framework audit doc", open_source_framework_path if open_source_framework_path.exists() else None),
            file_row("Industrial case expansion", industrial_expansion_path if industrial_expansion_path.exists() else None),
            file_row("Uplift model failure modes", model_failure_modes_path if model_failure_modes_path.exists() else None),
            file_row("Uplift scenario pitfalls", scenario_pitfalls_path if scenario_pitfalls_path.exists() else None),
            file_row("Uplift diagnostic cases", diagnostic_cases_path if diagnostic_cases_path.exists() else None),
            file_row("Uplift metric misleading cases", metric_misleading_cases_path if metric_misleading_cases_path.exists() else None),
            file_row("Research evidence gate", research_evidence_gate_path if research_evidence_gate_path.exists() else None),
            file_row("Benchmark suite doc", benchmark_suite_doc_path if benchmark_suite_doc_path.exists() else None),
            file_row("Dataset registry doc", dataset_registry_doc_path if dataset_registry_doc_path.exists() else None),
            file_row("Experiment evidence store doc", evidence_store_doc_path if evidence_store_doc_path.exists() else None),
            file_row("Regression warning audit doc", regression_warning_audit_doc_path if regression_warning_audit_doc_path.exists() else None),
            file_row("Promotion launch cards doc", promotion_launch_cards_doc_path if promotion_launch_cards_doc_path.exists() else None),
            file_row("Regression warning triage doc", regression_warning_triage_doc_path if regression_warning_triage_doc_path.exists() else None),
            file_row("Pilot readiness scorecard doc", pilot_readiness_scorecard_doc_path if pilot_readiness_scorecard_doc_path.exists() else None),
            file_row("Pilot experiment plan doc", pilot_experiment_plan_doc_path if pilot_experiment_plan_doc_path.exists() else None),
            file_row("Pilot experiment command smoke doc", pilot_experiment_command_smoke_doc_path if pilot_experiment_command_smoke_doc_path.exists() else None),
            file_row("Pilot experiment command coverage doc", pilot_experiment_command_coverage_doc_path if pilot_experiment_command_coverage_doc_path.exists() else None),
            file_row("Nuisance diagnostics doc", nuisance_diagnostics_doc_path if nuisance_diagnostics_doc_path.exists() else None),
            file_row("Long-term industrial optimization plan", long_term_plan_path if long_term_plan_path.exists() else None),
            *[file_row(f"Industry doc: {path.stem}", path if path.exists() else None) for path in industry_doc_paths],
        ]
    )

    ready_sources = audit.get("ready_source_counts") or {}
    ready_sources_text = ", ".join(f"{source}={count}" for source, count in sorted(ready_sources.items())) or "NA"
    ready_families = audit.get("ready_family_counts") or {}
    ready_families_text = ", ".join(f"{family}={count}" for family, count in sorted(ready_families.items())) or "NA"
    family_rows = "\n".join(
        f"| {family} | {count} |" for family, count in sorted(ready_families.items(), key=lambda item: (-item[1], item[0]))
    )
    source_rows = "\n".join(
        "| {source} | {registered} | {ready} | {guarded} | {needs_dependency} | {needs_python_3_11} |".format(**row)
        for row in audit.get("source_readiness", [])
    )
    regression_rows = regression.get("results") or []
    compare_rows = compare.get("ranking") or []

    return f"""# DeepUplift Agent Evidence Index

Generated at: `{generated_at}`

## Summary

| Metric | Value |
| --- | --- |
| Registered models | {audit.get("registered_models", "NA")} |
| Runnable models | {audit.get("ready_models", "NA")} |
| Ready backends | {ready_sources_text} |
| Ready families | {ready_families_text} |
| Regression runs | {len(regression_rows)} |
| Compare candidates | {len(compare_rows)} |
| Screenshot files | {len(screenshot_files)} |
| Latest run evidence files | {len(run_files)} |

## Evidence Files

| Artifact | Path | Type | Bytes |
| --- | --- | --- | ---: |
{rows}

## Ready Model Families

| Family | Ready Models |
| --- | ---: |
{family_rows or "| NA | 0 |"}

## Source Readiness

| Source | Registered | Ready | Guarded | Needs Dependency | Needs Python 3.11 |
| --- | ---: | ---: | ---: | ---: | ---: |
{source_rows or "| NA | 0 | 0 | 0 | 0 | 0 |"}

## Screenshot Set

{", ".join(screenshot_files) if screenshot_files else "No screenshot files found."}

## Latest Run Evidence Files

{", ".join(run_files) if run_files else "No run evidence files found."}

## How To Refresh

```bash
PYTHON_BIN=python3 \\
ROWS=180 \\
SENSITIVITY_SAMPLES=1 \\
APP_URL=http://localhost:8501 \\
scripts/full_regression_check.sh
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Markdown evidence index from latest DeepUplift artifacts.")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output", default="docs/DEEPUplift_AGENT_EVIDENCE_INDEX.md")
    args = parser.parse_args()

    markdown = build_index(Path(args.reports_dir), Path(args.runs_dir))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
