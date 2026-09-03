from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


REQUIRED_EXACT = [
    "MANIFEST.json",
    "README.md",
    "docs/RESUME_DEEPUplift_AGENT.md",
    "docs/RESUME_DEEPUplift_AGENT_SNAPSHOT.md",
    "docs/RESUME_BULLETS_CN.md",
    "docs/DEEPUplift_AGENT_ONEPAGER_CN.md",
    "docs/INTERVIEW_DIFFICULTIES_AND_SOLUTIONS.md",
    "docs/INTERVIEW_DEEPUplift_AGENT_DEEP_DIVE.md",
    "docs/AGENT_INTERVIEW_TRANSCRIPT_SAMPLE.md",
    "docs/DEEPUplift_AGENT_INTERVIEW_EVIDENCE_PACK.md",
    "docs/DEEPUplift_MODEL_DECONSTRUCTION_INDEX.md",
    "docs/model_deconstruction/CFRNet.md",
    "docs/model_deconstruction/DRLearnerGBM.md",
    "docs/model_deconstruction/DESCN.md",
    "docs/model_deconstruction/EFIN.md",
    "docs/model_deconstruction/MultiDRLearnerGBM.md",
    "docs/model_deconstruction/TLearnerGBM.md",
    "docs/INTERVIEW_QA_REHEARSAL_CN.md",
    "docs/INTERVIEW_DEMO_CHECKLIST_CN.md",
    "docs/DEEPUplift_AGENT_ARCHITECTURE.md",
    "docs/DEEPUplift_AGENT_2_0_ARCHITECTURE.md",
    "docs/DEEPUplift_AGENT_PRODUCTION_ROADMAP.md",
    "docs/DEEPUplift_AGENT_MODEL_BACKEND_STRATEGY.md",
    "docs/DEEPUplift_AGENT_MODEL_READINESS_TREND.md",
    "docs/DEEPUplift_AGENT_OFFLINE_ONLINE_PLAYBOOK.md",
    "docs/INDUSTRIAL_UPLIFT_CASE_STUDIES.md",
    "docs/UPLIFT_BUSINESS_SCENARIOS.md",
    "docs/UPLIFT_INTERVIEW_INDUSTRY_DEEP_DIVE.md",
    "docs/UPLIFT_MODEL_SELECTION_PLAYBOOK.md",
    "docs/UPLIFT_COUPON_ADS_GROWTH_PLAYBOOK.md",
    "docs/UPLIFT_ONLINE_EXPERIMENT_AND_INCREMENTALITY.md",
    "docs/UPLIFT_INDUSTRY_REFERENCES.md",
    "docs/INDUSTRY_SOURCE_QUALITY_AUDIT.md",
    "docs/UPLIFT_INDUSTRY_SOURCE_ADOPTION_BACKLOG.md",
    "docs/UPLIFT_SCENARIO_MODEL_MATRIX.md",
    "docs/UPLIFT_SCENARIO_MODEL_MATRIX_INTERVIEW_QA.md",
    "docs/UPLIFT_AGENT_INTERVIEW_RUNBOOK.md",
    "docs/DEEPUplift_DEEP_MODEL_TUNING_BENCHMARK.md",
    "docs/DEEPUplift_DEEP_MODEL_OBJECTIVE_DIAGNOSTICS.md",
    "docs/DEEPUplift_ALGORITHM_CLAIM_LEDGER.md",
    "docs/DEEPUplift_PAPER_REPRODUCTION_GAP_LEDGER.md",
    "docs/DEEPUplift_MODEL_EVIDENCE_PROMOTION_MATRIX.md",
    "docs/DEEPUplift_MODEL_UPGRADE_RECIPE_CARDS.md",
    "docs/DEEPUplift_DEEP_MODEL_TELEMETRY_GAP_MATRIX.md",
    "docs/DEEPUplift_DEEP_MODEL_TELEMETRY_SMOKE.md",
    "docs/DEEPUplift_DEEP_MODEL_FORWARD_HOOK_CONTRACTS.md",
    "docs/DEEPUplift_DEEP_MODEL_FORWARD_HOOK_SMOKE.md",
    "docs/DEEPUplift_DEEP_MODEL_HOOK_TRAINING_BRIDGE.md",
    "docs/DEEPUplift_DEEP_MODEL_TRAINER_COLLECTOR_SMOKE.md",
    "docs/DEEPUplift_DEEP_MODEL_TELEMETRY_BENCHMARK_LINKAGE.md",
    "docs/DEEPUplift_DEEP_MODEL_TELEMETRY_ABLATION_GATE.md",
    "docs/DEEPUplift_DEEP_MODEL_ABLATION_INTERPRETATION.md",
    "docs/DEEPUplift_DEEP_MODEL_ABLATION_PROMOTION_MATRIX.md",
    "docs/DEEPUplift_DEEP_MODEL_ABLATION_NEXT_EXPERIMENT_PLAN.md",
    "docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_CONTRACT.md",
    "docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_CONTRACT_SMOKE.md",
    "docs/DEEPUplift_DEEP_MODEL_ABLATION_RECOMMENDED_COMMAND_SMOKE.md",
    "docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_SMOKE_COVERAGE.md",
    "docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_SMOKE_COVERAGE_DIFF.md",
    "docs/DEEPUplift_BENCHMARK_V3_INTERPRETATION.md",
    "docs/DEEPUplift_REGRESSION_WARNING_AUDIT.md",
    "docs/DEEPUplift_PROMOTION_LAUNCH_CARDS.md",
    "docs/DEEPUplift_REGRESSION_WARNING_TRIAGE.md",
    "docs/DEEPUplift_PILOT_READINESS_SCORECARD.md",
    "docs/DEEPUplift_PILOT_EXPERIMENT_PLAN.md",
    "docs/DEEPUplift_PILOT_EXPERIMENT_COMMAND_SMOKE.md",
    "docs/DEEPUplift_AGENT_EVIDENCE_INDEX.md",
    "docs/DEEPUplift_AGENT_RELEASE_NOTE.md",
]


REQUIRED_PREFIX_SUFFIX = [
    ("reports/model_catalog_audit_", ".json"),
    ("reports/model_source_readiness_", ".csv"),
    ("reports/model_family_readiness_", ".csv"),
    ("reports/agent_regression_", ".json"),
    ("reports/agent_v2_regression_latest", ".json"),
    ("reports/model_deconstruction_catalog_latest", ".json"),
    ("reports/model_deconstruction_agent_latest", ".json"),
    ("reports/deep_model_tuned_benchmark_latest", ".json"),
    ("reports/deep_model_tuned_benchmark_leaderboard_latest", ".csv"),
    ("reports/deep_model_tuned_benchmark_battle_cards_latest", ".csv"),
    ("reports/deep_model_tuned_benchmark_run_diff_latest", ".json"),
    ("reports/deep_model_tuned_benchmark_manifest_latest", ".json"),
    ("reports/deep_model_objective_diagnostics_latest", ".json"),
    ("reports/algorithm_claim_ledger_latest", ".json"),
    ("reports/algorithm_claim_ledger_latest", ".csv"),
    ("reports/paper_reproduction_gap_ledger_latest", ".json"),
    ("reports/paper_reproduction_gap_ledger_latest", ".csv"),
    ("reports/model_evidence_promotion_matrix_latest", ".json"),
    ("reports/model_evidence_promotion_matrix_latest", ".csv"),
    ("reports/model_upgrade_recipe_cards_latest", ".json"),
    ("reports/model_upgrade_recipe_cards_latest", ".csv"),
    ("reports/deep_model_telemetry_gap_matrix_latest", ".json"),
    ("reports/deep_model_telemetry_gap_matrix_latest", ".csv"),
    ("reports/deep_model_telemetry_smoke_latest", ".json"),
    ("reports/deep_model_telemetry_smoke_latest", ".csv"),
    ("reports/deep_model_forward_hook_contracts_latest", ".json"),
    ("reports/deep_model_forward_hook_contracts_latest", ".csv"),
    ("reports/deep_model_forward_hook_smoke_latest", ".json"),
    ("reports/deep_model_forward_hook_smoke_latest", ".csv"),
    ("reports/deep_model_hook_training_bridge_latest", ".json"),
    ("reports/deep_model_hook_training_bridge_latest", ".csv"),
    ("reports/deep_model_trainer_collector_smoke_latest", ".json"),
    ("reports/deep_model_trainer_collector_models_latest", ".csv"),
    ("reports/deep_model_trainer_collector_epochs_latest", ".csv"),
    ("reports/deep_model_telemetry_benchmark_linkage_latest", ".json"),
    ("reports/deep_model_telemetry_benchmark_linkage_latest", ".csv"),
    ("reports/deep_model_telemetry_ablation_gate_latest", ".json"),
    ("reports/deep_model_telemetry_ablation_gate_latest", ".csv"),
    ("reports/deep_model_ablation_interpretation_latest", ".json"),
    ("reports/deep_model_ablation_interpretation_latest", ".csv"),
    ("reports/deep_model_ablation_promotion_matrix_latest", ".json"),
    ("reports/deep_model_ablation_promotion_matrix_latest", ".csv"),
    ("reports/deep_model_ablation_next_experiment_plan_latest", ".json"),
    ("reports/deep_model_ablation_next_experiment_plan_latest", ".csv"),
    ("reports/deep_model_ablation_command_contract_latest", ".json"),
    ("reports/deep_model_ablation_command_contract_latest", ".csv"),
    ("reports/deep_model_ablation_command_contract_smoke_latest", ".json"),
    ("reports/deep_model_ablation_command_contract_smoke_latest", ".csv"),
    ("reports/deep_model_ablation_recommended_command_smoke_latest", ".json"),
    ("reports/deep_model_ablation_recommended_command_smoke_latest", ".csv"),
    ("reports/deep_model_ablation_command_smoke_coverage_latest", ".json"),
    ("reports/deep_model_ablation_command_smoke_coverage_latest", ".csv"),
    ("reports/deep_model_ablation_command_smoke_coverage_diff_latest", ".json"),
    ("reports/deep_model_ablation_command_smoke_coverage_diff_latest", ".csv"),
    ("reports/paper_benchmark_interpretation_latest", ".json"),
    ("reports/regression_warning_audit_latest", ".json"),
    ("reports/regression_warning_audit_latest", ".csv"),
    ("reports/promotion_launch_cards_latest", ".json"),
    ("reports/promotion_launch_cards_latest", ".csv"),
    ("reports/regression_warning_triage_latest", ".json"),
    ("reports/regression_warning_triage_latest", ".csv"),
    ("reports/pilot_readiness_scorecard_latest", ".json"),
    ("reports/pilot_readiness_scorecard_latest", ".csv"),
    ("reports/pilot_experiment_plan_latest", ".json"),
    ("reports/pilot_experiment_plan_latest", ".csv"),
    ("reports/pilot_experiment_command_smoke_latest", ".json"),
    ("reports/pilot_experiment_command_smoke_latest", ".csv"),
    ("reports/pilot_experiment_command_coverage_latest", ".json"),
    ("reports/pilot_experiment_command_coverage_latest", ".csv"),
    ("reports/no_ui_compare_manifest_", ".json"),
    ("reports/interview_materials_audit", ".json"),
    ("reports/industry_playbook_audit", ".json"),
    ("reports/industry_source_audit", ".json"),
    ("reports/industry_source_backlog", ".json"),
    ("reports/industry_playbook", ".json"),
    ("reports/interview_evidence_pack", ".json"),
    ("reports/model_readiness_trend", ".csv"),
    ("reports/model_readiness_trend", ".json"),
    ("reports/scenario_model_matrix", ".csv"),
    ("reports/scenario_model_matrix", ".json"),
    ("reports/scenario_model_matrix_audit", ".json"),
    ("reports/ui_screenshots_", "/story.png"),
    ("runs/", "/metrics.json"),
    ("runs/", "/predictions.csv"),
    ("runs/", "/run_note.md"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the DeepUplift Agent interview handoff zip.")
    parser.add_argument("--zip", default="reports/deepuplift_agent_interview_pack_latest.zip")
    parser.add_argument("--min-files", type=int, default=30)
    parser.add_argument("--min-bytes", type=int, default=500_000)
    args = parser.parse_args()

    path = Path(args.zip)
    failures: list[str] = []
    if not path.exists():
        raise SystemExit(f"Missing interview pack zip: {path}")
    if path.stat().st_size < args.min_bytes:
        failures.append(f"Zip is too small: {path.stat().st_size} < {args.min_bytes}")

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        name_set = set(names)
        for required in REQUIRED_EXACT:
            if required not in name_set:
                failures.append(f"Missing required file: {required}")
        for prefix, suffix in REQUIRED_PREFIX_SUFFIX:
            if not any(name.startswith(prefix) and name.endswith(suffix) for name in names):
                failures.append(f"Missing required artifact matching {prefix}*{suffix}")
        if len(names) < args.min_files:
            failures.append(f"Too few files in zip: {len(names)} < {args.min_files}")
        try:
            manifest = json.loads(archive.read("MANIFEST.json").decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            manifest = {}
            failures.append(f"Could not read MANIFEST.json: {exc}")

    payload = {
        "status": "fail" if failures else "ok",
        "zip": str(path),
        "zip_bytes": path.stat().st_size,
        "files": len(names) if "names" in locals() else 0,
        "manifest_files": len(manifest.get("files") or []) if isinstance(manifest, dict) else 0,
        "failures": failures,
    }
    print(json.dumps(payload, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
