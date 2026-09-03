from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path
from typing import Iterable


DOC_FILES = [
    Path("README.md"),
    Path("docs/RESUME_DEEPUplift_AGENT.md"),
    Path("docs/RESUME_DEEPUplift_AGENT_SNAPSHOT.md"),
    Path("docs/RESUME_BULLETS_CN.md"),
    Path("docs/DEEPUplift_AGENT_ONEPAGER_CN.md"),
    Path("docs/INTERVIEW_DIFFICULTIES_AND_SOLUTIONS.md"),
    Path("docs/INTERVIEW_DEEPUplift_AGENT_DEEP_DIVE.md"),
    Path("docs/AGENT_INTERVIEW_TRANSCRIPT_SAMPLE.md"),
    Path("docs/DEEPUplift_AGENT_INTERVIEW_EVIDENCE_PACK.md"),
    Path("docs/DEEPUplift_MODEL_DECONSTRUCTION_INDEX.md"),
    Path("docs/INTERVIEW_QA_REHEARSAL_CN.md"),
    Path("docs/INTERVIEW_DEMO_CHECKLIST_CN.md"),
    Path("docs/DEEPUplift_AGENT_ARCHITECTURE.md"),
    Path("docs/DEEPUplift_AGENT_2_0_ARCHITECTURE.md"),
    Path("docs/DEEPUplift_AGENT_PRODUCTION_ROADMAP.md"),
    Path("docs/DEEPUplift_AGENT_MODEL_BACKEND_STRATEGY.md"),
    Path("docs/DEEPUplift_AGENT_MODEL_READINESS_TREND.md"),
    Path("docs/DEEPUplift_AGENT_OFFLINE_ONLINE_PLAYBOOK.md"),
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
    Path("docs/UPLIFT_INDUSTRIAL_SCENARIO_LAB.md"),
    Path("docs/UPLIFT_AGENT_INTERVIEW_RUNBOOK.md"),
    Path("docs/UPLIFT_DEEP_MODEL_TRAINING_EVIDENCE.md"),
    Path("docs/DEEPUplift_DEEP_MODEL_TUNING_BENCHMARK.md"),
    Path("docs/DEEPUplift_DEEP_MODEL_OBJECTIVE_DIAGNOSTICS.md"),
    Path("docs/DEEPUplift_ALGORITHM_CLAIM_LEDGER.md"),
    Path("docs/DEEPUplift_PAPER_REPRODUCTION_GAP_LEDGER.md"),
    Path("docs/DEEPUplift_MODEL_EVIDENCE_PROMOTION_MATRIX.md"),
    Path("docs/DEEPUplift_MODEL_UPGRADE_RECIPE_CARDS.md"),
    Path("docs/DEEPUplift_DEEP_MODEL_TELEMETRY_GAP_MATRIX.md"),
    Path("docs/DEEPUplift_DEEP_MODEL_TELEMETRY_SMOKE.md"),
    Path("docs/DEEPUplift_DEEP_MODEL_FORWARD_HOOK_CONTRACTS.md"),
    Path("docs/DEEPUplift_DEEP_MODEL_FORWARD_HOOK_SMOKE.md"),
    Path("docs/DEEPUplift_DEEP_MODEL_HOOK_TRAINING_BRIDGE.md"),
    Path("docs/DEEPUplift_DEEP_MODEL_TRAINER_COLLECTOR_SMOKE.md"),
    Path("docs/DEEPUplift_DEEP_MODEL_TELEMETRY_BENCHMARK_LINKAGE.md"),
    Path("docs/DEEPUplift_DEEP_MODEL_TELEMETRY_ABLATION_GATE.md"),
    Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_INTERPRETATION.md"),
    Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_PROMOTION_MATRIX.md"),
    Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_NEXT_EXPERIMENT_PLAN.md"),
    Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_CONTRACT.md"),
    Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_CONTRACT_SMOKE.md"),
    Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_RECOMMENDED_COMMAND_SMOKE.md"),
    Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_SMOKE_COVERAGE.md"),
    Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_SMOKE_COVERAGE_DIFF.md"),
    Path("docs/DEEPUplift_PAPER_LEVEL_BENCHMARK.md"),
    Path("docs/DEEPUplift_BENCHMARK_V3_INTERPRETATION.md"),
    Path("docs/DEEPUplift_REGRESSION_WARNING_AUDIT.md"),
    Path("docs/DEEPUplift_PROMOTION_LAUNCH_CARDS.md"),
    Path("docs/DEEPUplift_REGRESSION_WARNING_TRIAGE.md"),
    Path("docs/DEEPUplift_PILOT_READINESS_SCORECARD.md"),
    Path("docs/DEEPUplift_PILOT_EXPERIMENT_PLAN.md"),
    Path("docs/DEEPUplift_PILOT_EXPERIMENT_COMMAND_SMOKE.md"),
    Path("docs/DEEPUplift_GLOSSARY.md"),
    Path("docs/DEEPUplift_AGENT_DEMO_WALKTHROUGH.md"),
    Path("docs/DEEPUplift_AGENT_RISK_REGISTER.md"),
    Path("docs/DEEPUplift_AGENT_EVIDENCE_INDEX.md"),
    Path("docs/DEEPUplift_AGENT_RELEASE_NOTE.md"),
    Path("docs/UPLIFT_AGENT_ITERATION_LOG.md"),
]


def latest(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def latest_dir_with_file(root: Path, pattern: str, filename: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in matches:
        if path.is_dir() and (path / filename).exists():
            return path
    return None


def latest_run_evidence(runs_dir: Path) -> Path | None:
    run_dirs = [
        path
        for path in runs_dir.glob("*")
        if path.is_dir() and (path / "metrics.json").exists() and (path / "config.json").exists()
    ]
    return max(run_dirs, key=lambda path: path.stat().st_mtime) if run_dirs else None


def iter_files(path: Path | None) -> Iterable[Path]:
    if path is None:
        return []
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.is_file())
    return []


def build_manifest(paths: list[Path], output: Path) -> dict:
    return {
        "status": "ok",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "output": str(output),
        "files": [{"path": str(path), "bytes": path.stat().st_size} for path in paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a zip bundle for DeepUplift Agent interview/demo evidence.")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output", default="reports/deepuplift_agent_interview_pack_latest.zip")
    parser.add_argument("--manifest", default="reports/deepuplift_agent_interview_pack_manifest_latest.json")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    runs_dir = Path(args.runs_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    evidence_paths: list[Path] = []
    evidence_paths.extend(path for path in DOC_FILES if path.exists())
    latest_artifacts = [
        latest(reports_dir, "model_catalog_audit_*.json"),
        latest(reports_dir, "model_catalog_snapshot_*.csv"),
        latest(reports_dir, "model_source_readiness_*.csv"),
        latest(reports_dir, "model_family_readiness_*.csv"),
        latest(reports_dir, "agent_regression_*.json"),
        latest(reports_dir, "agent_v2_regression_latest.json"),
        latest(reports_dir, "model_deconstruction_catalog_latest.json"),
        latest(reports_dir, "model_deconstruction_agent_latest.json"),
        latest(reports_dir, "deep_model_training_evidence_latest.json"),
        latest(reports_dir, "deep_model_training_leaderboard_latest.csv"),
        latest(reports_dir, "deep_model_tuned_benchmark_latest.json"),
        latest(reports_dir, "deep_model_tuned_benchmark_leaderboard_latest.csv"),
        latest(reports_dir, "deep_model_tuned_benchmark_battle_cards_latest.csv"),
        latest(reports_dir, "deep_model_tuned_benchmark_run_diff_latest.json"),
        latest(reports_dir, "deep_model_tuned_benchmark_manifest_latest.json"),
        latest(reports_dir, "deep_model_objective_diagnostics_latest.json"),
        latest(reports_dir, "deep_model_objective_diagnostics_latest.csv"),
        latest(reports_dir, "algorithm_claim_ledger_latest.json"),
        latest(reports_dir, "algorithm_claim_ledger_latest.csv"),
        latest(reports_dir, "paper_reproduction_gap_ledger_latest.json"),
        latest(reports_dir, "paper_reproduction_gap_ledger_latest.csv"),
        latest(reports_dir, "model_evidence_promotion_matrix_latest.json"),
        latest(reports_dir, "model_evidence_promotion_matrix_latest.csv"),
        latest(reports_dir, "model_upgrade_recipe_cards_latest.json"),
        latest(reports_dir, "model_upgrade_recipe_cards_latest.csv"),
        latest(reports_dir, "deep_model_telemetry_gap_matrix_latest.json"),
        latest(reports_dir, "deep_model_telemetry_gap_matrix_latest.csv"),
        latest(reports_dir, "deep_model_telemetry_smoke_latest.json"),
        latest(reports_dir, "deep_model_telemetry_smoke_latest.csv"),
        latest(reports_dir, "deep_model_forward_hook_contracts_latest.json"),
        latest(reports_dir, "deep_model_forward_hook_contracts_latest.csv"),
        latest(reports_dir, "deep_model_forward_hook_smoke_latest.json"),
        latest(reports_dir, "deep_model_forward_hook_smoke_latest.csv"),
        latest(reports_dir, "deep_model_hook_training_bridge_latest.json"),
        latest(reports_dir, "deep_model_hook_training_bridge_latest.csv"),
        latest(reports_dir, "deep_model_trainer_collector_smoke_latest.json"),
        latest(reports_dir, "deep_model_trainer_collector_models_latest.csv"),
        latest(reports_dir, "deep_model_trainer_collector_epochs_latest.csv"),
        latest(reports_dir, "deep_model_telemetry_benchmark_linkage_latest.json"),
        latest(reports_dir, "deep_model_telemetry_benchmark_linkage_latest.csv"),
        latest(reports_dir, "deep_model_telemetry_ablation_gate_latest.json"),
        latest(reports_dir, "deep_model_telemetry_ablation_gate_latest.csv"),
        latest(reports_dir, "deep_model_ablation_interpretation_latest.json"),
        latest(reports_dir, "deep_model_ablation_interpretation_latest.csv"),
        latest(reports_dir, "deep_model_ablation_promotion_matrix_latest.json"),
        latest(reports_dir, "deep_model_ablation_promotion_matrix_latest.csv"),
        latest(reports_dir, "deep_model_ablation_next_experiment_plan_latest.json"),
        latest(reports_dir, "deep_model_ablation_next_experiment_plan_latest.csv"),
        latest(reports_dir, "deep_model_ablation_command_contract_latest.json"),
        latest(reports_dir, "deep_model_ablation_command_contract_latest.csv"),
        latest(reports_dir, "deep_model_ablation_command_contract_smoke_latest.json"),
        latest(reports_dir, "deep_model_ablation_command_contract_smoke_latest.csv"),
        latest(reports_dir, "deep_model_ablation_recommended_command_smoke_latest.json"),
        latest(reports_dir, "deep_model_ablation_recommended_command_smoke_latest.csv"),
        latest(reports_dir, "deep_model_ablation_command_smoke_coverage_latest.json"),
        latest(reports_dir, "deep_model_ablation_command_smoke_coverage_latest.csv"),
        latest(reports_dir, "deep_model_ablation_command_smoke_coverage_diff_latest.json"),
        latest(reports_dir, "deep_model_ablation_command_smoke_coverage_diff_latest.csv"),
        latest(reports_dir, "paper_benchmark_latest.json"),
        latest(reports_dir, "paper_benchmark_leaderboard_latest.csv"),
        latest(reports_dir, "paper_benchmark_run_diff_latest.json"),
        latest(reports_dir, "paper_benchmark_manifest_latest.json"),
        latest(reports_dir, "paper_benchmark_interpretation_latest.json"),
        latest(reports_dir, "regression_warning_audit_latest.json"),
        latest(reports_dir, "regression_warning_audit_latest.csv"),
        latest(reports_dir, "promotion_launch_cards_latest.json"),
        latest(reports_dir, "promotion_launch_cards_latest.csv"),
        latest(reports_dir, "regression_warning_triage_latest.json"),
        latest(reports_dir, "regression_warning_triage_latest.csv"),
        latest(reports_dir, "pilot_readiness_scorecard_latest.json"),
        latest(reports_dir, "pilot_readiness_scorecard_latest.csv"),
        latest(reports_dir, "pilot_experiment_plan_latest.json"),
        latest(reports_dir, "pilot_experiment_plan_latest.csv"),
        latest(reports_dir, "pilot_experiment_command_smoke_latest.json"),
        latest(reports_dir, "pilot_experiment_command_smoke_latest.csv"),
        latest(reports_dir, "pilot_experiment_command_coverage_latest.json"),
        latest(reports_dir, "pilot_experiment_command_coverage_latest.csv"),
        latest(reports_dir, "glossary_latest.json"),
        latest(reports_dir, "no_ui_compare_manifest_*.json"),
        latest(reports_dir, "interview_materials_audit*.json"),
        latest(reports_dir, "industry_playbook_audit*.json"),
        latest(reports_dir, "industry_source_audit*.json"),
        latest(reports_dir, "industry_source_backlog*.json"),
        latest(reports_dir, "industry_playbook*.json"),
        latest(reports_dir, "interview_evidence_pack*.json"),
        latest(reports_dir, "model_readiness_trend*.csv"),
        latest(reports_dir, "model_readiness_trend*.json"),
        latest(reports_dir, "scenario_model_matrix*.csv"),
        latest(reports_dir, "scenario_model_matrix*.json"),
        latest(reports_dir, "scenario_model_matrix_audit*.json"),
        latest(reports_dir, "industrial_scenario_datasets_smoke_latest.json"),
        latest(reports_dir, "industrial_scenario_training_smoke_latest.json"),
        latest(reports_dir, "industrial_policy_metrics_smoke_latest.json"),
        latest(reports_dir, "resume_scenario_workbench_smoke_latest.json"),
        latest(reports_dir, "industrial_scenario_compare_smoke_latest.json"),
        latest(reports_dir, "industrial_scenario_compare_latest.csv"),
        latest(reports_dir, "ui_screenshots_*"),
        latest_dir_with_file(reports_dir, "ui_screenshots_*", "story.png"),
        latest_dir_with_file(reports_dir, "deep_model_forward_hook_artifacts/*", "CFRNet/representation_samples.csv"),
        latest_dir_with_file(reports_dir, "deep_model_trainer_collector_artifacts/*", "CFRNet/representation_balance_by_epoch.csv"),
        Path("docs/model_deconstruction"),
        latest_run_evidence(runs_dir),
    ]
    for artifact in latest_artifacts:
        evidence_paths.extend(iter_files(artifact))

    unique_paths = []
    seen: set[str] = set()
    for path in evidence_paths:
        key = str(path)
        if key not in seen:
            unique_paths.append(path)
            seen.add(key)

    manifest = build_manifest(unique_paths, output)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for path in unique_paths:
            archive.write(path, arcname=str(path))

    manifest["zip_bytes"] = output.stat().st_size
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output), "manifest": str(manifest_path), "files": len(unique_paths), "zip_bytes": output.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
