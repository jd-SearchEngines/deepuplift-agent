from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import zipfile


MIN_REGISTERED_MODELS = 90
MIN_RUNTIME_READY_MODELS = 35


def latest(pattern: str, root: Path) -> Path:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        raise AssertionError(f"No artifact found for pattern: {pattern}")
    return matches[0]


def optional_latest(pattern: str, root: Path) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"Expected JSON object: {path}")
    return payload


def validate_audit(root: Path) -> dict:
    path = latest("model_catalog_audit_*.json", root)
    payload = read_json(path)
    if payload.get("registered_models", 0) < MIN_REGISTERED_MODELS:
        raise AssertionError(f"Too few registered models in {path}")
    if payload.get("ready_models", 0) < MIN_RUNTIME_READY_MODELS:
        raise AssertionError(f"Too few runtime-ready models in {path}")
    if payload.get("missing_descriptions") or payload.get("missing_model_cards") or payload.get("missing_model_presets") or payload.get("failures"):
        raise AssertionError(f"Model audit is not clean: {path}")
    source_readiness = payload.get("source_readiness") or []
    if not source_readiness:
        raise AssertionError(f"Model audit is missing source readiness rows: {path}")
    source_csv = optional_latest("model_source_readiness_*.csv", root)
    if source_csv is None:
        raise AssertionError("No model source readiness CSV found")
    family_csv = optional_latest("model_family_readiness_*.csv", root)
    if family_csv is None:
        raise AssertionError("No model family readiness CSV found")
    return {
        "artifact": "model_audit",
        "path": str(path),
        "registered": payload.get("registered_models"),
        "ready": payload.get("ready_models"),
        "sources": len(source_readiness),
        "source_csv": str(source_csv),
        "family_csv": str(family_csv),
    }


def validate_agent_regression(root: Path) -> dict:
    path = latest("agent_regression_*.json", root)
    payload = read_json(path)
    rows = payload.get("results") or []
    if len(rows) < 2:
        raise AssertionError(f"Agent regression should contain at least two results: {path}")
    failures = [row for row in rows if row.get("status") != "ok"]
    if failures:
        raise AssertionError(f"Agent regression contains failures: {failures}")
    for row in rows:
        if not row.get("run_id") or row.get("qini") is None or row.get("auuc") is None:
            raise AssertionError(f"Regression row is missing key metrics: {row}")
    return {"artifact": "agent_regression", "path": str(path), "results": len(rows)}


def validate_agent_v2_regression(root: Path) -> dict:
    path = optional_latest("agent_v2_regression_latest.json", root)
    if path is None:
        return {"artifact": "agent_v2_regression", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("rows") or []
    if payload.get("status") != "ok" or payload.get("failures"):
        raise AssertionError(f"Agent 2.0 regression failed: {path}")
    if len(rows) < 20:
        raise AssertionError(f"Agent 2.0 regression should cover at least 20 golden questions: {path}")
    if float(payload.get("avg_rubric_ratio") or 0.0) < 0.85:
        raise AssertionError(f"Agent 2.0 average rubric ratio is too low: {path}")
    required_checks = {"causal_parse", "risk_diagnosis", "model_recommendation", "evaluation_online", "evidence_grounding", "interview_packaging", "answer_modes"}
    for row in rows:
        checks = row.get("checks") or {}
        if not required_checks.issubset(checks):
            raise AssertionError(f"Agent 2.0 row missing rubric checks: {row}")
        failed = [name for name in required_checks if not checks.get(name)]
        if failed:
            raise AssertionError(f"Agent 2.0 row failed rubric checks {failed}: {row.get('prompt')}")
    return {
        "artifact": "agent_v2_regression",
        "path": str(path),
        "questions": len(rows),
        "avg_rubric_ratio": payload.get("avg_rubric_ratio"),
    }


def validate_model_deconstruction(root: Path) -> dict:
    catalog_path = optional_latest("model_deconstruction_catalog_latest.json", root)
    agent_path = optional_latest("model_deconstruction_agent_latest.json", root)
    if catalog_path is None or agent_path is None:
        return {"artifact": "model_deconstruction", "status": "skipped", "reason": "not_found"}
    catalog = read_json(catalog_path)
    agent = read_json(agent_path)
    models = catalog.get("models") or []
    rows = agent.get("rows") or []
    if len(models) < 5:
        raise AssertionError(f"Model deconstruction catalog should include at least 5 models: {catalog_path}")
    for model in models:
        if not model.get("model_id") or not model.get("code_paths") or not model.get("formulas") or not model.get("external_sources"):
            raise AssertionError(f"Malformed model deconstruction row: {model}")
    if agent.get("status") != "ok" or agent.get("failures"):
        raise AssertionError(f"Model deconstruction agent regression failed: {agent_path}")
    if len(rows) < 20:
        raise AssertionError(f"Model deconstruction agent should cover at least 20 questions: {agent_path}")
    if float(agent.get("avg_rubric_ratio") or 0.0) < 0.85:
        raise AssertionError(f"Model deconstruction average rubric ratio is too low: {agent_path}")
    required_checks = {"source_code", "math", "training", "evaluation", "license", "interview"}
    for row in rows:
        checks = row.get("checks") or {}
        if not required_checks.issubset(checks):
            raise AssertionError(f"Model deconstruction row missing checks: {row}")
        failed = [name for name in required_checks if not checks.get(name)]
        if failed:
            raise AssertionError(f"Model deconstruction row failed checks {failed}: {row.get('prompt')}")
    return {
        "artifact": "model_deconstruction",
        "catalog": str(catalog_path),
        "agent": str(agent_path),
        "models": len(models),
        "questions": len(rows),
        "avg_rubric_ratio": agent.get("avg_rubric_ratio"),
    }


def validate_deep_model_training_evidence(root: Path) -> dict:
    path = optional_latest("deep_model_training_evidence_latest.json", root)
    if path is None:
        return {"artifact": "deep_model_training_evidence", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("leaderboard") or []
    if payload.get("status") != "ok" or payload.get("failures"):
        raise AssertionError(f"Deep model training evidence failed: {path}")
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if len(ok_rows) < 3:
        raise AssertionError(f"Deep model training evidence should include at least 3 ok model rows: {path}")
    required_models = {"CFRNet", "DragonNet", "EFIN", "DESCN"}
    present_models = {row.get("model") for row in ok_rows}
    missing_models = required_models - present_models
    if missing_models:
        raise AssertionError(f"Deep model training evidence missing model rows {sorted(missing_models)}: {path}")
    for row in ok_rows:
        if row.get("qini_score") is None or row.get("auuc_score") is None:
            raise AssertionError(f"Deep model training row missing ranking metrics: {row}")
        manifest = row.get("evidence_manifest")
        if not manifest or not Path(manifest).exists():
            raise AssertionError(f"Deep model training row missing evidence manifest: {row}")
    csv_path = optional_latest("deep_model_training_leaderboard_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Deep model training leaderboard CSV is missing")
    return {
        "artifact": "deep_model_training_evidence",
        "path": str(path),
        "ok_models": len(ok_rows),
        "csv": str(csv_path),
    }


def validate_deep_model_tuned_benchmark(root: Path) -> dict:
    path = optional_latest("deep_model_tuned_benchmark_latest.json", root)
    if path is None:
        return {"artifact": "deep_model_tuned_benchmark", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    runs = payload.get("runs") or []
    leaderboard = payload.get("leaderboard") or []
    battle_cards = payload.get("battle_cards") or []
    baseline = payload.get("baseline_comparison") or []
    scorecards = payload.get("model_scorecards") or []
    if payload.get("status") != "ok" or payload.get("failures"):
        raise AssertionError(f"Deep model tuned benchmark failed: {path}")
    if len(runs) < 12:
        raise AssertionError(f"Deep model tuned benchmark should contain at least 12 run attempts: {path}")
    ok_rows = [row for row in runs if row.get("status") == "ok"]
    if len(ok_rows) < 8:
        raise AssertionError(f"Deep model tuned benchmark should contain at least 8 successful runs: {path}")
    attempted_models = {row.get("model") for row in runs}
    required_models = {"TLearnerGBM", "DRLearnerGBM", "CFRNet", "DragonNet", "EFIN", "DESCN"}
    missing_models = required_models - attempted_models
    if missing_models:
        raise AssertionError(f"Deep model tuned benchmark missing model attempts {sorted(missing_models)}: {path}")
    required_variants = {
        "CFRNet_low_balance",
        "DragonNet_no_tarreg",
        "EFIN_wide",
        "EFIN_no_attention",
        "EFIN_path_regularized",
        "DESCN_wide_dropout",
        "DESCN_no_constraint",
    }
    attempted_variants = {row.get("variant_id") for row in runs}
    missing_variants = required_variants - attempted_variants
    if missing_variants:
        raise AssertionError(f"Deep model tuned benchmark missing tuned variants {sorted(missing_variants)}: {path}")
    if not leaderboard or not battle_cards or not baseline or not scorecards:
        raise AssertionError(f"Deep model tuned benchmark missing leaderboard, battle cards, baseline comparison or scorecards: {path}")
    for required_track in ["30s", "5min", "15min", "30min"]:
        if not (payload.get("interview_tracks") or {}).get(required_track):
            raise AssertionError(f"Deep model tuned benchmark missing {required_track} interview track: {path}")
    if not payload.get("recommendations"):
        raise AssertionError(f"Deep model tuned benchmark missing recommendations: {path}")
    if not payload.get("failure_attribution") or not payload.get("run_diff"):
        raise AssertionError(f"Deep model tuned benchmark missing failure attribution or run diff: {path}")
    for row in ok_rows:
        if row.get("pehe") is None or row.get("ate_error") is None:
            raise AssertionError(f"Deep tuned benchmark row missing PEHE/ATE error: {row}")
        if row.get("qini") is None or row.get("auuc") is None:
            raise AssertionError(f"Deep tuned benchmark row missing QINI/AUUC: {row}")
        manifest = row.get("evidence_manifest")
        if not manifest or not Path(manifest).exists():
            raise AssertionError(f"Deep tuned benchmark row missing evidence manifest: {row}")
    leaderboard_path = optional_latest("deep_model_tuned_benchmark_leaderboard_latest.csv", root)
    battle_path = optional_latest("deep_model_tuned_benchmark_battle_cards_latest.csv", root)
    diff_path = optional_latest("deep_model_tuned_benchmark_run_diff_latest.json", root)
    manifest_path = optional_latest("deep_model_tuned_benchmark_manifest_latest.json", root)
    if leaderboard_path is None or battle_path is None or diff_path is None or manifest_path is None:
        raise AssertionError("Deep tuned benchmark latest CSV/battle/diff/manifest artifacts are missing")
    manifest_payload = read_json(manifest_path)
    if manifest_payload.get("artifact") != "deep_model_tuned_benchmark" or not manifest_payload.get("run_evidence_manifests"):
        raise AssertionError(f"Deep tuned benchmark manifest is incomplete: {manifest_path}")
    doc_path = Path("docs/DEEPUplift_DEEP_MODEL_TUNING_BENCHMARK.md")
    if not doc_path.exists():
        raise AssertionError("Deep tuned benchmark markdown doc is missing")
    return {
        "artifact": "deep_model_tuned_benchmark",
        "path": str(path),
        "runs": len(runs),
        "ok_runs": len(ok_rows),
        "leaderboard": len(leaderboard),
        "battle_cards": len(battle_cards),
        "scorecards": len(scorecards),
        "leaderboard_csv": str(leaderboard_path),
        "battle_csv": str(battle_path),
        "run_diff": str(diff_path),
        "manifest": str(manifest_path),
        "doc": str(doc_path),
    }


def validate_deep_model_objective_diagnostics(root: Path) -> dict:
    path = optional_latest("deep_model_objective_diagnostics_latest.json", root)
    if path is None:
        return {"artifact": "deep_model_objective_diagnostics", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    cards = payload.get("cards") or []
    if payload.get("status") != "ok":
        raise AssertionError(f"Deep model objective diagnostics is not ok: {path}")
    required_models = {"CFRNet", "DragonNet", "EFIN", "DESCN"}
    present = {row.get("model") for row in cards}
    missing = required_models - present
    if missing:
        raise AssertionError(f"Deep objective diagnostics missing models {sorted(missing)}: {path}")
    required = {
        "model",
        "objective",
        "loss_terms",
        "code_paths",
        "diagnostics",
        "observability_gaps",
        "failure_modes",
        "next_tasks",
        "safe_claim",
        "unsafe_claim",
        "interview_line",
        "evidence_refs",
    }
    for card in cards:
        if not required.issubset(card):
            raise AssertionError(f"Malformed deep objective card: {card}")
        if len(card.get("loss_terms") or []) < 2:
            raise AssertionError(f"Deep objective card should include multiple loss terms: {card.get('model')}")
    csv_path = optional_latest("deep_model_objective_diagnostics_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Deep model objective diagnostics CSV is missing")
    doc_path = Path("docs/DEEPUplift_DEEP_MODEL_OBJECTIVE_DIAGNOSTICS.md")
    if not doc_path.exists():
        raise AssertionError("Deep model objective diagnostics markdown doc is missing")
    return {
        "artifact": "deep_model_objective_diagnostics",
        "path": str(path),
        "models": len(cards),
        "loss_terms": (payload.get("summary") or {}).get("loss_terms"),
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_algorithm_claim_ledger(root: Path) -> dict:
    path = optional_latest("algorithm_claim_ledger_latest.json", root)
    if path is None:
        return {"artifact": "algorithm_claim_ledger", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("rows") or []
    summary = payload.get("summary") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Algorithm claim ledger is not ok: {path}")
    if len(rows) < 24:
        raise AssertionError(f"Algorithm claim ledger should contain at least 24 claim rows: {path}")
    required_models = {"TLearnerGBM", "DRLearnerGBM", "CFRNet", "DragonNet", "EFIN", "DESCN", "MultiDRLearnerGBM"}
    present_models = {row.get("model") for row in rows}
    missing_models = required_models - present_models
    if missing_models:
        raise AssertionError(f"Algorithm claim ledger missing models {sorted(missing_models)}: {path}")
    required_types = {"algorithm_definition", "training_objective", "benchmark_evidence", "license_and_adoption"}
    present_types = {row.get("claim_type") for row in rows}
    missing_types = required_types - present_types
    if missing_types:
        raise AssertionError(f"Algorithm claim ledger missing claim types {sorted(missing_types)}: {path}")
    required_fields = {
        "claim_id",
        "model",
        "family",
        "claim_type",
        "claim",
        "formula_or_test",
        "code_refs",
        "evidence_refs",
        "evidence_level",
        "benchmark_verdict",
        "safe_claim",
        "unsafe_claim",
        "interview_line",
        "ui_route",
        "next_action",
    }
    for row in rows:
        if not required_fields.issubset(row):
            raise AssertionError(f"Malformed algorithm claim row: {row}")
        if not row.get("code_refs") or not row.get("evidence_refs"):
            raise AssertionError(f"Algorithm claim row missing code/evidence refs: {row}")
    if int(summary.get("claims") or 0) != len(rows):
        raise AssertionError(f"Algorithm claim ledger summary count mismatch: {path}")
    csv_path = optional_latest("algorithm_claim_ledger_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Algorithm claim ledger CSV is missing")
    doc_path = Path("docs/DEEPUplift_ALGORITHM_CLAIM_LEDGER.md")
    if not doc_path.exists():
        raise AssertionError("Algorithm claim ledger markdown doc is missing")
    return {
        "artifact": "algorithm_claim_ledger",
        "path": str(path),
        "models": summary.get("models"),
        "claims": len(rows),
        "claim_types": summary.get("claim_type_counts") or {},
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_paper_reproduction_gap_ledger(root: Path) -> dict:
    path = optional_latest("paper_reproduction_gap_ledger_latest.json", root)
    if path is None:
        return {"artifact": "paper_reproduction_gap_ledger", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("rows") or []
    summary = payload.get("summary") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Paper reproduction gap ledger is not ok: {path}")
    if len(rows) < 10:
        raise AssertionError(f"Paper reproduction gap ledger should include paper/repo source rows: {path}")
    required_models = {"TLearnerGBM", "DRLearnerGBM", "CFRNet", "DragonNet", "EFIN", "DESCN", "MultiDRLearnerGBM"}
    present_models = {row.get("model") for row in rows}
    missing_models = required_models - present_models
    if missing_models:
        raise AssertionError(f"Paper reproduction gap ledger missing models {sorted(missing_models)}: {path}")
    required_fields = {
        "model",
        "source_name",
        "source_url",
        "source_type",
        "source_license",
        "license_risk",
        "adoption_decision",
        "paper_or_repo_claim",
        "local_reproduction_scope",
        "proof_level",
        "matched_formula",
        "matched_code_refs",
        "benchmark_verdict",
        "implementation_gaps",
        "validation_needed",
        "evidence_refs",
        "safe_claim",
        "unsafe_claim",
        "interview_line",
    }
    for row in rows:
        if not required_fields.issubset(row):
            raise AssertionError(f"Malformed paper reproduction row: {row}")
        if not row.get("matched_code_refs") or not row.get("evidence_refs"):
            raise AssertionError(f"Paper reproduction row missing code/evidence refs: {row}")
    if int(summary.get("source_rows") or 0) != len(rows):
        raise AssertionError(f"Paper reproduction gap summary count mismatch: {path}")
    csv_path = optional_latest("paper_reproduction_gap_ledger_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Paper reproduction gap ledger CSV is missing")
    doc_path = Path("docs/DEEPUplift_PAPER_REPRODUCTION_GAP_LEDGER.md")
    if not doc_path.exists():
        raise AssertionError("Paper reproduction gap ledger markdown doc is missing")
    return {
        "artifact": "paper_reproduction_gap_ledger",
        "path": str(path),
        "models": summary.get("models"),
        "source_rows": len(rows),
        "proof_levels": summary.get("proof_level_counts") or {},
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_model_evidence_promotion_matrix(root: Path) -> dict:
    path = optional_latest("model_evidence_promotion_matrix_latest.json", root)
    if path is None:
        return {"artifact": "model_evidence_promotion_matrix", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("rows") or []
    summary = payload.get("summary") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Model evidence promotion matrix is not ok: {path}")
    required_models = {"TLearnerGBM", "DRLearnerGBM", "CFRNet", "DragonNet", "EFIN", "DESCN", "MultiDRLearnerGBM"}
    present_models = {row.get("model") for row in rows}
    missing_models = required_models - present_models
    if missing_models:
        raise AssertionError(f"Model evidence promotion matrix missing models {sorted(missing_models)}: {path}")
    required_fields = {
        "model",
        "display_name",
        "family",
        "evidence_stage",
        "promotion_tier",
        "evidence_score",
        "paper_proof_level",
        "paper_verdict",
        "tuned_verdict",
        "objective_status",
        "training_status",
        "license_gate",
        "launch_decision",
        "claim_rows",
        "source_rows",
        "safe_claim",
        "blocked_claim",
        "next_evidence_step",
        "demo_route",
        "evidence_refs",
        "code_refs",
    }
    for row in rows:
        if not required_fields.issubset(row):
            raise AssertionError(f"Malformed model evidence promotion row: {row}")
        if not row.get("safe_claim") or not row.get("blocked_claim") or not row.get("evidence_refs"):
            raise AssertionError(f"Promotion matrix row missing claim/evidence boundary: {row}")
        score = int(row.get("evidence_score") or 0)
        if score < 0 or score > 100:
            raise AssertionError(f"Promotion matrix evidence score out of range: {row}")
    if int(summary.get("models") or 0) != len(rows):
        raise AssertionError(f"Promotion matrix summary count mismatch: {path}")
    if not summary.get("stage_counts") or not summary.get("promotion_tier_counts"):
        raise AssertionError(f"Promotion matrix missing stage/tier summaries: {path}")
    csv_path = optional_latest("model_evidence_promotion_matrix_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Model evidence promotion matrix CSV is missing")
    doc_path = Path("docs/DEEPUplift_MODEL_EVIDENCE_PROMOTION_MATRIX.md")
    if not doc_path.exists():
        raise AssertionError("Model evidence promotion matrix markdown doc is missing")
    return {
        "artifact": "model_evidence_promotion_matrix",
        "path": str(path),
        "models": len(rows),
        "stages": summary.get("stage_counts") or {},
        "tiers": summary.get("promotion_tier_counts") or {},
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_model_upgrade_recipe_cards(root: Path) -> dict:
    path = optional_latest("model_upgrade_recipe_cards_latest.json", root)
    if path is None:
        return {"artifact": "model_upgrade_recipe_cards", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("rows") or []
    summary = payload.get("summary") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Model upgrade recipe cards are not ok: {path}")
    required_models = {"TLearnerGBM", "DRLearnerGBM", "CFRNet", "DragonNet", "EFIN", "DESCN", "MultiDRLearnerGBM"}
    present_models = {row.get("model") for row in rows}
    missing_models = required_models - present_models
    if missing_models:
        raise AssertionError(f"Model upgrade recipe cards missing models {sorted(missing_models)}: {path}")
    required_fields = {
        "recipe_id",
        "model",
        "current_stage",
        "promotion_tier",
        "upgrade_goal",
        "hypothesis",
        "command",
        "dataset_scope",
        "metrics_to_watch",
        "pass_gate",
        "fail_action",
        "expected_artifacts",
        "owner_role",
        "ui_route",
        "interview_line",
        "evidence_refs",
    }
    for row in rows:
        if not required_fields.issubset(row):
            raise AssertionError(f"Malformed model upgrade recipe row: {row}")
        if not row.get("command") or not row.get("pass_gate") or not row.get("expected_artifacts"):
            raise AssertionError(f"Model upgrade recipe row missing execution contract: {row}")
    if int(summary.get("recipes") or 0) != len(rows):
        raise AssertionError(f"Model upgrade recipe summary count mismatch: {path}")
    csv_path = optional_latest("model_upgrade_recipe_cards_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Model upgrade recipe cards CSV is missing")
    doc_path = Path("docs/DEEPUplift_MODEL_UPGRADE_RECIPE_CARDS.md")
    if not doc_path.exists():
        raise AssertionError("Model upgrade recipe cards markdown doc is missing")
    return {
        "artifact": "model_upgrade_recipe_cards",
        "path": str(path),
        "models": summary.get("models"),
        "recipes": len(rows),
        "goals": summary.get("upgrade_goal_counts") or {},
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_deep_model_telemetry_gap_matrix(root: Path) -> dict:
    path = optional_latest("deep_model_telemetry_gap_matrix_latest.json", root)
    if path is None:
        return {"artifact": "deep_model_telemetry_gap_matrix", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("rows") or []
    summary = payload.get("summary") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Deep model telemetry gap matrix is not ok: {path}")
    required_models = {"CFRNet", "DragonNet", "EFIN", "DESCN"}
    present_models = {row.get("model") for row in rows}
    missing_models = required_models - present_models
    if missing_models:
        raise AssertionError(f"Deep model telemetry gap matrix missing models {sorted(missing_models)}: {path}")
    if len(rows) < 10:
        raise AssertionError(f"Deep model telemetry gap matrix should include loss-term telemetry rows: {path}")
    required_fields = {
        "telemetry_id",
        "model",
        "loss_term",
        "formula",
        "telemetry_family",
        "telemetry_status",
        "current_signal",
        "current_metrics",
        "missing_telemetry",
        "why_needed",
        "implementation_hook",
        "pass_gate",
        "fail_action",
        "expected_metric_link",
        "priority",
        "ui_route",
        "interview_line",
        "safe_claim",
        "blocked_claim",
        "code_refs",
        "evidence_refs",
    }
    for row in rows:
        if not required_fields.issubset(row):
            raise AssertionError(f"Malformed deep model telemetry row: {row}")
        if row.get("priority") not in {"P0", "P1", "P2", "P3"}:
            raise AssertionError(f"Deep model telemetry priority out of range: {row}")
        if not row.get("missing_telemetry") or not row.get("implementation_hook") or not row.get("pass_gate"):
            raise AssertionError(f"Deep model telemetry row missing execution contract: {row}")
    if int(summary.get("telemetry_rows") or 0) != len(rows):
        raise AssertionError(f"Deep model telemetry summary count mismatch: {path}")
    if int(summary.get("p0_rows") or 0) < 4:
        raise AssertionError(f"Deep model telemetry matrix should expose several P0 gaps: {path}")
    csv_path = optional_latest("deep_model_telemetry_gap_matrix_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Deep model telemetry gap matrix CSV is missing")
    doc_path = Path("docs/DEEPUplift_DEEP_MODEL_TELEMETRY_GAP_MATRIX.md")
    if not doc_path.exists():
        raise AssertionError("Deep model telemetry gap matrix markdown doc is missing")
    return {
        "artifact": "deep_model_telemetry_gap_matrix",
        "path": str(path),
        "models": summary.get("models"),
        "rows": len(rows),
        "priorities": summary.get("priority_counts") or {},
        "families": summary.get("telemetry_family_counts") or {},
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_deep_model_telemetry_smoke(root: Path) -> dict:
    path = optional_latest("deep_model_telemetry_smoke_latest.json", root)
    if path is None:
        return {"artifact": "deep_model_telemetry_smoke", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("rows") or []
    summary = payload.get("summary") or {}
    if payload.get("status") not in {"ok", "review"}:
        raise AssertionError(f"Deep model telemetry smoke has invalid status: {path}")
    required_models = {"CFRNet", "DragonNet", "EFIN", "DESCN"}
    present_models = {row.get("model") for row in rows}
    missing_models = required_models - present_models
    if missing_models:
        raise AssertionError(f"Deep model telemetry smoke missing models {sorted(missing_models)}: {path}")
    required_fields = {
        "model",
        "status",
        "run_dir",
        "epochs",
        "train_loss_delta",
        "qini_score",
        "auuc_score",
        "prediction_rows",
        "uplift_score_std",
        "top10_true_uplift_gain",
        "manifest_files",
        "basic_telemetry_pass",
        "advanced_hook_status",
        "advanced_hook_needed",
        "pass_gate",
        "fail_action",
        "ui_route",
        "interview_line",
    }
    for row in rows:
        if not required_fields.issubset(row):
            raise AssertionError(f"Malformed deep model telemetry smoke row: {row}")
        if int(row.get("epochs") or 0) < 1 or int(row.get("prediction_rows") or 0) < 1:
            raise AssertionError(f"Deep model telemetry smoke row has no history/predictions: {row}")
        if not row.get("basic_telemetry_pass"):
            raise AssertionError(f"Deep model telemetry basic pass should be true in default smoke: {row}")
    if int(summary.get("basic_telemetry_pass") or 0) < 4:
        raise AssertionError(f"Deep model telemetry smoke should pass four deep models: {path}")
    csv_path = optional_latest("deep_model_telemetry_smoke_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Deep model telemetry smoke CSV is missing")
    doc_path = Path("docs/DEEPUplift_DEEP_MODEL_TELEMETRY_SMOKE.md")
    if not doc_path.exists():
        raise AssertionError("Deep model telemetry smoke markdown doc is missing")
    return {
        "artifact": "deep_model_telemetry_smoke",
        "path": str(path),
        "models": summary.get("models"),
        "rows": len(rows),
        "advanced_hook_gaps": summary.get("advanced_hook_gaps"),
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_deep_model_forward_hook_contracts(root: Path) -> dict:
    path = optional_latest("deep_model_forward_hook_contracts_latest.json", root)
    if path is None:
        return {"artifact": "deep_model_forward_hook_contracts", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("rows") or []
    summary = payload.get("summary") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Deep model forward hook contracts are not ok: {path}")
    required_models = {"CFRNet", "DragonNet", "EFIN", "DESCN"}
    present_models = {row.get("model") for row in rows}
    missing_models = required_models - present_models
    if missing_models:
        raise AssertionError(f"Deep model forward hook contracts missing models {sorted(missing_models)}: {path}")
    if len(rows) < 8:
        raise AssertionError(f"Deep model forward hook contracts should include at least eight hook contracts: {path}")
    required_fields = {
        "contract_id",
        "model",
        "hook_name",
        "priority",
        "contract_type",
        "status",
        "guarded",
        "source_code_entry",
        "current_return_contract",
        "expected_tensor_keys",
        "tensor_shape_contract",
        "artifact_files",
        "producer",
        "consumer",
        "smoke_test",
        "pass_gate",
        "fail_action",
        "fallback_behavior",
        "backward_compat",
        "migration_step",
        "linked_gap_ids",
        "telemetry_smoke_status",
        "basic_telemetry_pass",
        "ui_route",
        "interview_line",
        "safe_claim",
        "blocked_claim",
        "evidence_refs",
    }
    for row in rows:
        if not required_fields.issubset(row):
            raise AssertionError(f"Malformed deep model forward hook contract row: {row}")
        if row.get("priority") not in {"P0", "P1", "P2", "P3"}:
            raise AssertionError(f"Forward hook contract priority out of range: {row}")
        if row.get("status") != "contract_ready" or not row.get("guarded"):
            raise AssertionError(f"Forward hook contract must be guarded and contract_ready: {row}")
        if not row.get("expected_tensor_keys") or not row.get("artifact_files"):
            raise AssertionError(f"Forward hook contract missing tensor/artifact contract: {row}")
        if not row.get("fallback_behavior") or not row.get("backward_compat"):
            raise AssertionError(f"Forward hook contract missing compatibility guard: {row}")
    if int(summary.get("contracts") or 0) != len(rows):
        raise AssertionError(f"Deep model forward hook contract summary count mismatch: {path}")
    if int(summary.get("p0_contracts") or 0) < 6:
        raise AssertionError(f"Deep model forward hook contracts should expose several P0 contracts: {path}")
    csv_path = optional_latest("deep_model_forward_hook_contracts_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Deep model forward hook contracts CSV is missing")
    doc_path = Path("docs/DEEPUplift_DEEP_MODEL_FORWARD_HOOK_CONTRACTS.md")
    if not doc_path.exists():
        raise AssertionError("Deep model forward hook contracts markdown doc is missing")
    return {
        "artifact": "deep_model_forward_hook_contracts",
        "path": str(path),
        "models": summary.get("models"),
        "contracts": len(rows),
        "p0_contracts": summary.get("p0_contracts"),
        "types": summary.get("contract_type_counts") or {},
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_deep_model_forward_hook_smoke(root: Path) -> dict:
    path = optional_latest("deep_model_forward_hook_smoke_latest.json", root)
    if path is None:
        return {"artifact": "deep_model_forward_hook_smoke", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("rows") or []
    summary = payload.get("summary") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Deep model forward hook smoke is not ok: {path}")
    required_models = {"CFRNet", "DragonNet", "EFIN", "DESCN"}
    present_models = {row.get("model") for row in rows}
    missing_models = required_models - present_models
    if missing_models:
        raise AssertionError(f"Deep model forward hook smoke missing models {sorted(missing_models)}: {path}")
    required_contracts = {
        "CFRNet.phi_x_representation",
        "DragonNet.e_hat_propensity",
        "DragonNet.targeted_regularization_components",
        "EFIN.interaction_attention",
        "EFIN.uplift_path_contribution",
        "DESCN.propensity_and_exposure",
        "DESCN.entire_space_heads",
        "DESCN.cross_head_constraints",
    }
    present_contracts = {row.get("contract_id") for row in rows}
    missing_contracts = required_contracts - present_contracts
    if missing_contracts:
        raise AssertionError(f"Deep model forward hook smoke missing contracts {sorted(missing_contracts)}: {path}")
    required_fields = {
        "contract_id",
        "model",
        "hook_name",
        "status",
        "tensor_keys_exported",
        "artifact_files",
        "metric_summary",
        "checks",
        "pass_gate",
        "remaining_gap",
        "safe_claim",
        "interview_line",
    }
    for row in rows:
        if not required_fields.issubset(row):
            raise AssertionError(f"Malformed deep model forward hook smoke row: {row}")
        if row.get("status") != "ok":
            raise AssertionError(f"Forward hook smoke contract did not pass: {row}")
        if not row.get("tensor_keys_exported") or not row.get("artifact_files"):
            raise AssertionError(f"Forward hook smoke missing tensor keys or artifacts: {row}")
        for artifact in row.get("artifact_files") or []:
            artifact_path = Path(str(artifact))
            if not artifact_path.exists():
                raise AssertionError(f"Forward hook smoke artifact file is missing: {artifact_path}")
    if int(summary.get("contracts") or 0) != len(rows):
        raise AssertionError(f"Deep model forward hook smoke summary count mismatch: {path}")
    if int(summary.get("ok_contracts") or 0) < 8:
        raise AssertionError(f"Deep model forward hook smoke should pass eight contracts: {path}")
    csv_path = optional_latest("deep_model_forward_hook_smoke_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Deep model forward hook smoke CSV is missing")
    doc_path = Path("docs/DEEPUplift_DEEP_MODEL_FORWARD_HOOK_SMOKE.md")
    if not doc_path.exists():
        raise AssertionError("Deep model forward hook smoke markdown doc is missing")
    return {
        "artifact": "deep_model_forward_hook_smoke",
        "path": str(path),
        "models": summary.get("models"),
        "contracts": len(rows),
        "ok_contracts": summary.get("ok_contracts"),
        "artifact_files": summary.get("artifact_files"),
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_deep_model_hook_training_bridge(root: Path) -> dict:
    path = optional_latest("deep_model_hook_training_bridge_latest.json", root)
    if path is None:
        return {"artifact": "deep_model_hook_training_bridge", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("rows") or []
    summary = payload.get("summary") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Deep model hook training bridge is not ok: {path}")
    required_models = {"CFRNet", "DragonNet", "EFIN", "DESCN"}
    present_models = {row.get("model") for row in rows}
    missing_models = required_models - present_models
    if missing_models:
        raise AssertionError(f"Deep model hook training bridge missing models {sorted(missing_models)}: {path}")
    required_contracts = {
        "CFRNet.phi_x_representation",
        "DragonNet.e_hat_propensity",
        "DragonNet.targeted_regularization_components",
        "EFIN.interaction_attention",
        "EFIN.uplift_path_contribution",
        "DESCN.propensity_and_exposure",
        "DESCN.entire_space_heads",
        "DESCN.cross_head_constraints",
    }
    present_contracts = {row.get("contract_id") for row in rows}
    missing_contracts = required_contracts - present_contracts
    if missing_contracts:
        raise AssertionError(f"Deep model hook training bridge missing contracts {sorted(missing_contracts)}: {path}")
    required_fields = {
        "bridge_id",
        "model",
        "contract_id",
        "hook_name",
        "priority",
        "bridge_status",
        "guarded",
        "current_forward_status",
        "current_training_status",
        "train_run_dir",
        "evidence_manifest",
        "current_training_columns",
        "current_forward_artifacts",
        "proposed_epoch_columns",
        "proposed_epoch_artifacts",
        "validation_metrics",
        "producer_change",
        "consumer_surfaces",
        "compatibility_rule",
        "pass_gate",
        "fail_action",
        "next_migration_step",
        "current_metric_link",
        "safe_claim",
        "blocked_claim",
        "interview_line",
    }
    for row in rows:
        if not required_fields.issubset(row):
            raise AssertionError(f"Malformed deep model hook training bridge row: {row}")
        if row.get("bridge_status") != "bridge_ready" or not row.get("guarded"):
            raise AssertionError(f"Hook training bridge must be guarded and bridge_ready: {row}")
        if row.get("priority") not in {"P0", "P1", "P2", "P3"}:
            raise AssertionError(f"Hook training bridge priority out of range: {row}")
        if not row.get("proposed_epoch_columns") or not row.get("proposed_epoch_artifacts"):
            raise AssertionError(f"Hook training bridge row missing proposed epoch telemetry: {row}")
        if not Path(str(row.get("train_run_dir"))).exists():
            raise AssertionError(f"Hook training bridge training run dir is missing: {row}")
        if not Path(str(row.get("evidence_manifest"))).exists():
            raise AssertionError(f"Hook training bridge evidence manifest is missing: {row}")
    if int(summary.get("bridges") or 0) != len(rows):
        raise AssertionError(f"Deep model hook training bridge summary count mismatch: {path}")
    if int(summary.get("bridge_ready") or 0) < 8:
        raise AssertionError(f"Deep model hook training bridge should mark eight bridge-ready contracts: {path}")
    if int(summary.get("p0_bridges") or 0) < 5:
        raise AssertionError(f"Deep model hook training bridge should expose several P0 bridges: {path}")
    csv_path = optional_latest("deep_model_hook_training_bridge_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Deep model hook training bridge CSV is missing")
    doc_path = Path("docs/DEEPUplift_DEEP_MODEL_HOOK_TRAINING_BRIDGE.md")
    if not doc_path.exists():
        raise AssertionError("Deep model hook training bridge markdown doc is missing")
    return {
        "artifact": "deep_model_hook_training_bridge",
        "path": str(path),
        "models": summary.get("models"),
        "bridges": len(rows),
        "bridge_ready": summary.get("bridge_ready"),
        "p0_bridges": summary.get("p0_bridges"),
        "epoch_columns": summary.get("proposed_epoch_columns"),
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_deep_model_trainer_collector_smoke(root: Path) -> dict:
    path = optional_latest("deep_model_trainer_collector_smoke_latest.json", root)
    if path is None:
        return {"artifact": "deep_model_trainer_collector_smoke", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    model_rows = payload.get("model_rows") or []
    epoch_rows = payload.get("epoch_rows") or []
    summary = payload.get("summary") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Deep model trainer collector smoke is not ok: {path}")
    required_models = {"CFRNet", "DragonNet", "EFIN", "DESCN"}
    present_models = {row.get("model") for row in model_rows}
    missing_models = required_models - present_models
    if missing_models:
        raise AssertionError(f"Deep model trainer collector smoke missing models {sorted(missing_models)}: {path}")
    required_contracts = {
        "CFRNet.phi_x_representation",
        "DragonNet.e_hat_propensity",
        "DragonNet.targeted_regularization_components",
        "EFIN.interaction_attention",
        "EFIN.uplift_path_contribution",
        "DESCN.propensity_and_exposure",
        "DESCN.entire_space_heads",
        "DESCN.cross_head_constraints",
    }
    covered_contracts = {contract for row in model_rows for contract in row.get("contracts_covered") or []}
    missing_contracts = required_contracts - covered_contracts
    if missing_contracts:
        raise AssertionError(f"Deep model trainer collector smoke missing contracts {sorted(missing_contracts)}: {path}")
    required_fields = {
        "model",
        "status",
        "epochs",
        "contracts_covered",
        "collected_epoch_columns",
        "artifact_files",
        "metric_first",
        "metric_last",
        "metric_delta",
        "pass_gate",
        "fail_action",
        "safe_claim",
        "blocked_claim",
        "interview_line",
    }
    for row in model_rows:
        if not required_fields.issubset(row):
            raise AssertionError(f"Malformed deep model trainer collector row: {row}")
        if row.get("status") != "ok":
            raise AssertionError(f"Trainer collector model row did not pass: {row}")
        if int(row.get("epochs") or 0) < 2:
            raise AssertionError(f"Trainer collector should include at least two epochs per model: {row}")
        if not row.get("collected_epoch_columns") or not row.get("contracts_covered"):
            raise AssertionError(f"Trainer collector row missing columns/contracts: {row}")
        for artifact in row.get("artifact_files") or []:
            artifact_path = Path(str(artifact))
            if not artifact_path.exists():
                raise AssertionError(f"Trainer collector artifact file is missing: {artifact_path}")
    if int(summary.get("ok_models") or 0) < 4:
        raise AssertionError(f"Deep model trainer collector smoke should pass four models: {path}")
    if int(summary.get("epoch_rows") or 0) < 8 or len(epoch_rows) < 8:
        raise AssertionError(f"Deep model trainer collector smoke should include per-epoch rows: {path}")
    if int(summary.get("contracts_covered") or 0) < 8:
        raise AssertionError(f"Deep model trainer collector smoke should cover eight contracts: {path}")
    model_csv = optional_latest("deep_model_trainer_collector_models_latest.csv", root)
    epoch_csv = optional_latest("deep_model_trainer_collector_epochs_latest.csv", root)
    if model_csv is None or epoch_csv is None:
        raise AssertionError("Deep model trainer collector CSV artifacts are missing")
    doc_path = Path("docs/DEEPUplift_DEEP_MODEL_TRAINER_COLLECTOR_SMOKE.md")
    if not doc_path.exists():
        raise AssertionError("Deep model trainer collector markdown doc is missing")
    return {
        "artifact": "deep_model_trainer_collector_smoke",
        "path": str(path),
        "models": summary.get("models"),
        "ok_models": summary.get("ok_models"),
        "epoch_rows": summary.get("epoch_rows"),
        "contracts_covered": summary.get("contracts_covered"),
        "epoch_columns": summary.get("collected_epoch_columns"),
        "model_csv": str(model_csv),
        "epoch_csv": str(epoch_csv),
        "doc": str(doc_path),
    }


def validate_deep_model_telemetry_benchmark_linkage(root: Path) -> dict:
    path = optional_latest("deep_model_telemetry_benchmark_linkage_latest.json", root)
    if path is None:
        return {"artifact": "deep_model_telemetry_benchmark_linkage", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("rows") or []
    summary = payload.get("summary") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Deep model telemetry-benchmark linkage is not ok: {path}")
    required_models = {"CFRNet", "DragonNet", "EFIN", "DESCN"}
    present_models = {row.get("model") for row in rows}
    missing_models = required_models - present_models
    if missing_models:
        raise AssertionError(f"Deep model telemetry-benchmark linkage missing models {sorted(missing_models)}: {path}")
    required_contracts = {
        "CFRNet.phi_x_representation",
        "DragonNet.e_hat_propensity",
        "DragonNet.targeted_regularization_components",
        "EFIN.interaction_attention",
        "EFIN.uplift_path_contribution",
        "DESCN.propensity_and_exposure",
        "DESCN.entire_space_heads",
        "DESCN.cross_head_constraints",
    }
    present_contracts = {row.get("contract_id") for row in rows}
    missing_contracts = required_contracts - present_contracts
    if missing_contracts:
        raise AssertionError(f"Deep model telemetry-benchmark linkage missing contracts {sorted(missing_contracts)}: {path}")
    required_fields = {
        "linkage_id",
        "model",
        "contract_id",
        "telemetry_family",
        "linkage_status",
        "evidence_level",
        "guarded",
        "collector_status",
        "bridge_status",
        "telemetry_columns",
        "telemetry_last",
        "benchmark_question",
        "failure_hypothesis",
        "next_ablation",
        "paper_verdict",
        "tuned_verdict",
        "tuned_baseline_verdict",
        "safe_claim",
        "blocked_claim",
        "interview_line",
    }
    for row in rows:
        if not required_fields.issubset(row):
            raise AssertionError(f"Malformed deep model telemetry-benchmark linkage row: {row}")
        if row.get("linkage_status") != "link_ready" or not row.get("guarded"):
            raise AssertionError(f"Telemetry-benchmark linkage row must be guarded and link_ready: {row}")
        if not row.get("telemetry_columns") or not row.get("telemetry_last"):
            raise AssertionError(f"Telemetry-benchmark linkage row missing telemetry values: {row}")
        if not row.get("next_ablation") or not row.get("blocked_claim"):
            raise AssertionError(f"Telemetry-benchmark linkage row missing ablation or claim boundary: {row}")
    if int(summary.get("linkage_rows") or 0) != len(rows):
        raise AssertionError(f"Deep model telemetry-benchmark linkage summary count mismatch: {path}")
    if int(summary.get("link_ready") or 0) < 8:
        raise AssertionError(f"Deep model telemetry-benchmark linkage should expose eight link-ready rows: {path}")
    csv_path = optional_latest("deep_model_telemetry_benchmark_linkage_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Deep model telemetry-benchmark linkage CSV is missing")
    doc_path = Path("docs/DEEPUplift_DEEP_MODEL_TELEMETRY_BENCHMARK_LINKAGE.md")
    if not doc_path.exists():
        raise AssertionError("Deep model telemetry-benchmark linkage markdown doc is missing")
    return {
        "artifact": "deep_model_telemetry_benchmark_linkage",
        "path": str(path),
        "models": summary.get("models"),
        "linkage_rows": len(rows),
        "link_ready": summary.get("link_ready"),
        "families": len(summary.get("telemetry_families") or {}),
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_deep_model_telemetry_ablation_gate(root: Path) -> dict:
    path = optional_latest("deep_model_telemetry_ablation_gate_latest.json", root)
    if path is None:
        return {"artifact": "deep_model_telemetry_ablation_gate", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("rows") or []
    summary = payload.get("summary") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Deep model telemetry ablation gate is not ok: {path}")
    required_models = {"CFRNet", "DragonNet", "EFIN", "DESCN"}
    present_models = {row.get("model") for row in rows}
    missing_models = required_models - present_models
    if missing_models:
        raise AssertionError(f"Deep model telemetry ablation gate missing models {sorted(missing_models)}: {path}")
    required_contracts = {
        "CFRNet.phi_x_representation",
        "DragonNet.e_hat_propensity",
        "DragonNet.targeted_regularization_components",
        "EFIN.interaction_attention",
        "EFIN.uplift_path_contribution",
        "DESCN.propensity_and_exposure",
        "DESCN.entire_space_heads",
        "DESCN.cross_head_constraints",
    }
    present_contracts = {row.get("contract_id") for row in rows}
    missing_contracts = required_contracts - present_contracts
    if missing_contracts:
        raise AssertionError(f"Deep model telemetry ablation gate missing contracts {sorted(missing_contracts)}: {path}")
    required_fields = {
        "gate_id",
        "model",
        "contract_id",
        "telemetry_family",
        "ablation_family",
        "gate_decision",
        "variant_coverage_status",
        "available_variants",
        "required_variants",
        "missing_variants",
        "command",
        "metrics_to_watch",
        "pass_gate",
        "fail_action",
        "safe_claim",
        "blocked_claim",
        "interview_line",
        "evidence_refs",
    }
    allowed_decisions = {
        "blocked_until_variant_exists",
        "blocked_until_ablation_beats_baseline",
        "review_candidate_after_ablation",
        "review_required",
    }
    for row in rows:
        if not required_fields.issubset(row):
            raise AssertionError(f"Malformed deep model telemetry ablation gate row: {row}")
        if row.get("gate_decision") not in allowed_decisions:
            raise AssertionError(f"Unknown ablation gate decision: {row}")
        if not row.get("required_variants") or not row.get("metrics_to_watch"):
            raise AssertionError(f"Ablation gate row missing variants or metrics: {row}")
        if not row.get("command") or not row.get("pass_gate") or not row.get("fail_action"):
            raise AssertionError(f"Ablation gate row missing executable contract: {row}")
        if not row.get("safe_claim") or not row.get("blocked_claim"):
            raise AssertionError(f"Ablation gate row missing claim boundary: {row}")
    if int(summary.get("gates") or 0) != len(rows) or len(rows) < 8:
        raise AssertionError(f"Deep model telemetry ablation gate should expose eight gates: {path}")
    if int(summary.get("missing_variant_rows") or 0) < 0:
        raise AssertionError(f"Deep model telemetry ablation gate summary is malformed: {path}")
    csv_path = optional_latest("deep_model_telemetry_ablation_gate_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Deep model telemetry ablation gate CSV is missing")
    doc_path = Path("docs/DEEPUplift_DEEP_MODEL_TELEMETRY_ABLATION_GATE.md")
    if not doc_path.exists():
        raise AssertionError("Deep model telemetry ablation gate markdown doc is missing")
    return {
        "artifact": "deep_model_telemetry_ablation_gate",
        "path": str(path),
        "models": summary.get("models"),
        "gates": len(rows),
        "missing_variant_rows": summary.get("missing_variant_rows"),
        "decision_counts": summary.get("decision_counts"),
        "coverage_counts": summary.get("coverage_counts"),
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_paper_benchmark(root: Path) -> dict:
    path = optional_latest("paper_benchmark_latest.json", root)
    if path is None:
        return {"artifact": "paper_benchmark", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    runs = payload.get("runs") or []
    leaderboard = payload.get("leaderboard") or []
    best = payload.get("best_by_dataset") or []
    if payload.get("status") != "ok" or payload.get("failures"):
        raise AssertionError(f"Paper-level benchmark failed: {path}")
    if len(runs) < 12:
        raise AssertionError(f"Paper-level benchmark should contain at least 12 run attempts: {path}")
    ok_rows = [row for row in runs if row.get("status") == "ok"]
    if len(ok_rows) < 12:
        raise AssertionError(f"Paper-level benchmark should contain at least 12 successful runs: {path}")
    required_datasets = {"ihdp_npci_1", "acic_style_synthetic_known_cate_1k6", "synthetic_known_cate_shift_1k8"}
    present_datasets = {row.get("dataset_id") for row in runs}
    missing_datasets = required_datasets - present_datasets
    if missing_datasets:
        raise AssertionError(f"Paper-level benchmark missing datasets {sorted(missing_datasets)}: {path}")
    required_models = {"TLearnerGBM", "DRLearnerGBM", "CFRNet", "DragonNet", "EFIN", "DESCN"}
    attempted_models = {row.get("model") for row in runs}
    missing_models = required_models - attempted_models
    if missing_models:
        raise AssertionError(f"Paper-level benchmark missing model attempts {sorted(missing_models)}: {path}")
    ok_models = {row.get("model") for row in ok_rows}
    if len(required_models & ok_models) < 5:
        raise AssertionError(f"Paper-level benchmark should have at least five successful required models: {path}")
    if not leaderboard or not best:
        raise AssertionError(f"Paper-level benchmark missing leaderboard or best_by_dataset rows: {path}")
    if not payload.get("rankings") or not payload.get("stability_summary") or not payload.get("metric_glossary"):
        raise AssertionError(f"Paper-level benchmark missing rankings, stability summary or metric glossary: {path}")
    if not payload.get("failure_attribution"):
        raise AssertionError(f"Paper-level benchmark missing failure attribution: {path}")
    if not payload.get("run_diff"):
        raise AssertionError(f"Paper-level benchmark missing run diff: {path}")
    for row in ok_rows:
        if row.get("pehe") is None or row.get("ate_error") is None:
            raise AssertionError(f"Paper-level benchmark run missing PEHE/ATE error: {row}")
        if row.get("qini") is None or row.get("auuc") is None:
            raise AssertionError(f"Paper-level benchmark run missing QINI/AUUC: {row}")
        manifest = row.get("evidence_manifest")
        if not manifest or not Path(manifest).exists():
            raise AssertionError(f"Paper-level benchmark row missing evidence manifest: {row}")
    leaderboard_path = optional_latest("paper_benchmark_leaderboard_latest.csv", root)
    diff_path = optional_latest("paper_benchmark_run_diff_latest.json", root)
    manifest_path = optional_latest("paper_benchmark_manifest_latest.json", root)
    if leaderboard_path is None or diff_path is None or manifest_path is None:
        raise AssertionError("Paper-level benchmark latest CSV/diff/manifest artifacts are missing")
    manifest_payload = read_json(manifest_path)
    if manifest_payload.get("artifact") != "paper_level_benchmark" or not manifest_payload.get("run_evidence_manifests"):
        raise AssertionError(f"Paper-level benchmark manifest is incomplete: {manifest_path}")
    return {
        "artifact": "paper_benchmark",
        "path": str(path),
        "runs": len(runs),
        "ok_runs": len(ok_rows),
        "leaderboard": len(leaderboard),
        "best_by_dataset": len(best),
        "leaderboard_csv": str(leaderboard_path),
        "run_diff": str(diff_path),
        "manifest": str(manifest_path),
    }


def validate_deep_model_ablation_interpretation(root: Path) -> dict:
    path = optional_latest("deep_model_ablation_interpretation_latest.json", root)
    if path is None:
        return {"artifact": "deep_model_ablation_interpretation", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("rows") or []
    scorecards = payload.get("variant_scorecards") or []
    summary = payload.get("summary") or {}
    tracks = payload.get("interview_tracks") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Deep model ablation interpretation is not ok: {path}")
    if len(rows) < 8 or len(scorecards) < 6:
        raise AssertionError(f"Deep model ablation interpretation should cover runnable ablations: {path}")
    required_variants = {
        "CFRNet_low_balance",
        "CFRNet_high_balance",
        "DragonNet_no_tarreg",
        "DragonNet_stronger_tarreg",
        "EFIN_wide",
        "EFIN_no_attention",
        "EFIN_path_regularized",
        "DESCN_wide_dropout",
        "DESCN_no_constraint",
    }
    present_variants = {row.get("variant_id") for row in rows}
    missing = required_variants - present_variants
    if missing:
        raise AssertionError(f"Deep model ablation interpretation missing variants {sorted(missing)}: {path}")
    required_fields = {
        "dataset_id",
        "model",
        "variant_id",
        "default_variant",
        "mechanism",
        "hypothesis",
        "verdict",
        "metric_movements",
        "safe_claim",
        "blocked_claim",
        "interview_line",
        "evidence_refs",
    }
    for row in rows:
        if not required_fields.issubset(row):
            raise AssertionError(f"Malformed deep model ablation interpretation row: {row}")
        if not row.get("safe_claim") or not row.get("blocked_claim") or not row.get("interview_line"):
            raise AssertionError(f"Deep model ablation interpretation row missing claim text: {row}")
    for required in ["30s", "5min", "15min", "30min"]:
        if not tracks.get(required):
            raise AssertionError(f"Deep model ablation interpretation missing {required} talk track: {path}")
    if int(summary.get("rows") or 0) != len(rows):
        raise AssertionError(f"Deep model ablation interpretation summary row count mismatch: {path}")
    csv_path = optional_latest("deep_model_ablation_interpretation_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Deep model ablation interpretation CSV is missing")
    doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_INTERPRETATION.md")
    if not doc_path.exists():
        raise AssertionError("Deep model ablation interpretation markdown doc is missing")
    return {
        "artifact": "deep_model_ablation_interpretation",
        "path": str(path),
        "rows": len(rows),
        "variants": summary.get("variants"),
        "support_rows": summary.get("support_rows"),
        "tradeoff_rows": summary.get("tradeoff_rows"),
        "blocked_rows": summary.get("blocked_rows"),
        "verdict_counts": summary.get("verdict_counts"),
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_deep_model_ablation_promotion_matrix(root: Path) -> dict:
    path = optional_latest("deep_model_ablation_promotion_matrix_latest.json", root)
    if path is None:
        return {"artifact": "deep_model_ablation_promotion_matrix", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("rows") or []
    summary = payload.get("summary") or {}
    tracks = payload.get("interview_tracks") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Deep model ablation promotion matrix is not ok: {path}")
    if len(rows) < 9:
        raise AssertionError(f"Deep model ablation promotion matrix should cover all runnable ablation variants: {path}")
    required_variants = {
        "CFRNet_low_balance",
        "CFRNet_high_balance",
        "DragonNet_no_tarreg",
        "DragonNet_stronger_tarreg",
        "EFIN_wide",
        "EFIN_no_attention",
        "EFIN_path_regularized",
        "DESCN_wide_dropout",
        "DESCN_no_constraint",
    }
    present_variants = {row.get("variant_id") for row in rows}
    missing = required_variants - present_variants
    if missing:
        raise AssertionError(f"Deep model ablation promotion matrix missing variants {sorted(missing)}: {path}")
    required_fields = {
        "model",
        "variant_id",
        "promotion_decision",
        "claim_tier",
        "promotion_score",
        "recommended_claim_boundary",
        "safe_claim",
        "blocked_claim",
        "failure_attribution",
        "interview_line",
        "evidence_refs",
    }
    for row in rows:
        if not required_fields.issubset(row):
            raise AssertionError(f"Malformed deep model ablation promotion matrix row: {row}")
        if not row.get("safe_claim") or not row.get("blocked_claim") or not row.get("interview_line"):
            raise AssertionError(f"Deep model ablation promotion row missing claim text: {row}")
    for required in ["30s", "5min", "15min", "30min"]:
        if not tracks.get(required):
            raise AssertionError(f"Deep model ablation promotion matrix missing {required} talk track: {path}")
    if int(summary.get("rows") or 0) != len(rows):
        raise AssertionError(f"Deep model ablation promotion matrix summary row count mismatch: {path}")
    if int(summary.get("promotion_ready_rows") or 0) < 1:
        raise AssertionError(f"Deep model ablation promotion matrix should contain at least one local mechanism candidate: {path}")
    if int(summary.get("tradeoff_rows") or 0) < 1:
        raise AssertionError(f"Deep model ablation promotion matrix should contain tradeoff evidence: {path}")
    csv_path = optional_latest("deep_model_ablation_promotion_matrix_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Deep model ablation promotion matrix CSV is missing")
    doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_PROMOTION_MATRIX.md")
    if not doc_path.exists():
        raise AssertionError("Deep model ablation promotion matrix markdown doc is missing")
    return {
        "artifact": "deep_model_ablation_promotion_matrix",
        "path": str(path),
        "rows": len(rows),
        "variants": summary.get("variants"),
        "promotion_ready_rows": summary.get("promotion_ready_rows"),
        "tradeoff_rows": summary.get("tradeoff_rows"),
        "blocked_or_missing_rows": summary.get("blocked_or_missing_rows"),
        "decision_counts": summary.get("decision_counts"),
        "claim_tier_counts": summary.get("claim_tier_counts"),
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_deep_model_ablation_next_experiment_plan(root: Path) -> dict:
    path = optional_latest("deep_model_ablation_next_experiment_plan_latest.json", root)
    if path is None:
        return {"artifact": "deep_model_ablation_next_experiment_plan", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("rows") or []
    summary = payload.get("summary") or {}
    tracks = payload.get("interview_tracks") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Deep model ablation next-experiment plan is not ok: {path}")
    if len(rows) < 9:
        raise AssertionError(f"Deep model ablation next-experiment plan should cover all ablation variants: {path}")
    required_fields = {
        "model",
        "variant_id",
        "priority",
        "experiment_type",
        "hypothesis",
        "recommended_command",
        "runnable_fallback_command",
        "command_contract_status",
        "planner_unsupported_flags",
        "fallback_supported_flags",
        "fallback_unsupported_flags",
        "fallback_expected_scope",
        "metrics_to_watch",
        "pass_gate",
        "fail_action",
        "owner_role",
        "expected_artifacts",
        "safe_interview_line",
        "source_refs",
    }
    for row in rows:
        if not required_fields.issubset(row):
            raise AssertionError(f"Malformed deep model ablation next-experiment row: {row}")
        if not row.get("recommended_command") or "run_deep_model_tuned_benchmark.py" not in row.get("recommended_command", ""):
            raise AssertionError(f"Deep model ablation next-experiment row missing runnable command: {row}")
        if not row.get("runnable_fallback_command") or "run_deep_model_tuned_benchmark.py" not in row.get("runnable_fallback_command", ""):
            raise AssertionError(f"Deep model ablation next-experiment row missing fallback command: {row}")
        if row.get("fallback_unsupported_flags"):
            raise AssertionError(f"Deep model ablation next-experiment fallback command has unsupported flags: {row}")
        if not row.get("metrics_to_watch") or not row.get("pass_gate") or not row.get("fail_action"):
            raise AssertionError(f"Deep model ablation next-experiment row missing gates: {row}")
    for required in ["30s", "5min", "15min", "30min"]:
        if not tracks.get(required):
            raise AssertionError(f"Deep model ablation next-experiment plan missing {required} talk track: {path}")
    if int(summary.get("rows") or 0) != len(rows):
        raise AssertionError(f"Deep model ablation next-experiment summary row count mismatch: {path}")
    if int(summary.get("p0_rows") or 0) < 1 or int(summary.get("p1_rows") or 0) < 1:
        raise AssertionError(f"Deep model ablation next-experiment plan should include P0 and P1 rows: {path}")
    if int(summary.get("fallback_ready_rows") or 0) != len(rows):
        raise AssertionError(f"Deep model ablation next-experiment plan should have a fallback command for every row: {path}")
    csv_path = optional_latest("deep_model_ablation_next_experiment_plan_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Deep model ablation next-experiment CSV is missing")
    doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_NEXT_EXPERIMENT_PLAN.md")
    if not doc_path.exists():
        raise AssertionError("Deep model ablation next-experiment markdown doc is missing")
    return {
        "artifact": "deep_model_ablation_next_experiment_plan",
        "path": str(path),
        "rows": len(rows),
        "variants": summary.get("variants"),
        "p0_rows": summary.get("p0_rows"),
        "p1_rows": summary.get("p1_rows"),
        "fallback_ready_rows": summary.get("fallback_ready_rows"),
        "planner_adapter_rows": summary.get("planner_adapter_rows"),
        "command_status_counts": summary.get("command_status_counts"),
        "priority_counts": summary.get("priority_counts"),
        "experiment_type_counts": summary.get("experiment_type_counts"),
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_deep_model_ablation_command_contract(root: Path) -> dict:
    path = optional_latest("deep_model_ablation_command_contract_latest.json", root)
    if path is None:
        return {"artifact": "deep_model_ablation_command_contract", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("rows") or []
    summary = payload.get("summary") or {}
    tracks = payload.get("interview_tracks") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Deep model ablation command contract is not ok: {path}")
    if len(rows) < 9:
        raise AssertionError(f"Deep model ablation command contract should cover all ablation variants: {path}")
    required_fields = {
        "model",
        "variant_id",
        "priority",
        "experiment_type",
        "contract_status",
        "fallback_ready",
        "recommended_command",
        "runnable_fallback_command",
        "recommended_unsupported_flags",
        "fallback_unsupported_flags",
        "contract_interview_line",
        "source_refs",
    }
    for row in rows:
        if not required_fields.issubset(row):
            raise AssertionError(f"Malformed deep model ablation command contract row: {row}")
        if not row.get("fallback_ready"):
            raise AssertionError(f"Deep model ablation command contract fallback is not ready: {row}")
        if row.get("fallback_unsupported_flags"):
            raise AssertionError(f"Deep model ablation command contract fallback has unsupported flags: {row}")
        if "run_deep_model_tuned_benchmark.py" not in row.get("runnable_fallback_command", ""):
            raise AssertionError(f"Deep model ablation command contract missing runner fallback: {row}")
    for required in ["30s", "5min", "15min", "30min"]:
        if not tracks.get(required):
            raise AssertionError(f"Deep model ablation command contract missing {required} talk track: {path}")
    if int(summary.get("rows") or 0) != len(rows):
        raise AssertionError(f"Deep model ablation command contract summary row count mismatch: {path}")
    if int(summary.get("fallback_ready_rows") or 0) != len(rows):
        raise AssertionError(f"Deep model ablation command contract should have all fallback rows ready: {path}")
    planner_adapter_rows = int(summary.get("planner_adapter_rows") or 0)
    unsupported_counts = summary.get("unsupported_flag_counts") or {}
    if planner_adapter_rows == 0 and unsupported_counts:
        raise AssertionError(f"Deep model ablation command contract has unsupported flags but no adapter rows: {path}")
    if planner_adapter_rows > 0 and not any(row.get("recommended_unsupported_flags") for row in rows):
        raise AssertionError(f"Deep model ablation command contract adapter summary does not match rows: {path}")
    csv_path = optional_latest("deep_model_ablation_command_contract_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Deep model ablation command contract CSV is missing")
    doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_CONTRACT.md")
    if not doc_path.exists():
        raise AssertionError("Deep model ablation command contract markdown doc is missing")
    return {
        "artifact": "deep_model_ablation_command_contract",
        "path": str(path),
        "rows": len(rows),
        "variants": summary.get("variants"),
        "fallback_ready_rows": summary.get("fallback_ready_rows"),
        "planner_adapter_rows": summary.get("planner_adapter_rows"),
        "status_counts": summary.get("status_counts"),
        "unsupported_flag_counts": summary.get("unsupported_flag_counts"),
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_deep_model_ablation_command_contract_smoke(root: Path) -> dict:
    path = optional_latest("deep_model_ablation_command_contract_smoke_latest.json", root)
    if path is None:
        return {"artifact": "deep_model_ablation_command_contract_smoke", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("rows") or []
    summary = payload.get("summary") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Deep model ablation command contract smoke is not ok: {path}")
    if not rows:
        raise AssertionError(f"Deep model ablation command contract smoke should execute at least one fallback command: {path}")
    for row in rows:
        if row.get("status") != "ok":
            raise AssertionError(f"Deep model ablation command contract smoke row failed: {row}")
        command = str(row.get("command") or "")
        if "--no-update-latest" not in command or "--output-prefix deep_model_ablation_command_contract_smoke" not in command:
            raise AssertionError(f"Deep model ablation command contract smoke must use isolated runner output: {row}")
        if not row.get("runner_manifest") or not row.get("runner_json"):
            raise AssertionError(f"Deep model ablation command contract smoke row missing runner artifacts: {row}")
    if int(summary.get("ok_rows") or 0) != len(rows):
        raise AssertionError(f"Deep model ablation command contract smoke summary mismatch: {path}")
    csv_path = optional_latest("deep_model_ablation_command_contract_smoke_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Deep model ablation command contract smoke CSV is missing")
    doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_CONTRACT_SMOKE.md")
    if not doc_path.exists():
        raise AssertionError("Deep model ablation command contract smoke markdown doc is missing")
    return {
        "artifact": "deep_model_ablation_command_contract_smoke",
        "path": str(path),
        "rows": len(rows),
        "ok_rows": summary.get("ok_rows"),
        "sample_limit": summary.get("sample_limit"),
        "source_contract_rows": summary.get("source_contract_rows"),
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_deep_model_ablation_recommended_command_smoke(root: Path) -> dict:
    path = optional_latest("deep_model_ablation_recommended_command_smoke_latest.json", root)
    if path is None:
        return {"artifact": "deep_model_ablation_recommended_command_smoke", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("rows") or []
    summary = payload.get("summary") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Deep model ablation recommended command smoke is not ok: {path}")
    if not rows:
        raise AssertionError(f"Deep model ablation recommended command smoke should execute at least one row: {path}")
    for row in rows:
        if row.get("status") != "ok":
            raise AssertionError(f"Deep model ablation recommended command smoke row failed: {row}")
        recommended_command = str(row.get("recommended_command") or "")
        safe_command = str(row.get("safe_smoke_command") or "")
        if "--models" not in recommended_command or "--include-variants" not in recommended_command:
            raise AssertionError(f"Recommended command smoke row does not exercise runner-native aliases: {row}")
        if "--no-update-latest" not in safe_command or "--output-prefix deep_model_ablation_recommended_command_smoke" not in safe_command:
            raise AssertionError(f"Recommended command smoke row must use isolated runner output: {row}")
        if not row.get("runner_manifest") or not row.get("runner_json"):
            raise AssertionError(f"Recommended command smoke row missing runner artifacts: {row}")
    if int(summary.get("ok_rows") or 0) != len(rows):
        raise AssertionError(f"Deep model ablation recommended command smoke ok row mismatch: {path}")
    if int(summary.get("models_alias_rows") or 0) != len(rows):
        raise AssertionError(f"Deep model ablation recommended command smoke should cover --models alias for every row: {path}")
    if int(summary.get("include_variants_alias_rows") or 0) != len(rows):
        raise AssertionError(f"Deep model ablation recommended command smoke should cover --include-variants alias for every row: {path}")
    if len(rows) >= 2 and int(summary.get("emit_battle_cards_alias_rows") or 0) < 1:
        raise AssertionError(f"Deep model ablation recommended command smoke should cover --emit-battle-cards alias: {path}")
    csv_path = optional_latest("deep_model_ablation_recommended_command_smoke_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Deep model ablation recommended command smoke CSV is missing")
    doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_RECOMMENDED_COMMAND_SMOKE.md")
    if not doc_path.exists():
        raise AssertionError("Deep model ablation recommended command smoke markdown doc is missing")
    return {
        "artifact": "deep_model_ablation_recommended_command_smoke",
        "path": str(path),
        "rows": len(rows),
        "ok_rows": summary.get("ok_rows"),
        "models_alias_rows": summary.get("models_alias_rows"),
        "include_variants_alias_rows": summary.get("include_variants_alias_rows"),
        "emit_battle_cards_alias_rows": summary.get("emit_battle_cards_alias_rows"),
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_deep_model_ablation_command_smoke_coverage(root: Path) -> dict:
    path = optional_latest("deep_model_ablation_command_smoke_coverage_latest.json", root)
    if path is None:
        return {"artifact": "deep_model_ablation_command_smoke_coverage", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("rows") or []
    summary = payload.get("summary") or {}
    tracks = payload.get("interview_tracks") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Deep model ablation command smoke coverage is not ok: {path}")
    if len(rows) < 9:
        raise AssertionError(f"Deep model ablation command smoke coverage should cover all contract rows: {path}")
    required_fields = {
        "model",
        "variant_id",
        "priority",
        "contract_status",
        "fallback_ready",
        "fallback_smoked",
        "recommended_smoked",
        "any_smoked",
        "coverage_status",
        "next_action",
        "source_refs",
    }
    for row in rows:
        if not required_fields.issubset(row):
            raise AssertionError(f"Malformed deep model ablation command smoke coverage row: {row}")
        if row.get("contract_status") == "runner_supported" and not row.get("fallback_ready"):
            raise AssertionError(f"Runner-supported row should retain fallback-ready contract: {row}")
    for required in ["30s", "5min", "15min", "30min"]:
        if not tracks.get(required):
            raise AssertionError(f"Deep model ablation command smoke coverage missing {required} talk track: {path}")
    contract_rows = int(summary.get("contract_rows") or 0)
    any_smoked_rows = int(summary.get("any_smoked_rows") or 0)
    recommended_smoked_rows = int(summary.get("recommended_smoked_rows") or 0)
    fallback_smoked_rows = int(summary.get("fallback_smoked_rows") or 0)
    if contract_rows != len(rows):
        raise AssertionError(f"Deep model ablation command smoke coverage summary row mismatch: {path}")
    if any_smoked_rows != sum(1 for row in rows if row.get("any_smoked")):
        raise AssertionError(f"Deep model ablation command smoke coverage any-smoked mismatch: {path}")
    if recommended_smoked_rows < 2:
        raise AssertionError(f"Deep model ablation command smoke coverage should include at least two recommended-smoked rows: {path}")
    if fallback_smoked_rows < 1:
        raise AssertionError(f"Deep model ablation command smoke coverage should include at least one fallback-smoked row: {path}")
    coverage_gate = str(summary.get("coverage_gate") or "")
    has_unsmoked_backlog = any(row.get("coverage_status") == "contract_ready_not_smoked" for row in rows)
    if coverage_gate == "review_required" and not has_unsmoked_backlog:
        raise AssertionError(f"Coverage report has review_required gate but no unsmoked contract-ready backlog rows: {path}")
    if coverage_gate == "ok" and any(not row.get("any_smoked") for row in rows):
        raise AssertionError(f"Coverage report gate is ok but some rows are not smoke-covered: {path}")
    csv_path = optional_latest("deep_model_ablation_command_smoke_coverage_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Deep model ablation command smoke coverage CSV is missing")
    doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_SMOKE_COVERAGE.md")
    if not doc_path.exists():
        raise AssertionError("Deep model ablation command smoke coverage markdown doc is missing")
    return {
        "artifact": "deep_model_ablation_command_smoke_coverage",
        "path": str(path),
        "rows": len(rows),
        "any_smoked_rows": any_smoked_rows,
        "recommended_smoked_rows": recommended_smoked_rows,
        "fallback_smoked_rows": fallback_smoked_rows,
        "coverage_gate": coverage_gate,
        "coverage_status_counts": summary.get("coverage_status_counts"),
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_deep_model_ablation_command_smoke_coverage_diff(root: Path) -> dict:
    path = optional_latest("deep_model_ablation_command_smoke_coverage_diff_latest.json", root)
    if path is None:
        return {"artifact": "deep_model_ablation_command_smoke_coverage_diff", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("rows") or []
    summary = payload.get("summary") or {}
    tracks = payload.get("interview_tracks") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Deep model ablation command smoke coverage diff is not ok: {path}")
    if len(rows) < 9:
        raise AssertionError(f"Deep model ablation command smoke coverage diff should cover all contract rows: {path}")
    required_fields = {
        "model",
        "variant_id",
        "baseline_fallback_smoked",
        "current_any_smoked",
        "runner_native_recommended_smoked",
        "transition",
        "smoke_gain",
        "interview_claim",
        "next_action",
    }
    for row in rows:
        if not required_fields.issubset(row):
            raise AssertionError(f"Malformed deep model ablation command smoke coverage diff row: {row}")
    for required in ["30s", "5min", "15min", "30min"]:
        if not tracks.get(required):
            raise AssertionError(f"Deep model ablation command smoke coverage diff missing {required} talk track: {path}")
    contract_rows = int(summary.get("contract_rows") or 0)
    baseline_rows = int(summary.get("baseline_fallback_smoked_rows") or 0)
    current_rows = int(summary.get("current_any_smoked_rows") or 0)
    runner_native_rows = int(summary.get("runner_native_recommended_smoked_rows") or 0)
    smoke_gain_rows = int(summary.get("smoke_gain_rows") or 0)
    if contract_rows != len(rows):
        raise AssertionError(f"Coverage diff summary row mismatch: {path}")
    if baseline_rows != sum(1 for row in rows if row.get("baseline_fallback_smoked")):
        raise AssertionError(f"Coverage diff baseline row mismatch: {path}")
    if current_rows != sum(1 for row in rows if row.get("current_any_smoked")):
        raise AssertionError(f"Coverage diff current row mismatch: {path}")
    if runner_native_rows != sum(1 for row in rows if row.get("runner_native_recommended_smoked")):
        raise AssertionError(f"Coverage diff runner-native row mismatch: {path}")
    if smoke_gain_rows != sum(1 for row in rows if int(row.get("smoke_gain") or 0) > 0):
        raise AssertionError(f"Coverage diff smoke gain mismatch: {path}")
    if current_rows < baseline_rows:
        raise AssertionError(f"Coverage diff should not regress current smoke coverage below baseline: {path}")
    if str(summary.get("coverage_gate_after_runner_native") or "") == "ok" and current_rows != contract_rows:
        raise AssertionError(f"Coverage diff gate is ok but not all rows are currently smoked: {path}")
    csv_path = optional_latest("deep_model_ablation_command_smoke_coverage_diff_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Deep model ablation command smoke coverage diff CSV is missing")
    doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_SMOKE_COVERAGE_DIFF.md")
    if not doc_path.exists():
        raise AssertionError("Deep model ablation command smoke coverage diff markdown doc is missing")
    return {
        "artifact": "deep_model_ablation_command_smoke_coverage_diff",
        "path": str(path),
        "rows": len(rows),
        "baseline_fallback_smoked_rows": baseline_rows,
        "current_any_smoked_rows": current_rows,
        "runner_native_recommended_smoked_rows": runner_native_rows,
        "coverage_gate_delta": summary.get("coverage_gate_delta"),
        "transition_counts": summary.get("transition_counts"),
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_paper_benchmark_interpretation(root: Path) -> dict:
    path = optional_latest("paper_benchmark_interpretation_latest.json", root)
    if path is None:
        return {"artifact": "paper_benchmark_interpretation", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    scorecards = payload.get("model_scorecards") or []
    tracks = payload.get("interview_tracks") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Paper benchmark interpretation is not ok: {path}")
    if len(scorecards) < 5:
        raise AssertionError(f"Paper benchmark interpretation should cover at least five models: {path}")
    for required in ["30s", "5min", "15min", "30min"]:
        if not tracks.get(required):
            raise AssertionError(f"Paper benchmark interpretation missing {required} talk track: {path}")
    if not payload.get("dataset_winners") or not payload.get("stability_reading") or not payload.get("recommendations"):
        raise AssertionError(f"Paper benchmark interpretation is missing review sections: {path}")
    doc_path = Path("docs/DEEPUplift_BENCHMARK_V3_INTERPRETATION.md")
    if not doc_path.exists():
        raise AssertionError("Paper benchmark interpretation markdown doc is missing")
    return {
        "artifact": "paper_benchmark_interpretation",
        "path": str(path),
        "scorecards": len(scorecards),
        "metric_conflicts": len(payload.get("metric_conflicts") or []),
        "doc": str(doc_path),
    }


def validate_glossary(root: Path) -> dict:
    path = optional_latest("glossary_latest.json", root)
    if path is None:
        return {"artifact": "glossary", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    terms = payload.get("terms") or []
    contexts = payload.get("contexts") or []
    if payload.get("status") != "ok" or len(terms) < 20:
        raise AssertionError(f"Glossary artifact is incomplete: {path}")
    required = {"PEHE", "CATE", "QINI", "AUUC", "OPE", "CFRNet", "DragonNet", "evidence manifest"}
    present = {row.get("term") for row in terms}
    missing = required - present
    if missing:
        raise AssertionError(f"Glossary missing required terms {sorted(missing)}: {path}")
    doc_path = Path("docs/DEEPUplift_GLOSSARY.md")
    if not doc_path.exists():
        raise AssertionError("Glossary markdown doc is missing")
    return {"artifact": "glossary", "path": str(path), "terms": len(terms), "contexts": len(contexts), "doc": str(doc_path)}


def validate_no_ui_compare(root: Path) -> dict:
    path = latest("no_ui_compare_manifest_*.json", root)
    payload = read_json(path)
    ranking = payload.get("ranking") or []
    runs = payload.get("runs") or []
    model_cards = payload.get("model_cards") or []
    if len(ranking) < 2:
        raise AssertionError(f"No-UI compare ranking should contain at least two rows: {path}")
    if not runs:
        raise AssertionError(f"No-UI compare should include run artifacts: {path}")
    if len(model_cards) < len(payload.get("models") or []):
        raise AssertionError(f"No-UI compare should include model cards for all requested models: {path}")
    for card in model_cards:
        if not card.get("model") or not card.get("parameter_presets") or not card.get("decision_use"):
            raise AssertionError(f"Malformed model card in no-UI compare manifest: {card}")
    for row in ranking:
        if not row.get("model") or (not row.get("error") and not row.get("run_id")):
            raise AssertionError(f"Malformed compare ranking row: {row}")
    return {"artifact": "no_ui_compare", "path": str(path), "ranking": len(ranking), "runs": len(runs), "model_cards": len(model_cards)}


def validate_benchmark_suite(root: Path) -> dict:
    path = optional_latest("benchmark_suite_latest.json", root)
    if path is None:
        return {"artifact": "benchmark_suite", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    runs = payload.get("runs") or []
    best = payload.get("best_by_dataset") or []
    if len(runs) < 2:
        raise AssertionError(f"Benchmark suite should contain at least two run rows: {path}")
    ok_rows = [row for row in runs if row.get("status") == "ok"]
    if not ok_rows:
        raise AssertionError(f"Benchmark suite should contain at least one successful run: {path}")
    if not best:
        raise AssertionError(f"Benchmark suite should include best_by_dataset rows: {path}")
    leaderboard = payload.get("leaderboard") or []
    if not leaderboard:
        raise AssertionError(f"Benchmark suite should include leaderboard rows: {path}")
    if not payload.get("failure_attribution"):
        raise AssertionError(f"Benchmark suite should include failure attribution: {path}")
    if not payload.get("run_diff"):
        raise AssertionError(f"Benchmark suite should include run_diff: {path}")
    missing_manifest = [row for row in ok_rows if not row.get("evidence_manifest")]
    if missing_manifest:
        raise AssertionError(f"Benchmark suite rows are missing evidence manifests: {missing_manifest[:3]}")
    for row in ok_rows:
        manifest_path = Path(row["evidence_manifest"])
        if not manifest_path.exists():
            raise AssertionError(f"Benchmark suite evidence manifest does not exist: {manifest_path}")
    return {
        "artifact": "benchmark_suite",
        "path": str(path),
        "runs": len(runs),
        "ok_runs": len(ok_rows),
        "best_by_dataset": len(best),
        "leaderboard": len(leaderboard),
    }


def validate_dataset_registry(root: Path) -> dict:
    path = optional_latest("dataset_registry_latest.json", root)
    if path is None:
        return {"artifact": "dataset_registry", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    cards = payload.get("cards") or []
    if len(cards) < 5:
        raise AssertionError(f"Dataset registry should include several cards: {path}")
    failures = payload.get("failures") or []
    if failures:
        raise AssertionError(f"Dataset registry contains failures: {failures[:3]}")
    for card in cards[:5]:
        if not card.get("dataset_id") or not (card.get("data_profile") or {}).get("file_sha256"):
            raise AssertionError(f"Malformed dataset card in {path}: {card}")
    csv_path = optional_latest("dataset_registry_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Dataset registry CSV is missing")
    return {"artifact": "dataset_registry", "path": str(path), "cards": len(cards), "csv": str(csv_path)}


def validate_ope_engine(root: Path) -> dict:
    path = optional_latest("ope_engine_smoke_latest.json", root)
    if path is None:
        return {"artifact": "ope_engine", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    if payload.get("status") != "ok" or payload.get("failures"):
        raise AssertionError(f"OPE smoke failed: {path}")
    metrics = ((payload.get("ope") or {}).get("metrics") or {})
    diagnostics = ((payload.get("ope") or {}).get("diagnostics") or {})
    if metrics.get("ips") is None or metrics.get("snips") is None or metrics.get("dr") is None:
        raise AssertionError(f"OPE smoke missing IPS/SNIPS/DR metrics: {path}")
    if not diagnostics.get("effective_sample_size"):
        raise AssertionError(f"OPE smoke missing effective sample size: {path}")
    return {
        "artifact": "ope_engine",
        "path": str(path),
        "dr": metrics.get("dr"),
        "effective_sample_size": diagnostics.get("effective_sample_size"),
    }


def validate_nuisance_diagnostics(root: Path) -> dict:
    path = optional_latest("nuisance_diagnostics_smoke_latest.json", root)
    if path is None:
        return {"artifact": "nuisance_diagnostics", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    if payload.get("status") != "ok" or payload.get("failures"):
        raise AssertionError(f"Nuisance diagnostics smoke failed: {path}")
    results = payload.get("results") or []
    if len(results) < 2:
        raise AssertionError(f"Nuisance diagnostics smoke should include DR and R rows: {path}")
    for row in results:
        diagnostics_path = Path(row.get("nuisance_diagnostics") or "")
        if row.get("status") != "ok" or not diagnostics_path.exists():
            raise AssertionError(f"Nuisance diagnostics row is not valid: {row}")
        diagnostics = read_json(diagnostics_path)
        if not (diagnostics.get("cross_fit") or {}).get("enabled"):
            raise AssertionError(f"Nuisance diagnostics should be cross-fitted: {diagnostics_path}")
        if (diagnostics.get("propensity") or {}).get("ipw_ess") is None:
            raise AssertionError(f"Nuisance diagnostics missing ESS: {diagnostics_path}")
    return {"artifact": "nuisance_diagnostics", "path": str(path), "results": len(results)}


def validate_budget_policy_optimizer(root: Path) -> dict:
    path = optional_latest("budget_policy_optimizer_smoke_latest.json", root)
    if path is None:
        return {"artifact": "budget_policy_optimizer", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    if payload.get("status") != "ok" or payload.get("failures"):
        raise AssertionError(f"Budget policy optimizer smoke failed: {path}")
    result = payload.get("result") or {}
    if int(result.get("selected_rows") or 0) <= 0:
        raise AssertionError(f"Budget policy optimizer selected no rows: {path}")
    if (result.get("budget_used") or 0.0) > (payload.get("budget") or 0.0) + 1e-9:
        raise AssertionError(f"Budget policy optimizer exceeded budget: {path}")
    if (result.get("net_value") or 0.0) <= 0:
        raise AssertionError(f"Budget policy optimizer has non-positive net value: {path}")
    if not payload.get("sensitivity"):
        raise AssertionError(f"Budget policy optimizer missing sensitivity curve: {path}")
    return {
        "artifact": "budget_policy_optimizer",
        "path": str(path),
        "selected_rows": result.get("selected_rows"),
        "net_value": result.get("net_value"),
    }


def validate_multi_treatment_benchmark(root: Path) -> dict:
    path = optional_latest("multi_treatment_benchmark_latest.json", root)
    if path is None:
        return {"artifact": "multi_treatment_benchmark", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    if payload.get("status") != "ok" or payload.get("failures"):
        raise AssertionError(f"Multi-treatment benchmark failed: {path}")
    leaderboard = payload.get("leaderboard") or []
    if len(leaderboard) < 2:
        raise AssertionError(f"Multi-treatment benchmark should include at least two leaderboard rows: {path}")
    for row in leaderboard:
        if row.get("incremental_policy_value_ipw") is None or int(row.get("action_count") or 0) <= 2:
            raise AssertionError(f"Malformed multi-treatment leaderboard row: {row}")
        run_dir = Path(row.get("run_dir") or "")
        if not (run_dir / "metrics.json").exists() or not (run_dir / "predictions.csv").exists():
            raise AssertionError(f"Multi-treatment benchmark run artifacts missing: {run_dir}")
    csv_path = optional_latest("multi_treatment_benchmark_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Multi-treatment benchmark CSV is missing")
    return {"artifact": "multi_treatment_benchmark", "path": str(path), "leaderboard": len(leaderboard), "csv": str(csv_path)}


def validate_continuous_treatment_policy(root: Path) -> dict:
    path = optional_latest("continuous_treatment_policy_smoke_latest.json", root)
    if path is None:
        return {"artifact": "continuous_treatment_policy", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    if payload.get("status") != "ok" or payload.get("failures"):
        raise AssertionError(f"Continuous treatment policy smoke failed: {path}")
    if not payload.get("policy_rows") or not payload.get("best_policy"):
        raise AssertionError(f"Continuous treatment policy smoke missing policy rows: {path}")
    support = payload.get("support") or {}
    if support.get("unsafe_extrapolation_rate") is None:
        raise AssertionError(f"Continuous treatment policy smoke missing support diagnostics: {path}")
    csv_path = optional_latest("continuous_treatment_policy_rows_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Continuous treatment policy CSV is missing")
    return {"artifact": "continuous_treatment_policy", "path": str(path), "policy_rows": len(payload.get("policy_rows") or []), "csv": str(csv_path)}


def validate_evidence_store(root: Path) -> dict:
    path = optional_latest("evidence_store_latest.json", root)
    if path is None:
        return {"artifact": "evidence_store", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    records = payload.get("records") or []
    if not records:
        raise AssertionError(f"Evidence store has no records: {path}")
    summary = payload.get("summary") or {}
    if not summary.get("manifest_count"):
        raise AssertionError(f"Evidence store summary is incomplete: {path}")
    diff_path = optional_latest("evidence_run_diff_latest.json", root)
    if diff_path is None:
        raise AssertionError("Evidence run diff artifact is missing")
    return {
        "artifact": "evidence_store",
        "path": str(path),
        "records": len(records),
        "diff": str(diff_path),
    }


def validate_regression_warning_audit(root: Path) -> dict:
    path = optional_latest("regression_warning_audit_latest.json", root)
    if path is None:
        return {"artifact": "regression_warning_audit", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("warnings") or []
    summary = payload.get("summary") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Regression warning audit is not ok: {path}")
    if not isinstance(rows, list):
        raise AssertionError(f"Regression warning audit warnings should be a list: {path}")
    if int(summary.get("total_warnings") or 0) != len(rows):
        raise AssertionError(f"Regression warning audit summary count mismatch: {path}")
    required = {
        "severity",
        "domain",
        "artifact",
        "entity",
        "signal",
        "why_it_matters",
        "recommended_action",
        "promotion_decision",
        "owner_role",
        "next_experiment",
        "ui_route",
        "interview_priority",
        "interview_line",
    }
    for row in rows:
        if not required.issubset(row):
            raise AssertionError(f"Malformed regression warning row: {row}")
    csv_path = optional_latest("regression_warning_audit_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Regression warning audit CSV is missing")
    doc_path = Path("docs/DEEPUplift_REGRESSION_WARNING_AUDIT.md")
    if not doc_path.exists():
        raise AssertionError("Regression warning audit markdown doc is missing")
    return {
        "artifact": "regression_warning_audit",
        "path": str(path),
        "warnings": len(rows),
        "severity_counts": summary.get("severity_counts") or {},
        "launch_gate": summary.get("launch_gate"),
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_promotion_launch_cards(root: Path) -> dict:
    path = optional_latest("promotion_launch_cards_latest.json", root)
    if path is None:
        return {"artifact": "promotion_launch_cards", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    cards = payload.get("cards") or []
    summary = payload.get("summary") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Promotion launch cards artifact is not ok: {path}")
    if not isinstance(cards, list) or not cards:
        raise AssertionError(f"Promotion launch cards should contain card rows: {path}")
    required = {
        "card_id",
        "domain",
        "entity",
        "warning_count",
        "overall_decision",
        "promotion_stage",
        "owner_role",
        "primary_blocker",
        "safe_claim",
        "unsafe_claim",
        "before_shadow",
        "before_ab",
        "before_ramp",
        "next_experiments",
        "metrics_to_watch",
        "ui_routes",
        "evidence_refs",
        "interview_line",
    }
    for card in cards:
        if not required.issubset(card):
            raise AssertionError(f"Malformed promotion launch card: {card}")
        if card.get("overall_decision") not in {"block_promotion", "review_before_promotion", "monitor"}:
            raise AssertionError(f"Unknown promotion decision in card: {card}")
    decisions = summary.get("decision_counts") or {}
    if sum(int(value or 0) for value in decisions.values()) != len(cards):
        raise AssertionError(f"Promotion card decision count mismatch: {path}")
    csv_path = optional_latest("promotion_launch_cards_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Promotion launch cards CSV is missing")
    doc_path = Path("docs/DEEPUplift_PROMOTION_LAUNCH_CARDS.md")
    if not doc_path.exists():
        raise AssertionError("Promotion launch cards markdown doc is missing")
    return {
        "artifact": "promotion_launch_cards",
        "path": str(path),
        "cards": len(cards),
        "launch_gate": summary.get("launch_gate"),
        "decision_counts": decisions,
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_regression_warning_triage(root: Path) -> dict:
    path = optional_latest("regression_warning_triage_latest.json", root)
    if path is None:
        return {"artifact": "regression_warning_triage", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("rows") or []
    summary = payload.get("summary") or {}
    tracks = payload.get("interview_tracks") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Regression warning triage is not ok: {path}")
    if not isinstance(rows, list) or not rows:
        raise AssertionError(f"Regression warning triage should contain rows: {path}")
    required = {
        "severity",
        "domain",
        "entity",
        "signal",
        "risk_family",
        "triage_status",
        "claim_boundary",
        "action_lane",
        "smoke_only",
        "launch_blocking",
        "owner_role",
        "safe_claim",
        "unsafe_claim",
        "next_experiment",
    }
    for row in rows:
        if not required.issubset(row):
            raise AssertionError(f"Malformed regression warning triage row: {row}")
    for required_track in ["30s", "5min", "15min", "30min"]:
        if not tracks.get(required_track):
            raise AssertionError(f"Regression warning triage missing {required_track} talk track: {path}")
    warning_rows = int(summary.get("warning_rows") or 0)
    smoke_only_rows = int(summary.get("smoke_only_rows") or 0)
    launch_blocking_rows = int(summary.get("launch_blocking_rows") or 0)
    if warning_rows != len(rows):
        raise AssertionError(f"Regression warning triage summary count mismatch: {path}")
    if smoke_only_rows != sum(1 for row in rows if row.get("smoke_only")):
        raise AssertionError(f"Regression warning triage smoke-only count mismatch: {path}")
    if launch_blocking_rows != sum(1 for row in rows if row.get("launch_blocking")):
        raise AssertionError(f"Regression warning triage launch-blocking count mismatch: {path}")
    if launch_blocking_rows and summary.get("triage_gate") != "blocked_for_pilot":
        raise AssertionError(f"Regression warning triage gate should block pilot when launch blockers exist: {path}")
    csv_path = optional_latest("regression_warning_triage_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Regression warning triage CSV is missing")
    doc_path = Path("docs/DEEPUplift_REGRESSION_WARNING_TRIAGE.md")
    if not doc_path.exists():
        raise AssertionError("Regression warning triage markdown doc is missing")
    return {
        "artifact": "regression_warning_triage",
        "path": str(path),
        "warning_rows": len(rows),
        "smoke_only_rows": smoke_only_rows,
        "launch_blocking_rows": launch_blocking_rows,
        "triage_gate": summary.get("triage_gate"),
        "risk_family_counts": summary.get("risk_family_counts"),
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_pilot_readiness_scorecard(root: Path) -> dict:
    path = optional_latest("pilot_readiness_scorecard_latest.json", root)
    if path is None:
        return {"artifact": "pilot_readiness_scorecard", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    scorecards = payload.get("scorecards") or []
    summary = payload.get("summary") or {}
    tracks = payload.get("interview_tracks") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Pilot readiness scorecard is not ok: {path}")
    if not isinstance(scorecards, list) or not scorecards:
        raise AssertionError(f"Pilot readiness scorecard should contain rows: {path}")
    required = {
        "scorecard_id",
        "domain",
        "entity",
        "pilot_gate",
        "readiness_score",
        "evidence_grade",
        "promotion_decision",
        "owner_role",
        "launch_blockers",
        "review_rows",
        "risk_families",
        "next_gate",
        "pilot_path",
        "next_experiments",
        "rollback_conditions",
        "required_artifacts",
        "safe_claim",
        "unsafe_claim",
        "demo_route",
        "interview_line",
    }
    allowed_gates = {"blocked", "paper_review_candidate", "shadow_candidate", "monitor"}
    for row in scorecards:
        if not required.issubset(row):
            raise AssertionError(f"Malformed pilot readiness scorecard row: {row}")
        if row.get("pilot_gate") not in allowed_gates:
            raise AssertionError(f"Unknown pilot gate in scorecard row: {row}")
        score_value = row.get("readiness_score")
        score = int(score_value if score_value is not None else -1)
        if score < 0 or score > 100:
            raise AssertionError(f"Pilot readiness score should be in [0, 100]: {row}")
    for required_track in ["30s", "5min", "15min", "30min"]:
        if not tracks.get(required_track):
            raise AssertionError(f"Pilot readiness scorecard missing {required_track} talk track: {path}")
    gate_counts = summary.get("gate_counts") or {}
    if sum(int(value or 0) for value in gate_counts.values()) != len(scorecards):
        raise AssertionError(f"Pilot readiness scorecard gate count mismatch: {path}")
    blocked = int(summary.get("blocked") or 0)
    if blocked and summary.get("pilot_gate") != "blocked":
        raise AssertionError(f"Pilot readiness scorecard should block when blocked rows exist: {path}")
    csv_path = optional_latest("pilot_readiness_scorecard_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Pilot readiness scorecard CSV is missing")
    doc_path = Path("docs/DEEPUplift_PILOT_READINESS_SCORECARD.md")
    if not doc_path.exists():
        raise AssertionError("Pilot readiness scorecard markdown doc is missing")
    return {
        "artifact": "pilot_readiness_scorecard",
        "path": str(path),
        "scorecards": len(scorecards),
        "pilot_gate": summary.get("pilot_gate"),
        "blocked": blocked,
        "paper_review_candidates": summary.get("paper_review_candidates"),
        "shadow_candidates": summary.get("shadow_candidates"),
        "monitor": summary.get("monitor"),
        "gate_counts": gate_counts,
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_pilot_experiment_plan(root: Path) -> dict:
    path = optional_latest("pilot_experiment_plan_latest.json", root)
    if path is None:
        return {"artifact": "pilot_experiment_plan", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("rows") or []
    summary = payload.get("summary") or {}
    tracks = payload.get("interview_tracks") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Pilot experiment plan is not ok: {path}")
    if not isinstance(rows, list) or not rows:
        raise AssertionError(f"Pilot experiment plan should contain rows: {path}")
    required = {
        "experiment_id",
        "scorecard_id",
        "domain",
        "entity",
        "pilot_gate",
        "readiness_score",
        "evidence_grade",
        "priority",
        "experiment_type",
        "owner_role",
        "hypothesis",
        "recommended_command",
        "refresh_command",
        "command_status",
        "metrics_to_watch",
        "expected_artifacts",
        "pass_gate",
        "fail_action",
        "claim_upgrade",
        "rollback_conditions",
        "demo_route",
        "interview_line",
    }
    allowed_priorities = {"P0", "P1", "P2", "P3"}
    allowed_types = {
        "blocker_repair_rerun",
        "paper_benchmark_or_ablation_review",
        "shadow_scoring_and_holdout_design",
        "evidence_refresh_and_monitoring",
    }
    allowed_gates = {"blocked", "paper_review_candidate", "shadow_candidate", "monitor"}
    for row in rows:
        if not required.issubset(row):
            raise AssertionError(f"Malformed pilot experiment plan row: {row}")
        if row.get("priority") not in allowed_priorities:
            raise AssertionError(f"Unknown priority in pilot experiment row: {row}")
        if row.get("experiment_type") not in allowed_types:
            raise AssertionError(f"Unknown experiment type in pilot experiment row: {row}")
        if row.get("pilot_gate") not in allowed_gates:
            raise AssertionError(f"Unknown pilot gate in pilot experiment row: {row}")
        if row.get("command_status") != "runner_supported_known_cli":
            raise AssertionError(f"Pilot experiment command must use a known CLI runner: {row}")
        if not row.get("recommended_command") or not row.get("pass_gate") or not row.get("fail_action"):
            raise AssertionError(f"Pilot experiment row is missing an executable decision contract: {row}")
        if not isinstance(row.get("metrics_to_watch"), list) or not row.get("metrics_to_watch"):
            raise AssertionError(f"Pilot experiment row should list metrics to watch: {row}")
        if not isinstance(row.get("expected_artifacts"), list) or not row.get("expected_artifacts"):
            raise AssertionError(f"Pilot experiment row should list expected artifacts: {row}")
    for required_track in ["30s", "5min", "15min", "30min"]:
        if not tracks.get(required_track):
            raise AssertionError(f"Pilot experiment plan missing {required_track} talk track: {path}")
    priority_counts = summary.get("priority_counts") or {}
    if sum(int(value or 0) for value in priority_counts.values()) != len(rows):
        raise AssertionError(f"Pilot experiment priority count mismatch: {path}")
    experiment_counts = summary.get("experiment_type_counts") or {}
    if sum(int(value or 0) for value in experiment_counts.values()) != len(rows):
        raise AssertionError(f"Pilot experiment type count mismatch: {path}")
    if int(summary.get("experiments") or 0) != len(rows):
        raise AssertionError(f"Pilot experiment summary row count mismatch: {path}")
    csv_path = optional_latest("pilot_experiment_plan_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Pilot experiment plan CSV is missing")
    doc_path = Path("docs/DEEPUplift_PILOT_EXPERIMENT_PLAN.md")
    if not doc_path.exists():
        raise AssertionError("Pilot experiment plan markdown doc is missing")
    return {
        "artifact": "pilot_experiment_plan",
        "path": str(path),
        "experiments": len(rows),
        "p0_rows": summary.get("p0_rows"),
        "p1_rows": summary.get("p1_rows"),
        "p2_rows": summary.get("p2_rows"),
        "experiment_type_counts": experiment_counts,
        "command_status_counts": summary.get("command_status_counts"),
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_pilot_experiment_command_smoke(root: Path) -> dict:
    path = optional_latest("pilot_experiment_command_smoke_latest.json", root)
    if path is None:
        return {"artifact": "pilot_experiment_command_smoke", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("rows") or []
    summary = payload.get("summary") or {}
    tracks = payload.get("interview_tracks") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Pilot experiment command smoke is not ok: {path}")
    if not isinstance(rows, list) or not rows:
        raise AssertionError(f"Pilot experiment command smoke should contain rows: {path}")
    required = {
        "experiment_id",
        "entity",
        "priority",
        "pilot_gate",
        "experiment_type",
        "status",
        "returncode",
        "elapsed_seconds",
        "recommended_command",
        "safe_smoke_command",
        "pass_gate",
        "fail_action",
        "claim_upgrade",
    }
    for row in rows:
        if not required.issubset(row):
            raise AssertionError(f"Malformed pilot experiment command smoke row: {row}")
        returncode_value = row.get("returncode")
        returncode = int(returncode_value if returncode_value is not None else -1)
        if row.get("status") != "ok" or returncode != 0:
            raise AssertionError(f"Pilot experiment command smoke row failed: {row}")
        if "--no-update-latest" not in str(row.get("safe_smoke_command") or "") and "generate_evidence_store.py" not in str(row.get("safe_smoke_command") or ""):
            raise AssertionError(f"Pilot experiment smoke command must avoid overwriting latest benchmark evidence: {row}")
        if not (row.get("runner_json") or row.get("runner_csv") or row.get("runner_manifest")):
            raise AssertionError(f"Pilot experiment smoke row should reference a runner artifact: {row}")
    for required_track in ["30s", "5min", "15min", "30min"]:
        if not tracks.get(required_track):
            raise AssertionError(f"Pilot experiment command smoke missing {required_track} talk track: {path}")
    if int(summary.get("smoke_rows") or 0) != len(rows):
        raise AssertionError(f"Pilot experiment command smoke summary row count mismatch: {path}")
    if int(summary.get("ok_rows") or 0) != len(rows):
        raise AssertionError(f"Pilot experiment command smoke should have all sampled rows ok: {path}")
    csv_path = optional_latest("pilot_experiment_command_smoke_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Pilot experiment command smoke CSV is missing")
    doc_path = Path("docs/DEEPUplift_PILOT_EXPERIMENT_COMMAND_SMOKE.md")
    if not doc_path.exists():
        raise AssertionError("Pilot experiment command smoke markdown doc is missing")
    return {
        "artifact": "pilot_experiment_command_smoke",
        "path": str(path),
        "smoke_rows": len(rows),
        "ok_rows": summary.get("ok_rows"),
        "experiment_type_counts": summary.get("experiment_type_counts"),
        "priority_counts": summary.get("priority_counts"),
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_pilot_experiment_command_coverage(root: Path) -> dict:
    path = optional_latest("pilot_experiment_command_coverage_latest.json", root)
    if path is None:
        return {"artifact": "pilot_experiment_command_coverage", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    rows = payload.get("rows") or []
    summary = payload.get("summary") or {}
    tracks = payload.get("interview_tracks") or {}
    if payload.get("status") != "ok":
        raise AssertionError(f"Pilot experiment command coverage is not ok: {path}")
    if not isinstance(rows, list) or not rows:
        raise AssertionError(f"Pilot experiment command coverage should contain rows: {path}")
    required = {
        "experiment_id",
        "entity",
        "pilot_gate",
        "priority",
        "experiment_type",
        "command_status",
        "coverage_status",
        "transition",
        "evidence_level",
        "sampled_smoked",
        "sampled_smoked_ok",
        "recommended_command",
        "pass_gate",
        "fail_action",
        "claim_upgrade",
        "claim_boundary",
        "next_action",
        "source_refs",
    }
    allowed_statuses = {"sampled_smoked_ok", "sampled_smoked_failed", "runner_supported_backlog", "command_needs_review"}
    sampled_ok_rows = 0
    for row in rows:
        if not required.issubset(row):
            raise AssertionError(f"Malformed pilot experiment command coverage row: {row}")
        if row.get("coverage_status") not in allowed_statuses:
            raise AssertionError(f"Unknown pilot experiment command coverage status: {row}")
        if row.get("coverage_status") == "sampled_smoked_failed":
            raise AssertionError(f"Pilot command coverage should not include failed sampled smoke rows: {row}")
        if row.get("sampled_smoked_ok"):
            sampled_ok_rows += 1
            if not row.get("runner_artifact"):
                raise AssertionError(f"Sampled-smoked pilot coverage row should reference runner artifact: {row}")
            if "--no-update-latest" not in str(row.get("safe_smoke_command") or "") and "generate_evidence_store.py" not in str(row.get("safe_smoke_command") or ""):
                raise AssertionError(f"Sampled-smoked pilot coverage row should preserve latest artifacts: {row}")
        else:
            if row.get("evidence_level") not in {"planned_runner_supported", "planned_only"}:
                raise AssertionError(f"Unsmoked pilot coverage row has wrong evidence level: {row}")
    for required_track in ["30s", "5min", "15min", "30min"]:
        if not tracks.get(required_track):
            raise AssertionError(f"Pilot experiment command coverage missing {required_track} talk track: {path}")
    if int(summary.get("plan_rows") or 0) != len(rows):
        raise AssertionError(f"Pilot experiment command coverage summary row count mismatch: {path}")
    if int(summary.get("sampled_smoked_ok_rows") or 0) != sampled_ok_rows:
        raise AssertionError(f"Pilot experiment command coverage sampled count mismatch: {path}")
    if sampled_ok_rows < 3:
        raise AssertionError(f"Pilot experiment command coverage should include at least 3 sampled-smoked rows: {path}")
    if summary.get("sampled_smoke_gate") != "ok":
        raise AssertionError(f"Pilot experiment command coverage sampled gate should be ok: {path}")
    if int(summary.get("backlog_rows") or 0) <= 0:
        raise AssertionError(f"Pilot command coverage should expose remaining backlog rows for claim hygiene: {path}")
    csv_path = optional_latest("pilot_experiment_command_coverage_latest.csv", root)
    if csv_path is None:
        raise AssertionError("Pilot experiment command coverage CSV is missing")
    doc_path = Path("docs/DEEPUplift_PILOT_EXPERIMENT_COMMAND_COVERAGE.md")
    if not doc_path.exists():
        raise AssertionError("Pilot experiment command coverage markdown doc is missing")
    return {
        "artifact": "pilot_experiment_command_coverage",
        "path": str(path),
        "plan_rows": len(rows),
        "sampled_smoked_ok_rows": sampled_ok_rows,
        "backlog_rows": summary.get("backlog_rows"),
        "coverage_pct": summary.get("coverage_pct"),
        "sampled_smoke_gate": summary.get("sampled_smoke_gate"),
        "full_coverage_gate": summary.get("full_coverage_gate"),
        "csv": str(csv_path),
        "doc": str(doc_path),
    }


def validate_ui_compare_bundle(root: Path, require: bool = False) -> dict:
    path = optional_latest("*compare_bundle_*.json", root)
    if path is None:
        if require:
            raise AssertionError("No UI compare bundle artifact found")
        return {"artifact": "ui_compare_bundle", "status": "skipped", "reason": "not_found"}
    payload = read_json(path)
    ranking = payload.get("ranking") or []
    runs = payload.get("runs") or []
    if not ranking:
        raise AssertionError(f"UI compare bundle should include ranking rows: {path}")
    if not runs:
        raise AssertionError(f"UI compare bundle should include run artifacts: {path}")
    for row in ranking:
        if not row.get("model") or (not row.get("error") and not row.get("run_id")):
            raise AssertionError(f"Malformed UI compare ranking row in {path}: {row}")
    schema_version = int(payload.get("schema_version") or 1)
    model_cards = payload.get("model_cards") or []
    models = sorted({row.get("model") for row in runs if row.get("model")})
    if schema_version >= 2:
        if len(model_cards) < len(models):
            raise AssertionError(f"UI compare bundle v{schema_version} should include model cards for all runs: {path}")
        if not payload.get("candidate_models"):
            raise AssertionError(f"UI compare bundle v{schema_version} should include candidate_models: {path}")
        for card in model_cards:
            if not card.get("model") or not card.get("parameter_presets") or not card.get("decision_use"):
                raise AssertionError(f"Malformed model card in UI compare bundle: {card}")
    return {
        "artifact": "ui_compare_bundle",
        "path": str(path),
        "schema_version": schema_version,
        "ranking": len(ranking),
        "runs": len(runs),
        "model_cards": len(model_cards),
        "status": "ok" if schema_version >= 2 else "legacy",
    }


def validate_ui_screenshots(root: Path, min_bytes: int) -> dict:
    candidates = sorted(root.glob("ui_screenshots_*"), key=lambda path: path.stat().st_mtime, reverse=True)
    partial = []
    for path in candidates:
        screenshots = sorted(path.glob("*.png"))
        tiny = [shot for shot in screenshots if shot.stat().st_size < min_bytes]
        if len(screenshots) >= 4 and not tiny:
            return {"artifact": "ui_screenshots", "path": str(path), "files": len(screenshots)}
        partial.append({"path": str(path), "files": len(screenshots), "tiny": len(tiny)})
    raise AssertionError(f"No complete UI screenshot set found; latest partial sets: {partial[:3]}")


def validate_latest_run_evidence(runs_dir: Path) -> dict:
    run_dirs = sorted(
        [
            path
            for path in runs_dir.glob("*")
            if path.is_dir()
            and (path / "metrics.json").exists()
            and (path / "config.json").exists()
            and (path / "evidence_manifest.json").exists()
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        raise AssertionError(f"No run evidence directories found in {runs_dir}")
    run_dir = run_dirs[0]
    required = ["config.json", "metrics.json", "predictions.csv", "promotion.json", "environment.json", "evidence_manifest.json"]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise AssertionError(f"Latest run is missing evidence files {missing}: {run_dir}")
    manifest = read_json(run_dir / "evidence_manifest.json")
    if int(manifest.get("schema_version") or 0) < 1:
        raise AssertionError(f"Evidence manifest is missing schema_version: {run_dir}")
    if manifest.get("run_id") != run_dir.name:
        raise AssertionError(f"Evidence manifest run_id does not match directory: {run_dir}")
    if not (manifest.get("data_fingerprint") or {}).get("content_sha256"):
        raise AssertionError(f"Evidence manifest is missing data fingerprint: {run_dir}")
    if not manifest.get("files"):
        raise AssertionError(f"Evidence manifest is missing file hashes: {run_dir}")
    environment = read_json(run_dir / "environment.json")
    if not environment.get("python") or not environment.get("packages"):
        raise AssertionError(f"Environment snapshot is incomplete: {run_dir}")
    promotion = read_json(run_dir / "promotion.json")
    if not promotion.get("level") or not promotion.get("checks"):
        raise AssertionError(f"Promotion gate is incomplete: {run_dir}")
    candidates = [
        "config.json",
        "metrics.json",
        "readiness.json",
        "promotion.json",
        "environment.json",
        "evidence_manifest.json",
        "nuisance_diagnostics.json",
        "predictions.csv",
        "run_note.md",
        "user_note.json",
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        added = []
        for name in candidates:
            path = run_dir / name
            if path.exists() and path.is_file():
                zf.write(path, arcname=f"{run_dir.name}/{name}")
                added.append(name)
    with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as zf:
        names = zf.namelist()
    for name in ["config.json", "metrics.json", "promotion.json", "environment.json", "evidence_manifest.json"]:
        if not any(item.endswith(f"/{name}") for item in names):
            raise AssertionError(f"Evidence ZIP contract missing {name}: {run_dir}")
    return {
        "artifact": "run_evidence",
        "path": str(run_dir),
        "files": len(names),
        "zip_bytes": len(buffer.getvalue()),
        "manifest_files": len(manifest.get("files") or []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the latest DeepUplift regression artifacts.")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--require-ui", action="store_true")
    parser.add_argument("--require-compare-bundle", action="store_true")
    parser.add_argument("--min-screenshot-bytes", type=int, default=20_000)
    args = parser.parse_args()

    root = Path(args.reports_dir)
    rows = [
        validate_audit(root),
        validate_agent_regression(root),
        validate_agent_v2_regression(root),
        validate_model_deconstruction(root),
        validate_deep_model_training_evidence(root),
        validate_deep_model_tuned_benchmark(root),
        validate_deep_model_objective_diagnostics(root),
        validate_algorithm_claim_ledger(root),
        validate_paper_reproduction_gap_ledger(root),
        validate_model_evidence_promotion_matrix(root),
        validate_model_upgrade_recipe_cards(root),
        validate_deep_model_telemetry_gap_matrix(root),
        validate_deep_model_telemetry_smoke(root),
        validate_deep_model_forward_hook_contracts(root),
        validate_deep_model_forward_hook_smoke(root),
        validate_deep_model_hook_training_bridge(root),
        validate_deep_model_trainer_collector_smoke(root),
        validate_deep_model_telemetry_benchmark_linkage(root),
        validate_deep_model_telemetry_ablation_gate(root),
        validate_deep_model_ablation_interpretation(root),
        validate_deep_model_ablation_promotion_matrix(root),
        validate_deep_model_ablation_next_experiment_plan(root),
        validate_deep_model_ablation_command_contract(root),
        validate_deep_model_ablation_command_contract_smoke(root),
        validate_deep_model_ablation_recommended_command_smoke(root),
        validate_deep_model_ablation_command_smoke_coverage(root),
        validate_deep_model_ablation_command_smoke_coverage_diff(root),
        validate_paper_benchmark(root),
        validate_paper_benchmark_interpretation(root),
        validate_glossary(root),
        validate_no_ui_compare(root),
        validate_benchmark_suite(root),
        validate_dataset_registry(root),
        validate_ope_engine(root),
        validate_nuisance_diagnostics(root),
        validate_budget_policy_optimizer(root),
        validate_multi_treatment_benchmark(root),
        validate_continuous_treatment_policy(root),
        validate_evidence_store(root),
        validate_regression_warning_audit(root),
        validate_promotion_launch_cards(root),
        validate_regression_warning_triage(root),
        validate_pilot_readiness_scorecard(root),
        validate_pilot_experiment_plan(root),
        validate_pilot_experiment_command_smoke(root),
        validate_pilot_experiment_command_coverage(root),
        validate_ui_compare_bundle(root, require=args.require_compare_bundle),
        validate_latest_run_evidence(Path(args.runs_dir)),
    ]
    if args.require_ui:
        rows.append(validate_ui_screenshots(root, args.min_screenshot_bytes))
    print(json.dumps({"status": "ok", "artifacts": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
