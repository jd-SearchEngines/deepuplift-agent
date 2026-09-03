from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "prompt_tokens",
    "ambiguity_score",
    "retrieval_need",
    "safety_risk",
    "user_value",
    "latency_slo_ms",
    "prior_fail_rate",
    "model_pool_load",
    "task_type",
    "language",
    "propensity",
    "small_model_quality",
    "strong_model_quality",
    "true_uplift",
    "incremental_cost",
    "latency_penalty",
    "oracle_policy_value",
    "treatment",
    "outcome",
}

MULTI_ACTION_COLUMNS = {
    "tool_need",
    "evidence_requirement",
    "judge_noise_risk",
    "budget_pressure",
    "route_action",
    "best_action",
    "cheap_quality",
    "rag_quality",
    "tool_quality",
    "human_quality",
    "selected_action_quality",
    "hallucination_risk",
    "evidence_failure_risk",
    "oracle_best_policy_value",
}

ORACLE_COLUMNS = {
    "propensity",
    "small_model_quality",
    "strong_model_quality",
    "cheap_quality",
    "rag_quality",
    "tool_quality",
    "human_quality",
    "selected_action_quality",
    "true_uplift",
    "incremental_cost",
    "latency_penalty",
    "hallucination_risk",
    "evidence_failure_risk",
    "oracle_policy_value",
    "oracle_best_policy_value",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the synthetic LLM routing uplift dataset.")
    parser.add_argument("--manifest", default="examples/datasets/manifest.json")
    parser.add_argument("--dataset-id", default="synthetic_llm_routing_uplift_8k")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    dataset = next((row for row in manifest.get("datasets", []) if row.get("id") == args.dataset_id), None)
    failures: list[str] = []
    if not dataset:
        failures.append(f"Dataset not found: {args.dataset_id}")
        payload = {"status": "fail", "failures": failures}
        print(json.dumps(payload, ensure_ascii=False))
        raise SystemExit(1)

    path = Path(dataset["path"])
    if not path.exists():
        failures.append(f"Dataset file missing: {path}")
        payload = {"status": "fail", "dataset": dataset, "failures": failures}
        print(json.dumps(payload, ensure_ascii=False))
        raise SystemExit(1)

    df = pd.read_csv(path)
    expected_columns = set(REQUIRED_COLUMNS)
    if args.dataset_id == "synthetic_llm_multi_action_routing_9k":
        expected_columns = (expected_columns - {"small_model_quality", "strong_model_quality"}) | MULTI_ACTION_COLUMNS
    missing_columns = sorted(expected_columns - set(df.columns))
    if missing_columns:
        failures.append(f"Missing columns: {missing_columns}")
    leakage = sorted(ORACLE_COLUMNS & set(dataset.get("feature_cols", [])))
    if leakage:
        failures.append(f"Oracle columns should not be feature columns: {leakage}")

    treatment_rate = float(df["treatment"].mean())
    outcome_rate = float(df["outcome"].mean())
    true_uplift = df["true_uplift"]
    policy_value = df["oracle_policy_value"]
    if len(df) < 8000:
        failures.append(f"Expected at least 8000 rows, found {len(df)}")
    if not 0.25 <= treatment_rate <= 0.75:
        failures.append(f"Treatment rate outside smoke bounds: {treatment_rate:.3f}")
    if not 0.2 <= outcome_rate <= 0.85:
        failures.append(f"Outcome rate outside smoke bounds: {outcome_rate:.3f}")
    if not (true_uplift.min() < 0 and true_uplift.max() > 0.25):
        failures.append("true_uplift should include negative/small and high positive routing effects.")
    if policy_value.quantile(0.9) <= policy_value.quantile(0.1):
        failures.append("oracle_policy_value should have meaningful spread.")

    task_count = int(df["task_type"].nunique())
    language_count = int(df["language"].nunique())
    if task_count < 5:
        failures.append(f"Expected at least 5 task types, found {task_count}")
    if language_count < 3:
        failures.append(f"Expected 3 language groups, found {language_count}")
    action_count = None
    oracle_best_gap = None
    if args.dataset_id == "synthetic_llm_multi_action_routing_9k":
        action_count = int(df["route_action"].nunique()) if "route_action" in df.columns else 0
        if action_count < 4:
            failures.append(f"Expected 4 route actions, found {action_count}")
        if {"oracle_best_policy_value", "oracle_policy_value"} <= set(df.columns):
            oracle_best_gap = float((df["oracle_best_policy_value"] - df["oracle_policy_value"]).mean())
            if oracle_best_gap <= -0.02:
                failures.append(f"Oracle best action should not be materially worse than planned action: {oracle_best_gap:.4f}")

    payload = {
        "status": "fail" if failures else "ok",
        "dataset_id": dataset.get("id"),
        "path": str(path),
        "rows": len(df),
        "treatment_rate": treatment_rate,
        "outcome_rate": outcome_rate,
        "true_uplift_min": float(true_uplift.min()),
        "true_uplift_max": float(true_uplift.max()),
        "policy_value_p10": float(policy_value.quantile(0.1)),
        "policy_value_p90": float(policy_value.quantile(0.9)),
        "task_types": task_count,
        "languages": language_count,
        "route_actions": action_count,
        "oracle_best_minus_planned_policy_mean": oracle_best_gap,
        "failures": failures,
    }
    output = Path("reports/llm_routing_dataset_smoke_latest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
