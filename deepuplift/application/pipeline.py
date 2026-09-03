from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from deepuplift.contracts import CausalDataset, DataDiagnostics, EffectPrediction, PolicyResult
from deepuplift.data import AssignmentType, TreatmentType, create_causal_dataset, diagnose_dataset, split_dataset
from deepuplift.decision import build_binary_policy, build_experiment_plan, calibration_metrics, ranking_metrics
from deepuplift.models.registry import build_model, is_model_available, missing_dependencies


@dataclass
class ModelBenchmarkResult:
    rows: list[dict[str, Any]]
    recommended_model: str
    recommendation_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UpliftPipelineResult:
    dataset: CausalDataset
    diagnostics: DataDiagnostics
    benchmark: ModelBenchmarkResult
    prediction: EffectPrediction
    policy: PolicyResult
    experiment_plan: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnostics": self.diagnostics.to_dict(),
            "benchmark": self.benchmark.to_dict(),
            "prediction": self.prediction.to_frame().to_dict(orient="records"),
            "policy": self.policy.to_dict(),
            "experiment_plan": self.experiment_plan,
        }


def _model_score(row: dict[str, Any]) -> float:
    ranking = row.get("qini") if row.get("qini") is not None else -1e9
    top_k = row.get("uplift_at_10") if row.get("uplift_at_10") is not None else -1e9
    calibration = row.get("calibration_mae")
    calibration_component = -calibration if calibration is not None else -1e9
    seconds = float(row.get("train_seconds") or 0.0) + float(row.get("predict_seconds") or 0.0)
    speed_component = -seconds / 1000.0
    return 0.45 * ranking + 0.30 * top_k + 0.20 * calibration_component + 0.05 * speed_component


def benchmark_models(
    train_dataset: CausalDataset,
    test_dataset: CausalDataset,
    *,
    model_names: Iterable[str] = ("S-Learner", "T-Learner", "X-Learner", "DR-Learner"),
    task: str = "classification",
    random_state: int = 42,
) -> tuple[ModelBenchmarkResult, dict[str, EffectPrediction]]:
    rows: list[dict[str, Any]] = []
    predictions: dict[str, EffectPrediction] = {}
    for model_name in model_names:
        if not is_model_available(model_name):
            rows.append({"model": model_name, "status": "SKIPPED", "missing_dependencies": missing_dependencies(model_name)})
            continue
        started = time.perf_counter()
        try:
            model = build_model(model_name, task=task, random_state=random_state)
            model.fit(train_dataset)
            train_seconds = time.perf_counter() - started
            prediction_started = time.perf_counter()
            prediction = model.predict(test_dataset)
            predict_seconds = time.perf_counter() - prediction_started
            metrics = ranking_metrics(prediction, test_dataset)
            calibration = calibration_metrics(prediction, test_dataset)
            top10 = next((item["uplift"] for item in metrics["uplift_at_k"] if item["fraction"] == 0.1), None)
            row = {
                "model": model_name,
                "status": "PASS",
                "qini": metrics["qini"],
                "auuc": metrics["auuc"],
                "uplift_at_10": top10,
                "calibration_mae": calibration["mae"],
                "train_seconds": train_seconds,
                "predict_seconds": predict_seconds,
            }
            predictions[model_name] = prediction
        except Exception as exc:
            row = {"model": model_name, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        row["decision_score"] = _model_score(row) if row["status"] == "PASS" else None
        rows.append(row)
    passed = [row for row in rows if row["status"] == "PASS"]
    if not passed:
        raise RuntimeError("No candidate model completed successfully.")
    selected = max(passed, key=lambda row: row["decision_score"])
    reason = (
        "Selected using a composite of QINI, Top-10% uplift, calibration error and runtime; "
        "the decision is not based on QINI alone."
    )
    return ModelBenchmarkResult(rows, selected["model"], reason), predictions


def run_uplift_pipeline(
    data: Any,
    *,
    feature_cols: list[str],
    treatment_col: str,
    outcome_col: str,
    treatment_type: TreatmentType | str = TreatmentType.BINARY,
    assignment_type: AssignmentType | str = AssignmentType.RANDOMIZED,
    id_column: str | None = None,
    model_names: Iterable[str] = ("S-Learner", "T-Learner", "X-Learner", "DR-Learner"),
    task: str = "classification",
    test_size: float = 0.25,
    random_state: int = 42,
    treatment_cost: float = 0.0,
    outcome_value: float = 1.0,
    budget: float | None = None,
    max_contacts: int | None = None,
) -> UpliftPipelineResult:
    dataset = create_causal_dataset(
        data,
        feature_cols=feature_cols,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        treatment_type=treatment_type,
        assignment_type=assignment_type,
        id_column=id_column,
    )
    if dataset.treatment_type != TreatmentType.BINARY:
        raise NotImplementedError("The reference pipeline is complete for binary treatment; use layer contracts for multi/continuous extensions.")
    diagnostics = diagnose_dataset(dataset)
    train_dataset, test_dataset = split_dataset(dataset, test_size=test_size, random_state=random_state)
    benchmark, _ = benchmark_models(
        train_dataset,
        test_dataset,
        model_names=model_names,
        task=task,
        random_state=random_state,
    )
    selected_model = build_model(benchmark.recommended_model, task=task, random_state=random_state)
    selected_model.fit(dataset)
    prediction = selected_model.predict(dataset)
    policy = build_binary_policy(
        prediction,
        treatment_cost=treatment_cost,
        outcome_value=outcome_value,
        budget=budget,
        max_contacts=max_contacts,
    )
    experiment_plan = build_experiment_plan(policy, assignment_type=dataset.assignment_type, randomization_unit=id_column or "unit_id")
    experiment_plan["benchmark_holdout_metrics"] = next(
        row for row in benchmark.rows if row.get("model") == benchmark.recommended_model
    )
    return UpliftPipelineResult(dataset, diagnostics, benchmark, prediction, policy, experiment_plan)
