from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.application import run_uplift_pipeline
from deepuplift.data import synthetic_ground_truth


def main() -> int:
    source = synthetic_ground_truth(800, seed=42)
    result = run_uplift_pipeline(
        source.to_pandas(),
        feature_cols=source.feature_cols,
        treatment_col=source.treatment_col,
        outcome_col=source.outcome_col,
        treatment_type=source.treatment_type,
        assignment_type="observational",
        id_column=source.id_column,
        model_names=["DR-Learner", "R-Learner"],
        nuisance_config={"estimator": "logistic", "cross_fit": True, "n_splits": 5, "weighting": "overlap", "trim_threshold": 0.05},
        treatment_cost=0.01,
        outcome_value=1.0,
    )
    summary = result.policy.summary
    print("DeepUplift Observational Targeting Demo")
    print(f"SCENARIO   rows={len(result.dataset.to_pandas())} assignment={result.dataset.assignment_type.value} treatment={result.dataset.treatment_type.value}")
    print(f"DIAGNOSTIC readiness={result.diagnostics.readiness} nuisance_used={result.experiment_plan['nuisance_used']} weighting={result.experiment_plan['nuisance_config']['weighting']}")
    print(f"MODEL      recommended={result.benchmark.recommended_model}")
    print(f"DECISION   targets={summary['target_count']} net_value={summary['expected_net_value']:.2f} iROAS={summary['iroas']}")
    print("WARNING    Conditional ignorability and positivity remain assumptions; this is offline evidence only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
