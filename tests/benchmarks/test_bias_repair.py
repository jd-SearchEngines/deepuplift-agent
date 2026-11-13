import numpy as np
import pandas as pd

from deepuplift.benchmarks.bias_repair import (
    run_binary_bias_repair,
    run_continuous_gps_repair,
)
from deepuplift.contracts import AssignmentType, TreatmentType
from deepuplift.data.schema import create_causal_dataset


def _binary_dataset(seed=3):
    rng = np.random.default_rng(seed)
    n = 180
    x = rng.normal(size=(n, 2))
    p = 1 / (1 + np.exp(-x[:, 0]))
    t = rng.binomial(1, p)
    y = 1.5 * t + x[:, 0] + rng.normal(scale=.4, size=n)
    frame = pd.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "t": t, "y": y})
    return create_causal_dataset(frame, feature_cols=["x0", "x1"], treatment_col="t", outcome_col="y", treatment_type=TreatmentType.BINARY, assignment_type=AssignmentType.OBSERVATIONAL)


def test_binary_bias_repair_reports_all_estimators_and_crossfit():
    result = run_binary_bias_repair(_binary_dataset(), bootstrap_reps=25)
    names = {row["method"] for row in result["methods"]}
    assert {"ipw", "stabilized_ipw", "overlap_weighting", "propensity_matching", "aipw_dr_crossfit"} <= names
    assert result["propensity"]["common_support_05_95"] > 0
    assert all(np.isfinite(row["estimate"]) for row in result["methods"] if row["status"] == "PASS")
    assert all(row["weight_diagnostics"]["ess_overall"] is not None for row in result["methods"])


def test_continuous_gps_reports_density_ratio_clipping_and_support():
    rng = np.random.default_rng(4)
    n = 160
    x = rng.normal(size=(n, 2))
    dose = np.clip(10 + 2 * x[:, 0] + rng.normal(size=n), 0.1, None)
    y = dose + x[:, 1] + rng.normal(size=n)
    frame = pd.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "dose": dose, "y": y})
    dataset = create_causal_dataset(frame, feature_cols=["x0", "x1"], treatment_col="dose", outcome_col="y", treatment_type=TreatmentType.CONTINUOUS, assignment_type=AssignmentType.OBSERVATIONAL)
    result = run_continuous_gps_repair(dataset)
    assert result["gps"]["dose_range"][0] < result["gps"]["dose_range"][1]
    assert len(result["clipping_ablations"]) == 3
    assert all(row["weight_diagnostics"]["ess"] is not None for row in result["clipping_ablations"])
