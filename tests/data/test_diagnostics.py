import numpy as np
import pandas as pd

from deepuplift.data import AssignmentType, TreatmentType, create_causal_dataset, diagnose_dataset


def _frame(rows=200):
    rng = np.random.default_rng(7)
    treatment = rng.binomial(1, 0.5, rows)
    return pd.DataFrame({
        "user_id": [f"u{i}" for i in range(rows)],
        "x": rng.normal(size=rows),
        "segment": rng.choice(["a", "b"], rows),
        "treatment": treatment,
        "outcome": rng.binomial(1, 0.3 + 0.1 * treatment, rows),
    })


def test_randomized_diagnostics_are_distinct():
    dataset = create_causal_dataset(
        _frame(), feature_cols=["x", "segment"], treatment_col="treatment", outcome_col="outcome", id_column="user_id",
        treatment_type=TreatmentType.BINARY, assignment_type=AssignmentType.RANDOMIZED,
    )
    report = diagnose_dataset(dataset)
    assert report.sample_size == 200
    assert report.propensity_summary["status"] == "not_requested"
    assert report.metadata["assignment_type"] == "randomized"


def test_observational_diagnostics_estimate_propensity():
    dataset = create_causal_dataset(
        _frame(), feature_cols=["x", "segment"], treatment_col="treatment", outcome_col="outcome", id_column="user_id",
        treatment_type="binary", assignment_type="observational",
    )
    report = diagnose_dataset(dataset)
    assert report.propensity_summary["status"] == "estimated"
    assert "common_support_rate" in report.overlap_summary
    assert any("assumptions" in warning for warning in report.warnings)
