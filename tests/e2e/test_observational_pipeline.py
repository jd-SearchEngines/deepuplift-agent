import numpy as np

from deepuplift.application import run_uplift_pipeline
from deepuplift.data import synthetic_ground_truth


def test_formal_observational_application_pipeline_uses_nuisance_and_policy():
    source = synthetic_ground_truth(260, 17)
    result = run_uplift_pipeline(source.to_pandas(), feature_cols=source.feature_cols, treatment_col=source.treatment_col, outcome_col=source.outcome_col, treatment_type=source.treatment_type, assignment_type=source.assignment_type, id_column=source.id_column, model_names=["DR-Learner"], treatment_cost=.01, outcome_value=1.0, nuisance_config={"estimator": "logistic", "cross_fit": True, "n_splits": 5, "weighting": "overlap", "trim_threshold": .05})
    assert result.nuisance is not None
    assert result.experiment_plan["nuisance_used"] is True
    assert result.experiment_plan["nuisance_config"]["weighting"] == "overlap"
    assert np.isfinite(result.prediction.uplift).all()
    assert result.policy.rows
    assert result.prediction.metadata["nuisance_provenance"]["provenance"] == "deepuplift unified nuisance layer"
    assert result.diagnostics.metadata["observational_causal_assumption"]

