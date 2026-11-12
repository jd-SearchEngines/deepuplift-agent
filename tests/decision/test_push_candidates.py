from __future__ import annotations

import numpy as np
import pandas as pd

from deepuplift.decision import build_push_candidate_bundle, optimize_push_candidate_bundle
from deepuplift.decision.optimization import AllocationConstraints


def test_push_candidate_values_come_from_model_predictions_and_remain_non_monetary():
    predictions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "time_slot": [0, 1, 0, 1],
            "action": ["activity", "diet", "activity", "diet"],
            "predicted_incremental_effect": [0.9, 0.2, 0.8, -0.1],
            "eligible_flag": [True, True, True, False],
        }
    )
    bundle = build_push_candidate_bundle(
        predictions,
        outcome="activity_steps_60min",
        model_version="wcls-test-v1",
        eligibility_col="eligible_flag",
    )
    assert bundle.metadata["source"] == "fitted_model_predictions"
    assert bundle.metadata["raw_business_cost_observed"] is False
    assert np.allclose(bundle.candidates.incremental_value, [0.9, 0.2, 0.8, -0.1])
    assert bundle.frame.loc[3, "eligible"] is False or bool(bundle.frame.loc[3, "eligible"]) is False
    assert bundle.frame["resource_cost_unit"].iloc[0] == "push_capacity_units"


def test_push_candidate_bundle_runs_existing_solvers_and_independent_feasibility():
    predictions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "time_slot": [0, 1, 0, 1],
            "action": ["activity", "diet", "activity", "diet"],
            "predicted_incremental_effect": [4.0, 3.0, 2.0, 1.0],
        }
    )
    bundle = build_push_candidate_bundle(predictions, outcome="steps", model_version="test")
    report = optimize_push_candidate_bundle(
        bundle,
        AllocationConstraints(budget=2, slot_capacity={0: 1, 1: 1}, user_frequency_cap=1, action_quota={"activity": 1, "diet": 1}),
        include_exact=True,
    )
    assert report["status"] == "PASS"
    assert report["solvers"]["greedy"]["independent_feasibility"]["feasible"]
    assert report["solvers"]["heap"]["candidate_provenance"]["model_version"] == "test"
