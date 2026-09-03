from deepuplift.application import run_uplift_pipeline
from deepuplift.scenarios.coupon_allocation import generate_coupon_data


def test_coupon_allocation_vertical_slice():
    frame = generate_coupon_data(700, random_state=19)
    result = run_uplift_pipeline(
        frame,
        feature_cols=["affinity", "price_sensitivity", "sessions_7d", "orders_30d", "spend_90d", "recency_days", "region"],
        treatment_col="treatment",
        outcome_col="outcome",
        id_column="user_id",
        budget=1000,
        treatment_cost=10,
        outcome_value=35,
    )
    assert result.diagnostics.sample_size == 700
    assert len([row for row in result.benchmark.rows if row["status"] == "PASS"]) == 4
    assert result.prediction.metadata["model_name"] == result.benchmark.recommended_model
    assert result.policy.summary["target_count"] <= 100
    assert result.experiment_plan["status"] == "READY_FOR_EXPERIMENT_DESIGN"
