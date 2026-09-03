import pandas as pd

from deepuplift.data import create_causal_dataset
from deepuplift.models.binary import DRLearner, SLearner, TLearner, XLearner
from deepuplift.scenarios.coupon_allocation import generate_coupon_data


def test_native_binary_baselines_return_effect_prediction():
    frame = generate_coupon_data(500, random_state=11)
    dataset = create_causal_dataset(
        frame,
        feature_cols=["affinity", "price_sensitivity", "sessions_7d", "orders_30d", "spend_90d", "recency_days", "region"],
        treatment_col="treatment",
        outcome_col="outcome",
        id_column="user_id",
    )
    for model_type in (SLearner, TLearner, XLearner, DRLearner):
        prediction = model_type(random_state=11).fit(dataset).predict(dataset)
        assert len(prediction.to_frame()) == 500
        assert "uplift" in prediction.to_frame()
