from __future__ import annotations

import numpy as np
import pandas as pd

from deepuplift.contracts import CausalDataset, EffectPrediction, TreatmentType
from deepuplift.decision.evaluation.ranking import ranking_metrics


def test_ranking_metrics_can_skip_materialized_curves_for_large_runs():
    rows = 2000
    frame = pd.DataFrame(
        {
            "unit_id": np.arange(rows),
            "x": np.linspace(-1.0, 1.0, rows),
            "treatment": np.arange(rows) % 2,
            "outcome": np.linspace(0.0, 1.0, rows),
        }
    )
    dataset = CausalDataset(
        frame,
        feature_cols=["x"],
        treatment_col="treatment",
        outcome_col="outcome",
        treatment_type=TreatmentType.BINARY,
        id_column="unit_id",
    )
    prediction = EffectPrediction(
        unit_id=frame["unit_id"].to_numpy(),
        treatment_type=TreatmentType.BINARY,
        cate=frame["x"].to_numpy(),
    )

    metrics = ranking_metrics(prediction, dataset, include_curves=False)

    assert metrics["rows"] == rows
    assert metrics["qini"] is not None
    assert metrics["curve"] is None
    assert metrics["gain_curve"] is None
    assert metrics["treatment_balance_curve"] is None
