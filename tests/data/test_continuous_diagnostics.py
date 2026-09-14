import numpy as np
import pandas as pd

from deepuplift.contracts import TreatmentType
from deepuplift.data import diagnose_continuous_treatment, local_treatment_support
from deepuplift.data.schema import create_causal_dataset


def test_continuous_dose_and_local_support_diagnostics_report_empirical_coverage():
    rng = np.random.default_rng(9)
    rows = 90
    segment = np.where(np.arange(rows) < rows // 2, "low", "high")
    dose = np.where(segment == "low", rng.uniform(0, 8, rows), rng.uniform(12, 20, rows))
    frame = pd.DataFrame({"id": np.arange(rows), "x": rng.normal(size=rows), "segment": segment, "dose": dose, "y": dose + rng.normal(size=rows)})
    data = create_causal_dataset(frame, feature_cols=["x", "segment"], treatment_col="dose", outcome_col="y", treatment_type=TreatmentType.CONTINUOUS, id_column="id")
    summary = diagnose_continuous_treatment(data, segment_cols=["segment"], bins=5)
    assert summary["treatment_min"] == float(dose.min())
    assert len(summary["dose_histogram"]["counts"]) == 5
    assert len(summary["feature_segments"]["segment"]) == 2
    local = local_treatment_support(data.subset(np.arange(70)), data.subset(np.arange(70, 90)), k=12, dose_grid=[0.0, 10.0, 20.0])
    assert local["diagnostic_name"] == "local_empirical_treatment_support_diagnostic"
    assert len(local["support_score"]) == 20
    assert all(0.0 <= score <= 1.0 for score in local["support_score"])
