from __future__ import annotations

import numpy as np
import pandas as pd

from deepuplift.contracts import CausalDataset, EffectPrediction, TreatmentType
from deepuplift.data.preprocessing import TabularPreprocessor


class CausalForestDMLAdapter:
    """EconML CausalForestDML adapter with an explicit internal-nuisance boundary."""

    name = "CausalForestDML"
    backend = "econml"
    nuisance_mode = "internal"

    def __init__(self, estimator_name: str = "CausalForestDML", **kwargs):
        self.name = estimator_name
        self.kwargs = kwargs

    def fit(self, dataset: CausalDataset, nuisance=None):
        if dataset.treatment_type != TreatmentType.BINARY:
            raise ValueError("CausalForestDMLAdapter supports binary treatment only.")
        try:
            from econml.dml import CausalForestDML
        except ImportError as exc:
            raise ImportError("CausalForestDML requires the optional 'econml' extra.") from exc
        frame = dataset.to_pandas().reset_index(drop=True)
        values = sorted(frame[dataset.treatment_col].dropna().unique().tolist(), key=lambda item: str(item))
        if len(values) != 2:
            raise ValueError("CausalForestDMLAdapter requires exactly two treatment values.")
        treatment = frame[dataset.treatment_col].map({values[0]: 0, values[1]: 1}).to_numpy()
        outcome = pd.to_numeric(frame[dataset.outcome_col], errors="coerce").to_numpy(dtype="float64")
        self.preprocessor = TabularPreprocessor(dataset.feature_cols)
        x = self.preprocessor.fit_transform(frame[dataset.feature_cols])
        params = {"discrete_treatment": True, "random_state": 42, "n_estimators": 100, "n_jobs": -1, **self.kwargs}
        self.model = CausalForestDML(**params)
        self.model.fit(outcome, treatment, X=x)
        self.feature_cols = list(dataset.feature_cols)
        self.metadata = {"model_name": self.name, "backend": "econml", "nuisance_mode": "internal", "nuisance_provenance": "EconML CausalForestDML internally manages nuisance estimation", "caller_nuisance_supplied": nuisance is not None, "treatment_mapping": {str(k): v for k, v in {values[0]: 0, values[1]: 1}.items()}}
        return self

    def predict(self, dataset: CausalDataset) -> EffectPrediction:
        if not hasattr(self, "model"):
            raise RuntimeError("Model must be fitted before predict.")
        x = self.preprocessor.transform(dataset.to_pandas()[self.feature_cols])
        uplift = np.asarray(self.model.effect(x), dtype="float64").reshape(-1)
        return EffectPrediction(unit_id=dataset.unit_ids.to_numpy(), treatment_type=TreatmentType.BINARY, uplift=uplift, cate=uplift, y0=np.zeros(len(uplift)), y1=uplift, recommended_effect=uplift, propensity=np.full(len(uplift), .5), metadata=self.metadata)


EconMLAdapter = CausalForestDMLAdapter

__all__ = ["CausalForestDMLAdapter", "EconMLAdapter"]
