from __future__ import annotations

import numpy as np
import pandas as pd

from deepuplift.contracts import CausalDataset, EffectPrediction, TreatmentType
from deepuplift.data.preprocessing import TabularPreprocessor


class CausalMLUpliftAdapter:
    def __init__(self, estimator_name: str = "CausalMLUpliftTree", **kwargs):
        self.name = estimator_name
        self.kwargs = kwargs

    def _class(self):
        try:
            from causalml.inference.tree import UpliftRandomForestClassifier, UpliftTreeClassifier
        except ImportError as exc:
            raise ImportError("CausalML uplift models require the optional 'causalml' extra.") from exc
        return UpliftRandomForestClassifier if self.name == "CausalMLUpliftRandomForest" else UpliftTreeClassifier

    def fit(self, dataset: CausalDataset, nuisance=None):
        if dataset.treatment_type != TreatmentType.BINARY:
            raise ValueError("CausalML uplift adapters support binary treatment only.")
        frame = dataset.to_pandas().reset_index(drop=True)
        values = sorted(frame[dataset.treatment_col].dropna().unique().tolist(), key=lambda item: str(item))
        if len(values) != 2:
            raise ValueError("CausalML uplift adapters require exactly two treatment values.")
        self.preprocessor = TabularPreprocessor(dataset.feature_cols)
        x = self.preprocessor.fit_transform(frame[dataset.feature_cols])
        treatment = frame[dataset.treatment_col].map({values[0]: "control", values[1]: "treatment"}).to_numpy()
        outcome = pd.to_numeric(frame[dataset.outcome_col], errors="coerce").to_numpy(dtype="float64")
        cls = self._class()
        params = dict(self.kwargs)
        if self.name == "CausalMLUpliftTree":
            params.setdefault("control_name", "control")
        self.model = cls(**params)
        self.model.fit(X=x, treatment=treatment, y=outcome)
        self.feature_cols = list(dataset.feature_cols)
        self.metadata = {"model_name": self.name, "backend": "causalml", "nuisance_mode": "internal", "nuisance_provenance": "CausalML uplift estimator owns its internal fitting path", "caller_nuisance_supplied": nuisance is not None, "treatment_mapping": {str(values[0]): "control", str(values[1]): "treatment"}}
        return self

    def predict(self, dataset: CausalDataset) -> EffectPrediction:
        if not hasattr(self, "model"):
            raise RuntimeError("Model must be fitted before predict.")
        x = self.preprocessor.transform(dataset.to_pandas()[self.feature_cols])
        raw = np.asarray(self.model.predict(x))
        uplift = raw[:, -1] if raw.ndim == 2 else raw.reshape(-1)
        return EffectPrediction(unit_id=dataset.unit_ids.to_numpy(), treatment_type=TreatmentType.BINARY, uplift=uplift, cate=uplift, y0=np.zeros(len(uplift)), y1=uplift, recommended_effect=uplift, propensity=np.full(len(uplift), .5), metadata=self.metadata)


CausalMLAdapter = CausalMLUpliftAdapter

__all__ = ["CausalMLUpliftAdapter", "CausalMLAdapter"]
