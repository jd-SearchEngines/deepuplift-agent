from __future__ import annotations

import numpy as np
import pandas as pd

from deepuplift.contracts import CausalDataset, EffectPrediction, TreatmentType
from deepuplift.data.preprocessing import TabularPreprocessor
from deepuplift.models.binary.base import OutcomeEstimator


class MultiTreatmentOutcomeModel:
    """One outcome model per discrete action, with action-specific effects."""

    name = "MultiTreatmentOutcome"

    def __init__(self, task: str = "classification", random_state: int = 42):
        self.task = task
        self.random_state = random_state

    def fit(self, dataset: CausalDataset) -> "MultiTreatmentOutcomeModel":
        if dataset.treatment_type != TreatmentType.MULTI_DISCRETE:
            raise ValueError("MultiTreatmentOutcomeModel requires multi_discrete treatment.")
        frame = dataset.to_pandas().dropna(subset=[dataset.treatment_col, dataset.outcome_col]).reset_index(drop=True)
        self.preprocessor = TabularPreprocessor(dataset.feature_cols)
        self.x = self.preprocessor.fit_transform(frame[dataset.feature_cols])
        self.values = sorted(frame[dataset.treatment_col].unique().tolist(), key=lambda item: str(item))
        if len(self.values) < 3:
            raise ValueError("MultiTreatmentOutcomeModel expects at least three discrete actions.")
        self.models = {}
        for value in self.values:
            mask = frame[dataset.treatment_col].to_numpy() == value
            if mask.sum() < 2:
                raise ValueError(f"Treatment action {value!r} has fewer than two rows.")
            outcome = pd.to_numeric(frame.loc[mask, dataset.outcome_col], errors="coerce").to_numpy(dtype="float64")
            if np.isnan(outcome).any():
                raise ValueError("Outcome must be numeric for the reference multi-treatment adapter.")
            outcome_task = "regression" if self.task == "classification" and len(np.unique(outcome)) > 2 else self.task
            self.models[value] = OutcomeEstimator(outcome_task, self.random_state).fit(self.x.loc[mask], outcome)
        self.feature_cols = list(dataset.feature_cols)
        return self

    def predict(self, dataset: CausalDataset) -> EffectPrediction:
        if not getattr(self, "models", None):
            raise RuntimeError("Model must be fitted before predict.")
        x = self.preprocessor.transform(dataset.to_pandas()[self.feature_cols])
        outcomes = {value: model.predict(x) for value, model in self.models.items()}
        baseline = self.values[0]
        effects = {value: values - outcomes[baseline] for value, values in outcomes.items() if value != baseline}
        effect_frame = pd.DataFrame(effects)
        recommended_treatment = effect_frame.idxmax(axis=1).to_numpy() if not effect_frame.empty else np.full(len(x), baseline)
        recommended_effect = effect_frame.max(axis=1).to_numpy() if not effect_frame.empty else np.zeros(len(x))
        recommended_outcome = (
            np.asarray([outcomes[value][index] for index, value in enumerate(recommended_treatment)])
            if len(x)
            else np.array([])
        )
        return EffectPrediction(
            unit_id=dataset.unit_ids.to_numpy(),
            treatment_type=TreatmentType.MULTI_DISCRETE,
            y0=outcomes[baseline],
            y1=recommended_outcome,
            treatment_effects=effects,
            recommended_treatment=recommended_treatment,
            recommended_effect=recommended_effect,
            metadata={"model_name": self.name, "baseline_treatment": baseline, "treatment_values": self.values},
        )
