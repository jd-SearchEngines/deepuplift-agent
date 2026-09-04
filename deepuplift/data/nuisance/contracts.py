from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass
class NuisanceResult:
    """Cross-fitted nuisance estimates and their audit trail.

    Arrays are deliberately retained in memory for effect learners.  ``to_dict``
    is an evidence-safe JSON representation and never serializes fitted objects.
    """

    propensity_scores: np.ndarray
    propensity_model_metadata: dict[str, Any]
    outcome_control_oof: np.ndarray
    outcome_treated_oof: np.ndarray
    fold_ids: np.ndarray
    overlap_mask: np.ndarray
    trim_mask: np.ndarray
    sample_weights: np.ndarray
    effective_sample_size: dict[str, float | None]
    clipping_range: tuple[float, float] | None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    propensity_model: Any | None = field(default=None, repr=False)
    propensity_preprocessor: Any | None = field(default=None, repr=False)

    def predict_propensity(self, x: Any) -> np.ndarray:
        if self.propensity_model is None or self.propensity_preprocessor is None:
            return np.full(len(x), float(np.mean(self.propensity_scores)), dtype="float64")
        model_columns = getattr(self.propensity_model, "feature_names_in_", None)
        if model_columns is not None and hasattr(x, "columns") and set(model_columns).issubset(set(x.columns)):
            transformed = x.loc[:, list(model_columns)]
        else:
            transformed = self.propensity_preprocessor.transform(x)
        return np.asarray(self.propensity_model.predict_proba(transformed)[:, 1], dtype="float64")

    def to_dict(self) -> dict[str, Any]:
        payload = {"propensity_scores": self.propensity_scores, "propensity_model_metadata": self.propensity_model_metadata, "outcome_control_oof": self.outcome_control_oof, "outcome_treated_oof": self.outcome_treated_oof, "fold_ids": self.fold_ids, "overlap_mask": self.overlap_mask, "trim_mask": self.trim_mask, "sample_weights": self.sample_weights, "effective_sample_size": self.effective_sample_size, "clipping_range": self.clipping_range, "diagnostics": self.diagnostics, "metadata": self.metadata}
        return _jsonable(payload)


__all__ = ["NuisanceResult"]
