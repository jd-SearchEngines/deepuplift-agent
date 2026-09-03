from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from .treatment import TreatmentType, coerce_treatment_type


def _length(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        lengths = [_length(item) for item in value.values()]
        lengths = [item for item in lengths if item is not None]
        return lengths[0] if lengths else 0
    try:
        return len(value)
    except TypeError:
        return None


@dataclass
class EffectPrediction:
    """Model-independent effect output consumed by the decision layer."""

    unit_id: Any
    treatment_type: TreatmentType | str
    uplift: Any | None = None
    cate: Any | None = None
    y0: Any | None = None
    y1: Any | None = None
    treatment_effects: Dict[Any, Any] = field(default_factory=dict)
    recommended_treatment: Any | None = None
    recommended_effect: Any | None = None
    lower_bound: Any | None = None
    upper_bound: Any | None = None
    propensity: Any | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.treatment_type = coerce_treatment_type(self.treatment_type)
        if self.uplift is None and self.cate is not None:
            self.uplift = self.cate
        if self.cate is None and self.uplift is not None:
            self.cate = self.uplift
        expected = _length(self.unit_id)
        for name in [
            "uplift",
            "cate",
            "y0",
            "y1",
            "recommended_effect",
            "recommended_treatment",
            "lower_bound",
            "upper_bound",
            "propensity",
        ]:
            value_length = _length(getattr(self, name))
            if expected is not None and value_length is not None and value_length != expected:
                raise ValueError(f"EffectPrediction field '{name}' has length {value_length}; expected {expected}.")
        for treatment, effects in self.treatment_effects.items():
            value_length = _length(effects)
            if expected is not None and value_length is not None and value_length != expected:
                raise ValueError(f"EffectPrediction treatment effect '{treatment}' has invalid length.")

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame({"unit_id": np.asarray(self.unit_id)})
        for name in ["uplift", "cate", "y0", "y1", "recommended_treatment", "recommended_effect", "lower_bound", "upper_bound", "propensity"]:
            value = getattr(self, name)
            if value is not None:
                frame[name] = np.asarray(value)
        for treatment, effects in self.treatment_effects.items():
            frame[f"effect_{treatment}"] = np.asarray(effects)
        return frame
