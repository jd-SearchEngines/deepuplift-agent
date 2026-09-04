from __future__ import annotations

from typing import Any

from deepuplift.contracts import CausalDataset


def fit_model(model: Any, dataset: CausalDataset) -> Any:
    """Common runtime hook for native and future third-party adapters."""

    return model.fit(dataset)
