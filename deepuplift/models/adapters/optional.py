from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

from deepuplift.contracts import CausalDataset, EffectPrediction
from deepuplift.models.capabilities import ModelCapabilities


@dataclass
class OptionalBackendAdapter:
    """Explicit boundary for estimators owned by third-party libraries.

    The adapter does not silently replace a missing dependency with a native
    estimator. This keeps benchmark provenance honest and makes installation
    errors actionable.
    """

    estimator_name: str
    backend: str
    dependency: str
    capabilities: ModelCapabilities
    estimator: Any | None = None

    @property
    def name(self) -> str:
        return self.estimator_name

    @property
    def available(self) -> bool:
        return importlib.util.find_spec(self.dependency) is not None

    def missing_dependency(self) -> str | None:
        return None if self.available else self.dependency

    def fit(self, dataset: CausalDataset) -> "OptionalBackendAdapter":
        if not self.available:
            raise ImportError(f"{self.estimator_name} requires optional dependency '{self.dependency}'.")
        if self.estimator is None:
            raise NotImplementedError(
                f"{self.estimator_name} is a backend boundary; provide an estimator implementation for this version."
            )
        self.estimator.fit(dataset)
        return self

    def predict(self, dataset: CausalDataset) -> EffectPrediction:
        if self.estimator is None:
            raise NotImplementedError(f"{self.estimator_name} has no configured estimator implementation.")
        return self.estimator.predict(dataset)


class EconMLAdapter(OptionalBackendAdapter):
    def __init__(self, estimator_name: str = "EconML", estimator: Any | None = None):
        super().__init__(estimator_name, "econml", "econml", ModelCapabilities(backend="econml"), estimator)


class CausalMLAdapter(OptionalBackendAdapter):
    def __init__(self, estimator_name: str = "CausalML", estimator: Any | None = None):
        super().__init__(estimator_name, "causalml", "causalml", ModelCapabilities(backend="causalml"), estimator)


class SkLiftAdapter(OptionalBackendAdapter):
    def __init__(self, estimator_name: str = "scikit-uplift", estimator: Any | None = None):
        super().__init__(estimator_name, "scikit-uplift", "sklift", ModelCapabilities(backend="scikit-uplift"), estimator)
