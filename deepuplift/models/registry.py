from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .adapters import CausalMLAdapter, EconMLAdapter, SkLiftAdapter
from .binary import DRLearner, SLearner, TLearner, XLearner
from .capabilities import DEFAULT_BINARY_CAPABILITIES, ModelCapabilities
from .continuous import DoseResponseGBM
from .multi import MultiTreatmentOutcomeModel


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[..., Any]
    capabilities: ModelCapabilities
    backend: str = "native"
    dependency: str | None = None


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "S-Learner": ModelSpec("S-Learner", SLearner, DEFAULT_BINARY_CAPABILITIES),
    "T-Learner": ModelSpec("T-Learner", TLearner, DEFAULT_BINARY_CAPABILITIES),
    "X-Learner": ModelSpec("X-Learner", XLearner, DEFAULT_BINARY_CAPABILITIES),
    "DR-Learner": ModelSpec("DR-Learner", DRLearner, ModelCapabilities(requires_propensity=True)),
    "MultiTreatmentOutcome": ModelSpec(
        "MultiTreatmentOutcome",
        MultiTreatmentOutcomeModel,
        ModelCapabilities(supports_binary=False, supports_multi=True, supports_classification=True, supports_regression=True),
    ),
    "DoseResponseGBM": ModelSpec(
        "DoseResponseGBM",
        DoseResponseGBM,
        ModelCapabilities(supports_binary=False, supports_continuous=True, supports_classification=True, supports_regression=True),
    ),
    "EconMLCausalForest": ModelSpec(
        "EconMLCausalForest", EconMLAdapter, ModelCapabilities(backend="econml"), "econml", "econml"
    ),
    "CausalMLUpliftTree": ModelSpec(
        "CausalMLUpliftTree", CausalMLAdapter, ModelCapabilities(backend="causalml"), "causalml", "causalml"
    ),
    "SkLiftTwoModels": ModelSpec(
        "SkLiftTwoModels", SkLiftAdapter, ModelCapabilities(backend="scikit-uplift"), "scikit-uplift", "sklift"
    ),
}

ALIASES = {
    "SLearnerGBM": "S-Learner",
    "TLearnerGBM": "T-Learner",
    "XLearnerGBM": "X-Learner",
    "DRLearnerGBM": "DR-Learner",
}


def _canonical(name: str) -> str:
    return ALIASES.get(name, name)


def missing_dependencies(model_name: str) -> list[str]:
    spec = MODEL_REGISTRY.get(_canonical(model_name))
    if spec is None or spec.dependency is None:
        return []
    import importlib.util

    return [] if importlib.util.find_spec(spec.dependency) else [spec.dependency]


def is_model_available(model_name: str) -> bool:
    return not missing_dependencies(model_name)


def available_models(*, only_available: bool = False, backend: str | None = None) -> list[str]:
    names = [name for name, spec in MODEL_REGISTRY.items() if backend is None or spec.backend == backend]
    if only_available:
        names = [name for name in names if is_model_available(name)]
    return sorted(names)


def build_model(model_name: str, *, task: str = "classification", random_state: int = 42, **kwargs: Any):
    canonical = _canonical(model_name)
    if canonical not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available_models()}.")
    spec = MODEL_REGISTRY[canonical]
    missing = missing_dependencies(canonical)
    if missing:
        raise ImportError(f"{canonical} requires optional dependency: {', '.join(missing)}")
    if canonical in {"EconMLCausalForest", "CausalMLUpliftTree", "SkLiftTwoModels"}:
        return spec.factory(estimator_name=canonical, **kwargs)
    return spec.factory(task=task, random_state=random_state, **kwargs)
