from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .adapters import CausalMLAdapter, EconMLAdapter, SkLiftAdapter
from .binary import DRLearner, IPWLearner, RLearner, SLearner, TLearner, XLearner
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
    maturity: str = "STABLE"
    runnable: bool = True
    notes: str = ""


def _native(cls, outcome_model: str = "gbm"):
    return lambda *, task="classification", random_state=42, **kwargs: cls(task=task, random_state=random_state, outcome_model=outcome_model, **kwargs)


_rf_capabilities = ModelCapabilities(supports_sample_weight=True)
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "S-Learner": ModelSpec("S-Learner", _native(SLearner), DEFAULT_BINARY_CAPABILITIES),
    "T-Learner": ModelSpec("T-Learner", _native(TLearner), DEFAULT_BINARY_CAPABILITIES),
    "X-Learner": ModelSpec("X-Learner", _native(XLearner), DEFAULT_BINARY_CAPABILITIES),
    "DR-Learner": ModelSpec("DR-Learner", _native(DRLearner), ModelCapabilities(requires_propensity=True)),
    "R-Learner": ModelSpec("R-Learner", _native(RLearner), ModelCapabilities(requires_propensity=True)),
    "IPW-Learner": ModelSpec("IPW-Learner", _native(IPWLearner), ModelCapabilities(requires_propensity=True, supports_sample_weight=True)),
    "S-Learner-RF": ModelSpec("S-Learner-RF", _native(SLearner, "random_forest"), _rf_capabilities, notes="S-Learner with RandomForest outcome models"),
    "T-Learner-RF": ModelSpec("T-Learner-RF", _native(TLearner, "random_forest"), _rf_capabilities),
    "X-Learner-RF": ModelSpec("X-Learner-RF", _native(XLearner, "random_forest"), _rf_capabilities),
    "DR-Learner-RF": ModelSpec("DR-Learner-RF", _native(DRLearner, "random_forest"), ModelCapabilities(requires_propensity=True, supports_sample_weight=True)),
    "MultiTreatmentOutcome": ModelSpec("MultiTreatmentOutcome", MultiTreatmentOutcomeModel, ModelCapabilities(supports_binary=False, supports_multi=True, supports_classification=True, supports_regression=True), maturity="EXPERIMENTAL"),
    "DoseResponseGBM": ModelSpec("DoseResponseGBM", DoseResponseGBM, ModelCapabilities(supports_binary=False, supports_continuous=True, supports_classification=True, supports_regression=True), maturity="EXPERIMENTAL", notes="Reference dose-response implementation; not industrial continuous-treatment evidence."),
    "EconMLCausalForest": ModelSpec("EconMLCausalForest", EconMLAdapter, ModelCapabilities(backend="econml", requires_propensity=True), "econml", "econml", "OPTIONAL", False, "Interface-only until an EconML estimator adapter is configured."),
    "CausalMLUpliftTree": ModelSpec("CausalMLUpliftTree", CausalMLAdapter, ModelCapabilities(backend="causalml"), "causalml", "causalml", "OPTIONAL", False, "Interface-only until a CausalML estimator adapter is configured."),
    "SkLiftTwoModels": ModelSpec("SkLiftTwoModels", SkLiftAdapter, ModelCapabilities(backend="scikit-uplift"), "scikit-uplift", "sklift", "OPTIONAL", False, "Interface-only; no silent native fallback."),
}

for _deep_name in ("TarNet", "CFRNet", "DragonNet", "CEVAE", "GANITE", "DESCN", "EFIN", "EUEN", "EEUEN", "ContrastiveUpliftNet"):
    MODEL_REGISTRY[_deep_name] = ModelSpec(_deep_name, lambda **kwargs: None, ModelCapabilities(), "legacy", None, "EXPERIMENTAL", False, "Deep implementation is retained for research compatibility; it is not a v0.3 runnable EffectPrediction adapter.")

ALIASES = {"SLearnerGBM": "S-Learner", "TLearnerGBM": "T-Learner", "XLearnerGBM": "X-Learner", "DRLearnerGBM": "DR-Learner"}


def _canonical(name: str) -> str:
    return ALIASES.get(name, name)


def model_info(model_name: str) -> dict[str, Any]:
    canonical = _canonical(model_name)
    if canonical not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available_models()}.")
    spec = MODEL_REGISTRY[canonical]
    return {"name": spec.name, "backend": spec.backend, "dependency": spec.dependency, "maturity": spec.maturity, "runnable": spec.runnable and not missing_dependencies(canonical), "missing_dependencies": missing_dependencies(canonical), "capabilities": spec.capabilities.to_dict(), "notes": spec.notes}


def missing_dependencies(model_name: str) -> list[str]:
    spec = MODEL_REGISTRY.get(_canonical(model_name))
    if spec is None or spec.dependency is None:
        return []
    import importlib.util
    return [] if importlib.util.find_spec(spec.dependency) else [spec.dependency]


def is_model_available(model_name: str) -> bool:
    spec = MODEL_REGISTRY.get(_canonical(model_name))
    return bool(spec and spec.runnable and not missing_dependencies(model_name))


def available_models(*, only_available: bool = False, backend: str | None = None, runnable_only: bool = False) -> list[str]:
    names = [name for name, spec in MODEL_REGISTRY.items() if (backend is None or spec.backend == backend) and (not runnable_only or spec.runnable)]
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
    if not spec.runnable:
        raise NotImplementedError(f"{canonical} is {spec.maturity} and interface-only: {spec.notes}")
    if spec.backend != "native":
        return spec.factory(estimator_name=canonical, **kwargs)
    return spec.factory(task=task, random_state=random_state, **kwargs)


__all__ = ["MODEL_REGISTRY", "ModelSpec", "available_models", "model_info", "build_model", "is_model_available", "missing_dependencies"]
