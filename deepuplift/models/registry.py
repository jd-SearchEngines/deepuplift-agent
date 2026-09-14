from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .adapters import CausalMLAdapter, CausalForestDMLAdapter, EconMLAdapter, SkLiftAdapter
from .binary import DRLearner, IPWLearner, RLearner, SLearner, TLearner, XLearner
from .capabilities import DEFAULT_BINARY_CAPABILITIES, ModelCapabilities
from .continuous import CCPFNAdapter, DRNet, DoseResponseGBM, GIKSEstimator, VCNet
from .continuous.transtee import TransTEEContinuousAdapter
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
    requires_network: bool = False
    requires_weights: bool = False
    extra: str | None = None
    paper: str | None = None
    license: str | None = None


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
    "DoseResponseGBM": ModelSpec("DoseResponseGBM", DoseResponseGBM, ModelCapabilities(supports_binary=False, supports_continuous=True, supports_classification=True, supports_regression=True), maturity="EXPERIMENTAL", notes="Reference dose-response implementation; not industrial continuous-treatment evidence.", license="MIT (DeepUplift native implementation)."),
    "DRNet": ModelSpec("DRNet", DRNet, ModelCapabilities(supports_binary=False, supports_continuous=True, supports_classification=False, supports_regression=True, supports_sample_weight=True, backend="pytorch"), backend="native", dependency="torch", maturity="OPTIONAL/EXPERIMENTAL", extra="continuous", paper="Nie et al., ICLR 2021 (DRNet baseline)", license="MIT (DeepUplift native implementation); source project license undeclared; no code copied."),
    "VCNet": ModelSpec("VCNet", VCNet, ModelCapabilities(supports_binary=False, supports_continuous=True, supports_classification=False, supports_regression=True, supports_sample_weight=True, backend="pytorch"), backend="native", dependency="torch", maturity="OPTIONAL/EXPERIMENTAL", extra="continuous", paper="Nie et al., ICLR 2021", license="MIT (DeepUplift native implementation); source project license undeclared; no code copied."),
    "GIKS-VCNet": ModelSpec("GIKS-VCNet", GIKSEstimator, ModelCapabilities(supports_binary=False, supports_continuous=True, supports_classification=False, supports_regression=True, supports_sample_weight=True, backend="pytorch"), backend="native", dependency="torch", maturity="OPTIONAL/EXPERIMENTAL", extra="giks", paper="Continuous Treatment Effect Estimation Using Gradient Interpolation and Kernel Smoothing (AAAI 2024)", license="MIT (DeepUplift implementation); Apache-2.0 reference; no code copied."),
    "GIKS-DRNet": ModelSpec("GIKS-DRNet", GIKSEstimator, ModelCapabilities(supports_binary=False, supports_continuous=True, supports_classification=False, supports_regression=True, supports_sample_weight=True, backend="pytorch"), backend="native", dependency="torch", maturity="OPTIONAL/EXPERIMENTAL", extra="giks", paper="Continuous Treatment Effect Estimation Using Gradient Interpolation and Kernel Smoothing (AAAI 2024)", license="MIT (DeepUplift implementation); Apache-2.0 reference; no code copied."),
    "CCPFN": ModelSpec("CCPFN", CCPFNAdapter, ModelCapabilities(supports_binary=False, supports_continuous=True, supports_classification=False, supports_regression=True, backend="ccpfn"), backend="ccpfn", dependency="ccpfn", maturity="OPTIONAL/FRONTIER_EXPERIMENTAL", notes="Published pretrained CEPO inference adapter; first use downloads 219 MB of weights unless cached.", requires_network=True, requires_weights=True, extra="ccpfn", paper="Causal Foundation Models with Continuous Treatments (2026)", license="Apache-2.0"),
    "TransTEE": ModelSpec("TransTEE", TransTEEContinuousAdapter, ModelCapabilities(supports_binary=False, supports_continuous=True, supports_classification=False, supports_regression=True, backend="transtee"), backend="transtee", dependency=None, maturity="OPTIONAL_NOT_VALIDATED", runnable=False, notes="Official repository has no installable PyPI package; continuous/dosage integration is not validated in this release.", extra=None, paper="Exploring Transformer Backbones for Heterogeneous Treatment Effect Estimation (2022)", license="MIT"),
    "CausalForestDML": ModelSpec("CausalForestDML", CausalForestDMLAdapter, ModelCapabilities(backend="econml", requires_propensity=True), "econml", "econml", "OPTIONAL", True, "EconML internally manages nuisance; provenance is recorded by the adapter."),
    "CausalMLUpliftTree": ModelSpec("CausalMLUpliftTree", CausalMLAdapter, ModelCapabilities(backend="causalml"), "causalml", "causalml", "OPTIONAL", True, "CausalML UpliftTreeClassifier adapter."),
    "CausalMLUpliftRandomForest": ModelSpec("CausalMLUpliftRandomForest", CausalMLAdapter, ModelCapabilities(backend="causalml"), "causalml", "causalml", "OPTIONAL", True, "CausalML UpliftRandomForestClassifier adapter."),
    "SkLiftTwoModels": ModelSpec("SkLiftTwoModels", SkLiftAdapter, ModelCapabilities(backend="scikit-uplift"), "scikit-uplift", "sklift", "OPTIONAL", False, "Interface-only; no silent native fallback."),
}

for _deep_name in ("TarNet", "CFRNet", "DragonNet", "CEVAE", "GANITE", "DESCN", "EFIN", "EUEN", "EEUEN", "ContrastiveUpliftNet"):
    MODEL_REGISTRY[_deep_name] = ModelSpec(_deep_name, lambda **kwargs: None, ModelCapabilities(), "legacy", None, "EXPERIMENTAL", False, "Deep implementation is retained for research compatibility; it is not a v0.3 runnable EffectPrediction adapter.")

ALIASES = {"SLearnerGBM": "S-Learner", "TLearnerGBM": "T-Learner", "XLearnerGBM": "X-Learner", "DRLearnerGBM": "DR-Learner", "EconMLCausalForest": "CausalForestDML", "CausalForest": "CausalForestDML", "CausalMLUpliftRF": "CausalMLUpliftRandomForest"}
ALIASES["GIKS"] = "GIKS-VCNet"


def _canonical(name: str) -> str:
    return ALIASES.get(name, name)


def model_info(model_name: str) -> dict[str, Any]:
    canonical = _canonical(model_name)
    if canonical not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available_models()}.")
    spec = MODEL_REGISTRY[canonical]
    missing = missing_dependencies(canonical)
    installed = (not missing) if spec.dependency is not None else (True if spec.backend == "native" else None)
    weights_available = None
    if spec.requires_weights:
        from .continuous.ccpfn import local_weights_available
        weights_available = local_weights_available()
    return {"name": spec.name, "backend": spec.backend, "dependency": spec.dependency, "dependency_installed": installed, "requires_torch": spec.dependency == "torch" or spec.name == "CCPFN", "maturity": spec.maturity, "runnable": spec.runnable and installed is not False, "missing_dependencies": missing, "weights_required": spec.requires_weights, "weights_available": weights_available, "requires_network": spec.requires_network and weights_available is not True, "extra": spec.extra, "paper": spec.paper, "license": spec.license, "capabilities": spec.capabilities.to_dict(), "notes": spec.notes}


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
    if canonical == "CCPFN":
        return spec.factory(random_state=random_state, **kwargs)
    if canonical == "GIKS-VCNet":
        kwargs.setdefault("base_model", "VCNet")
    elif canonical == "GIKS-DRNet":
        kwargs.setdefault("base_model", "DRNet")
    if canonical in {"DRNet", "VCNet"}:
        kwargs.setdefault("task", "regression")
    if canonical in {"DRNet", "VCNet", "GIKS-VCNet", "GIKS-DRNet"}:
        return spec.factory(random_state=random_state, **kwargs)
    if spec.backend != "native":
        return spec.factory(estimator_name=canonical, **kwargs)
    return spec.factory(task=task, random_state=random_state, **kwargs)


__all__ = ["MODEL_REGISTRY", "ModelSpec", "available_models", "model_info", "build_model", "is_model_available", "missing_dependencies"]
