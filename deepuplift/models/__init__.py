"""Model-layer entry points and implementations."""

from .registry import (
    MODEL_REGISTRY,
    ModelSpec,
    available_models,
    build_model,
    is_model_available,
    missing_dependencies,
    model_info,
)
from .capabilities import ModelCapabilities

__version__ = "0.4.0a1"

__all__ = [
    "MODEL_REGISTRY",
    "ModelCapabilities",
    "ModelSpec",
    "available_models",
    "build_model",
    "is_model_available",
    "missing_dependencies",
    "model_info",
]
