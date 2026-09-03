"""Model-layer entry points and implementations.

Concrete neural implementations keep their historical module names for
backwards compatibility. Registry functions are re-exported here so callers
do not need to know about the application-service package.
"""

from deepuplift.core.registry import (
    MODEL_REGISTRY,
    ModelSpec,
    available_models,
    build_model,
    is_model_available,
    missing_dependencies,
)

__version__ = "1.0.0"

__all__ = [
    "MODEL_REGISTRY",
    "ModelSpec",
    "available_models",
    "build_model",
    "is_model_available",
    "missing_dependencies",
]
