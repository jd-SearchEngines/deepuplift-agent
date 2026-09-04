from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class ModelCapabilities:
    """Machine-readable model boundary used by discovery and routing."""

    supports_binary: bool = True
    supports_multi: bool = False
    supports_continuous: bool = False
    supports_classification: bool = True
    supports_regression: bool = True
    supports_sample_weight: bool = False
    supports_out_of_core: bool = False
    requires_propensity: bool = False
    supports_observational: bool = True
    supports_uncertainty: bool = False
    supports_large_data: bool = False
    backend: str = "native"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_BINARY_CAPABILITIES = ModelCapabilities()
