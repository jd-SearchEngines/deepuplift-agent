from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class DataDiagnostics:
    sample_size: int
    treatment_distribution: Dict[str, Any] = field(default_factory=dict)
    missingness: Dict[str, Any] = field(default_factory=dict)
    feature_balance: Dict[str, Any] = field(default_factory=dict)
    propensity_summary: Dict[str, Any] = field(default_factory=dict)
    overlap_summary: Dict[str, Any] = field(default_factory=dict)
    leakage_warnings: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    readiness: str = "REVIEW"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
