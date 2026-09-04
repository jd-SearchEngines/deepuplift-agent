from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass
class PolicyResult:
    """Selected treatment per unit plus aggregate business outcomes."""

    rows: List[Dict[str, Any]]
    summary: Dict[str, Any]
    policy_name: str = "uplift_policy"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
