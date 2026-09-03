from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class UpliftConfig:
    treatment_col: str
    outcome_col: str
    feature_cols: List[str]
    model_name: str = "TarNet"
    task: str = "classification"
    test_size: float = 0.2
    valid_perc: Optional[float] = 0.2
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-4
    random_state: int = 42
    tensorboard: bool = False
    artifacts_dir: str = "runs"
    run_name: Optional[str] = None
    max_rows: Optional[int] = None
    bootstrap_samples: int = 0
    sensitivity_samples: int = 0
    policy_contact_cost: float = 0.0
    policy_conversion_value: float = 1.0
    control_value: Optional[Any] = None
    treated_value: Optional[Any] = None
    categorical_cols: Optional[List[str]] = None
    model_params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "UpliftConfig":
        return cls(**values)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UpliftRunResult:
    run_id: str
    run_dir: str
    config: Dict[str, Any]
    train_history: List[Dict[str, Any]]
    eval_metrics: Dict[str, Any]
    curves: Dict[str, Any]
    artifacts: Dict[str, str]
    preview: List[Dict[str, Any]]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
