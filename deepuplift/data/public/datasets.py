from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deepuplift.contracts import AssignmentType, CausalDataset, TreatmentType
from deepuplift.data.schema import create_causal_dataset


def synthetic_ground_truth(rows: int = 2000, seed: int = 42) -> CausalDataset:
    """Deterministic DGP with observed confounding and known unit-level effect."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=rows), rng.normal(size=rows)
    true_effect = 0.08 + 0.18 * (x1 > 0) - 0.05 * x2
    propensity = 1 / (1 + np.exp(-(0.7 * x1 - 0.5 * x2)))
    treatment = rng.binomial(1, propensity)
    baseline = 0.25 + 0.15 * x1 + 0.08 * x2
    outcome = np.clip(baseline + treatment * true_effect + rng.normal(0, 0.08, rows), 0, 1)
    frame = pd.DataFrame({"unit_id": [f"synthetic-{i}" for i in range(rows)], "x1": x1, "x2": x2, "treatment": treatment, "outcome": outcome, "true_effect": true_effect})
    return create_causal_dataset(frame, feature_cols=["x1", "x2"], treatment_col="treatment", outcome_col="outcome", treatment_type=TreatmentType.BINARY, assignment_type=AssignmentType.OBSERVATIONAL, id_column="unit_id", metadata={"name": "synthetic_ground_truth", "source": "DeepUplift deterministic DGP", "true_effect_col": "true_effect", "seed": seed})


def _from_csv(path: str | Path, *, name: str, feature_cols: list[str], treatment_col: str, outcome_col: str, assignment_type: AssignmentType, id_column: str | None, metadata: dict[str, Any]) -> CausalDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} data was not found at {path}. Download it from the documented upstream source; raw data is not redistributed.")
    frame = pd.read_csv(path)
    missing = [c for c in [*feature_cols, treatment_col, outcome_col] if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} loader missing columns: {missing}")
    return create_causal_dataset(frame, feature_cols=feature_cols, treatment_col=treatment_col, outcome_col=outcome_col, treatment_type=TreatmentType.BINARY, assignment_type=assignment_type, id_column=id_column, metadata={"name": name, **metadata, "source_path": str(path), "raw_data_redistributed": False})


def load_hillstrom(path: str | Path) -> CausalDataset:
    frame = pd.read_csv(path)
    treatment_map = {"No E-Mail": 0, "Mens E-Mail": 1, "Womens E-Mail": 1}
    if "segment" in frame and frame["segment"].dtype == object:
        frame["treatment"] = frame["segment"].map(treatment_map)
    elif "treatment" not in frame:
        raise ValueError("Hillstrom loader needs segment or treatment column.")
    if "conversion" not in frame:
        raise ValueError("Hillstrom loader needs conversion outcome column.")
    feature_cols = [c for c in ["history_segment", "zip_code", "channel", "mens", "womens", "newbie", "recency", "history"] if c in frame.columns]
    return create_causal_dataset(frame, feature_cols=feature_cols, treatment_col="treatment", outcome_col="conversion", treatment_type=TreatmentType.BINARY, assignment_type=AssignmentType.RANDOMIZED, id_column="customer_id" if "customer_id" in frame else None, metadata={"name": "hillstrom", "source": "Hillstrom MineThatData challenge", "license": "Upstream dataset terms apply", "raw_data_redistributed": False, "treatment_mapping": treatment_map})


def load_criteo(path: str | Path, *, sample_rows: int | None = None, seed: int = 42) -> CausalDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Criteo data was not found at {path}; raw data is not redistributed.")
    frame = pd.read_csv(path, nrows=sample_rows)
    if sample_rows is not None and len(frame) > sample_rows:
        frame = frame.sample(sample_rows, random_state=seed)
    treatment_col = "treatment" if "treatment" in frame else "treatment_group"
    outcome_col = "conversion" if "conversion" in frame else "visit" if "visit" in frame else "outcome"
    if treatment_col not in frame or outcome_col not in frame:
        raise ValueError("Criteo loader expects treatment/treatment_group and conversion/visit/outcome columns.")
    excluded = {treatment_col, outcome_col, "id", "user_id"}
    features = [c for c in frame.columns if c not in excluded and frame[c].nunique(dropna=False) > 1]
    return create_causal_dataset(frame, feature_cols=features, treatment_col=treatment_col, outcome_col=outcome_col, treatment_type=TreatmentType.BINARY, assignment_type=AssignmentType.RANDOMIZED, id_column="user_id" if "user_id" in frame else None, metadata={"name": "criteo", "source": "Official Criteo uplift dataset", "license": "Upstream dataset terms apply", "mode": "sample" if sample_rows else "full", "sample_rows": sample_rows, "raw_data_redistributed": False})


def load_retail(path: str | Path, *, name: str = "retail") -> CausalDataset:
    path = Path(path)
    frame = pd.read_csv(path)
    treatment_col = "treatment" if "treatment" in frame else "promo" if "promo" in frame else "campaign"
    outcome_col = "outcome" if "outcome" in frame else "sales" if "sales" in frame else "revenue"
    if treatment_col not in frame or outcome_col not in frame:
        raise ValueError("Retail loader expects treatment/promo/campaign and outcome/sales/revenue columns.")
    features = [c for c in frame.columns if c not in {treatment_col, outcome_col, "id", "user_id"}]
    return create_causal_dataset(frame, feature_cols=features, treatment_col=treatment_col, outcome_col=outcome_col, treatment_type=TreatmentType.BINARY, assignment_type=AssignmentType.OBSERVATIONAL, id_column="user_id" if "user_id" in frame else None, metadata={"name": name, "source": "User-provided retail file; choose Lenta or X5 upstream data", "license": "Upstream dataset terms apply", "raw_data_redistributed": False})


def load_public_dataset(name: str, path: str | Path | None = None, **kwargs: Any) -> CausalDataset:
    key = name.lower().replace("-", "_")
    if key in {"synthetic", "synthetic_ground_truth"}:
        return synthetic_ground_truth(**kwargs)
    if path is None:
        raise ValueError(f"{name} requires a local path; public raw data is never downloaded implicitly.")
    if key == "hillstrom":
        return load_hillstrom(path)
    if key == "criteo":
        return load_criteo(path, **kwargs)
    if key in {"retail", "lenta", "x5"}:
        return load_retail(path, name=key)
    raise ValueError(f"Unknown public dataset '{name}'.")


def controlled_observational_stress_test(dataset: CausalDataset, *, seed: int = 42, strength: float = 1.0) -> CausalDataset:
    """Turn an RCT frame into a controlled confounding stress test.

    The original randomized assignment is retained for audit; the returned
    treatment is newly sampled from an X-dependent mechanism and is not real
    observational ground truth.
    """
    rng = np.random.default_rng(seed)
    frame = dataset.to_pandas().reset_index(drop=True).copy()
    numeric = frame[dataset.feature_cols].select_dtypes(include=["number"])
    score = numeric.fillna(0).to_numpy(dtype="float64").sum(axis=1) if not numeric.empty else np.zeros(len(frame))
    score = (score - score.mean()) / (score.std() or 1.0)
    biased_probability = 1 / (1 + np.exp(-strength * score))
    frame["original_randomized_treatment"] = frame[dataset.treatment_col].to_numpy()
    frame[dataset.treatment_col] = rng.binomial(1, biased_probability)
    metadata = {**dataset.metadata, "name": f"{dataset.metadata.get('name', 'dataset')}_controlled_observational_stress", "source": "Controlled Observational Stress Test transformation", "stress_test": {"seed": seed, "strength": strength, "mechanism": "logistic propensity from pre-treatment numeric features", "original_treatment_col": "original_randomized_treatment", "not_real_observational_ground_truth": True}}
    return create_causal_dataset(frame, feature_cols=dataset.feature_cols, treatment_col=dataset.treatment_col, outcome_col=dataset.outcome_col, treatment_type=dataset.treatment_type, assignment_type=AssignmentType.OBSERVATIONAL, id_column=dataset.id_column, metadata=metadata)


__all__ = ["synthetic_ground_truth", "load_hillstrom", "load_criteo", "load_retail", "load_public_dataset", "controlled_observational_stress_test"]
