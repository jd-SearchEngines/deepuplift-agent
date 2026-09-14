from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd

from deepuplift.contracts import AssignmentType, CausalDataset, TreatmentType
from deepuplift.data.schema import create_causal_dataset


GIKS_UPSTREAM = "https://github.com/nlokeshiisc/GIKS_release"


@dataclass
class SyntheticContinuousData:
    dataset: CausalDataset
    dose_grid: np.ndarray
    true_response_curves: np.ndarray
    true_effect_curves: np.ndarray
    true_optimal_effect_dose: np.ndarray
    true_optimal_economic_dose: np.ndarray
    seed: int
    outcome_value: float
    license: str = "DeepUplift-generated synthetic data"
    fixed_train_indices: np.ndarray | None = None
    fixed_test_indices: np.ndarray | None = None
    cost_dose_range: float = 20.0

    def subset(self, indices: Any) -> "SyntheticContinuousData":
        ids = np.asarray(indices)
        return SyntheticContinuousData(
            dataset=self.dataset.subset(ids),
            dose_grid=self.dose_grid.copy(),
            true_response_curves=self.true_response_curves[ids],
            true_effect_curves=self.true_effect_curves[ids],
            true_optimal_effect_dose=self.true_optimal_effect_dose[ids],
            true_optimal_economic_dose=self.true_optimal_economic_dose[ids],
            seed=self.seed,
            outcome_value=self.outcome_value,
            license=self.license,
            cost_dose_range=self.cost_dose_range,
        )


def _response_terms(x: np.ndarray, dose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype="float64")
    dose = np.asarray(dose, dtype="float64").reshape(-1)
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    normalized = dose[None, :] / 20.0
    base = 0.35 + 0.16 * x1[:, None] - 0.11 * x2[:, None] + 0.06 * x3[:, None]
    amplitude = 1.20 + 0.35 * np.tanh(x1)
    curvature = 1.15 + 0.20 / (1.0 + np.exp(-x2))
    effect = amplitude[:, None] * normalized - curvature[:, None] * normalized**2
    return base, effect


def continuous_dose_cost(dose: Any) -> np.ndarray:
    values = np.asarray(dose, dtype="float64")
    return 0.015 * values + 0.0025 * values**2


def synthetic_dose_response(
    rows: int = 600,
    seed: int = 42,
    *,
    dose_grid: Any | None = None,
    noise_sd: float = 0.18,
    outcome_value: float = 1.0,
) -> SyntheticContinuousData:
    """Confounded nonlinear dose-response data with exact held-out truth.

    Dose assignment depends on pre-treatment covariates and noisy selection,
    while the response has covariate-varying linear and quadratic dose terms.
    No ground-truth targets are added to the observed CausalDataset.
    """
    if rows < 20:
        raise ValueError("Synthetic continuous benchmarks need at least 20 rows.")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, 3))
    assignment_score = 1.15 * x[:, 0] - 0.75 * x[:, 1] + 0.40 * x[:, 2] + 0.20 * x[:, 2] ** 2
    expected_dose = 20.0 / (1.0 + np.exp(-assignment_score))
    dose = np.clip(expected_dose + rng.normal(0.0, 2.1, rows), 0.0, 20.0)
    grid = np.asarray(dose_grid if dose_grid is not None else np.linspace(0.0, 20.0, 21), dtype="float64")
    if grid.ndim != 1 or len(grid) < 3 or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise ValueError("dose_grid must be a finite, strictly increasing one-dimensional grid with at least three points.")
    features = x
    base, effect = _response_terms(features, dose)
    y = (base[:, 0] + effect[:, 0] + rng.normal(0.0, noise_sd, rows))
    true_base, true_effect = _response_terms(features, grid)
    true_response = true_base + true_effect
    true_opt_effect = grid[np.argmax(true_effect, axis=1)]
    cost_domain_range = max(float(grid[-1] - grid[0]), 1e-6)
    utility = outcome_value * true_effect - continuous_dose_cost(grid * (20.0 / cost_domain_range))[None, :]
    utility[:, np.argmin(np.abs(grid))] = np.maximum(utility[:, np.argmin(np.abs(grid))], 0.0)
    true_opt_economic = grid[np.argmax(utility, axis=1)]
    frame = pd.DataFrame({"unit_id": [f"continuous-{seed}-{i}" for i in range(rows)], "x1": features[:, 0], "x2": features[:, 1], "x3": features[:, 2], "dose": dose, "outcome": y})
    dataset = create_causal_dataset(
        frame,
        feature_cols=["x1", "x2", "x3"],
        treatment_col="dose",
        outcome_col="outcome",
        treatment_type=TreatmentType.CONTINUOUS,
        assignment_type=AssignmentType.OBSERVATIONAL,
        id_column="unit_id",
        metadata={
            "name": "synthetic_continuous_observational",
            "source": "DeepUplift deterministic nonlinear data-generating process",
            "license": "DeepUplift-generated synthetic data",
            "sample_size": int(rows),
            "treatment_range": [float(dose.min()), float(dose.max())],
            "feature_dimension": int(features.shape[1]),
            "seed": int(seed),
            "split": "not split by generator",
            "ground_truth_available": True,
            "assignment_mechanism": "dose = clip(20 * sigmoid(1.15*x1 - 0.75*x2 + 0.40*x3 + 0.20*x3^2 + Normal(0,2.1)), 0, 20)",
            "outcome_mechanism": "covariate-varying concave quadratic dose response plus Gaussian noise",
        },
    )
    return SyntheticContinuousData(dataset, grid, true_response, true_effect, true_opt_effect, true_opt_economic, int(seed), float(outcome_value), cost_dose_range=cost_domain_range)


_UPSTREAMS = {
    "ihdp": f"{GIKS_UPSTREAM}/tree/main/dataset/ihdp",
    "news": f"{GIKS_UPSTREAM}/tree/main/dataset/news",
    "tcga": f"{GIKS_UPSTREAM} (TCGA data is documented for separate download)",
}


def load_continuous_local(
    name: str,
    path: str | Path,
    *,
    treatment_col: str = "dose",
    outcome_col: str = "outcome",
    feature_cols: list[str] | None = None,
    id_column: str | None = "unit_id",
    assignment_type: AssignmentType | str = AssignmentType.OBSERVATIONAL,
) -> CausalDataset:
    """Load a user-provided continuous benchmark CSV; never downloads raw data."""
    key = name.lower()
    if key not in _UPSTREAMS:
        raise ValueError(f"Unknown local continuous benchmark '{name}'; expected IHDP, NEWS, or TCGA.")
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"{name.upper()} continuous data not found at {source_path}; obtain it from {_UPSTREAMS[key]}. Raw data is not redistributed.")
    frame = pd.read_csv(source_path)
    if feature_cols is None:
        excluded = {treatment_col, outcome_col, id_column, "true_effect", "true_outcome"}
        feature_cols = [col for col in frame.columns if col not in excluded and not str(col).startswith("true_")]
    missing = [c for c in [treatment_col, outcome_col, *feature_cols] if c not in frame.columns]
    if missing:
        raise ValueError(f"{name.upper()} normalized CSV is missing columns: {missing}.")
    return create_causal_dataset(
        frame,
        feature_cols=feature_cols,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        treatment_type=TreatmentType.CONTINUOUS,
        assignment_type=assignment_type,
        id_column=id_column if id_column in frame.columns else None,
        metadata={
            "name": key.upper(),
            "source": _UPSTREAMS[key],
            "license": "UPSTREAM_TERMS_UNVERIFIED; source data is user-provided and not redistributed",
            "sample_size": int(len(frame)),
            "treatment_range": [float(pd.to_numeric(frame[treatment_col], errors="coerce").min()), float(pd.to_numeric(frame[treatment_col], errors="coerce").max())],
            "feature_dimension": len(feature_cols),
            "ground_truth_available": False,
            "source_path": str(source_path),
            "raw_data_redistributed": False,
        },
    )


def load_ihdp(path: str | Path, **kwargs: Any) -> CausalDataset:
    if _has_giks_matrix(path):
        return load_giks_continuous_benchmark("ihdp", path, **{k: v for k, v in kwargs.items() if k in {"split_id", "outcome_value"}}).dataset
    return load_continuous_local("ihdp", path, **kwargs)


def load_news(path: str | Path, **kwargs: Any) -> CausalDataset:
    if _has_giks_matrix(path):
        return load_giks_continuous_benchmark("news", path, **{k: v for k, v in kwargs.items() if k in {"split_id", "outcome_value"}}).dataset
    return load_continuous_local("news", path, **kwargs)


def load_tcga(path: str | Path, **kwargs: Any) -> CausalDataset:
    if _has_giks_matrix(path):
        return _load_giks_matrix_dataset("tcga", path)
    return load_continuous_local("tcga", path, **kwargs)


def _has_giks_matrix(path: str | Path) -> bool:
    candidate = Path(path)
    return candidate.is_file() and candidate.suffix == ".pt" or (candidate / "data_matrix.pt").is_file()


def _matrix_from_giks(path: str | Path) -> tuple[np.ndarray, Path]:
    candidate = Path(path)
    matrix_path = candidate if candidate.is_file() else candidate / "data_matrix.pt"
    if not matrix_path.exists():
        raise FileNotFoundError(f"No data_matrix.pt found at {matrix_path}.")
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Loading the upstream GIKS .pt matrix requires PyTorch; install deepuplift[continuous].") from exc
    matrix = torch.load(matrix_path, map_location="cpu", weights_only=True)
    matrix = matrix.detach().cpu().numpy() if hasattr(matrix, "detach") else np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[1] < 4:
        raise ValueError("GIKS data_matrix.pt must contain [dose, features..., outcome].")
    return np.asarray(matrix, dtype="float64"), matrix_path.parent


def _load_giks_matrix_dataset(name: str, path: str | Path) -> CausalDataset:
    matrix, root = _matrix_from_giks(path)
    feature_cols = [f"x{i}" for i in range(matrix.shape[1] - 2)]
    frame = pd.DataFrame(matrix, columns=["dose", *feature_cols, "outcome"])
    frame.insert(0, "unit_id", [f"{name}-{i}" for i in range(len(frame))])
    return create_causal_dataset(
        frame,
        feature_cols=feature_cols,
        treatment_col="dose",
        outcome_col="outcome",
        treatment_type=TreatmentType.CONTINUOUS,
        assignment_type=AssignmentType.OBSERVATIONAL,
        id_column="unit_id",
        metadata={
            "name": name.upper(),
            "source": _UPSTREAMS.get(name, GIKS_UPSTREAM),
            "license": "UPSTREAM_DATA_TERMS_UNVERIFIED; local user-provided data only",
            "sample_size": int(len(matrix)),
            "treatment_range": [float(matrix[:, 0].min()), float(matrix[:, 0].max())],
            "feature_dimension": int(matrix.shape[1] - 2),
            "ground_truth_available": False,
            "source_path": str(root),
            "raw_data_redistributed": False,
        },
    )


class _NumpyArrayUnpickler(pickle.Unpickler):
    """Read NumPy arrays from benchmark pickle files without arbitrary globals."""

    _ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(f"Disallowed global in benchmark array pickle: {module}.{name}")
        if name == "_reconstruct":
            try:
                return importlib.import_module("numpy._core.multiarray")._reconstruct
            except ImportError:
                return importlib.import_module("numpy.core.multiarray")._reconstruct
        return getattr(np, name)


def load_giks_continuous_benchmark(
    name: str,
    path: str | Path,
    *,
    split_id: int = 0,
    outcome_value: float = 1.0,
) -> SyntheticContinuousData:
    """Load official GIKS preprocessed IHDP or NEWS data and curve truth.

    Files are read in place and never copied into DeepUplift. PyTorch tensors
    are loaded with ``weights_only=True``; the NumPy response pickle uses a
    restricted unpickler that allows only ndarray reconstruction.
    """
    key = name.lower()
    if key not in {"ihdp", "news"}:
        raise ValueError("Ground-truth GIKS loader supports IHDP and NEWS; TCGA truth is not bundled.")
    matrix, root = _matrix_from_giks(path)
    response_path = root / f"{key}_response.pkl"
    if not response_path.is_file():
        raise FileNotFoundError(f"No response-curve truth file at {response_path}; only observed rows can be loaded.")
    with response_path.open("rb") as stream:
        response = np.asarray(_NumpyArrayUnpickler(stream).load(), dtype="float64")
    if response.ndim != 2 or response.shape[0] != len(matrix):
        raise ValueError(f"{key.upper()} response truth must have shape (rows, doses).")
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Loading the upstream GIKS evaluation split requires PyTorch.") from exc
    split_path = root / "eval" / str(int(split_id))
    train_path, test_path = split_path / "idx_train.pt", split_path / "idx_test.pt"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(f"No official split {split_id} under {split_path}.")
    train_ids = torch.load(train_path, map_location="cpu", weights_only=True).numpy().astype("int64")
    test_ids = torch.load(test_path, map_location="cpu", weights_only=True).numpy().astype("int64")
    if np.intersect1d(train_ids, test_ids).size or len(np.union1d(train_ids, test_ids)) != len(matrix):
        raise ValueError("Official benchmark train/test indices must partition every dataset row exactly once.")
    response_grid = np.arange(0.01, 1.0, 1.0 / response.shape[1], dtype="float64")
    if len(response_grid) != response.shape[1]:
        raise ValueError("Unexpected response grid in the official GIKS benchmark.")
    dose_grid = np.concatenate([[0.0], response_grid])
    true_response = np.column_stack([response[:, 0], response])
    true_effect = true_response - true_response[:, :1]
    true_opt_effect = dose_grid[np.argmax(true_effect, axis=1)]
    cost_domain_range = float(dose_grid[-1] - dose_grid[0])
    utility = outcome_value * true_effect - continuous_dose_cost(dose_grid * (20.0 / cost_domain_range))[None, :]
    utility[:, 0] = np.maximum(utility[:, 0], 0.0)
    true_opt_economic = dose_grid[np.argmax(utility, axis=1)]
    feature_cols = [f"x{i}" for i in range(matrix.shape[1] - 2)]
    frame = pd.DataFrame(matrix, columns=["dose", *feature_cols, "outcome"])
    frame.insert(0, "unit_id", [f"{key}-{i}" for i in range(len(frame))])
    dataset = create_causal_dataset(
        frame,
        feature_cols=feature_cols,
        treatment_col="dose",
        outcome_col="outcome",
        treatment_type=TreatmentType.CONTINUOUS,
        assignment_type=AssignmentType.OBSERVATIONAL,
        id_column="unit_id",
        metadata={
            "name": key.upper(),
            "source": _UPSTREAMS[key],
            "license": "UPSTREAM_DATA_TERMS_UNVERIFIED; official repo data used locally, not redistributed",
            "sample_size": int(len(matrix)),
            "treatment_range": [float(matrix[:, 0].min()), float(matrix[:, 0].max())],
            "feature_dimension": len(feature_cols),
            "ground_truth_available": True,
            "split": f"official GIKS split {split_id}",
            "source_path": str(root),
            "raw_data_redistributed": False,
            "ground_truth_baseline_note": "The published response grid starts at dose 0.01; its first response value is used as the dose-0 boundary estimate.",
        },
    )
    return SyntheticContinuousData(
        dataset,
        dose_grid,
        true_response,
        true_effect,
        true_opt_effect,
        true_opt_economic,
        int(split_id),
        float(outcome_value),
        license="UPSTREAM_DATA_TERMS_UNVERIFIED",
        fixed_train_indices=train_ids,
        fixed_test_indices=test_ids,
        cost_dose_range=cost_domain_range,
    )


__all__ = ["SyntheticContinuousData", "synthetic_dose_response", "continuous_dose_cost", "load_continuous_local", "load_ihdp", "load_news", "load_tcga"]
