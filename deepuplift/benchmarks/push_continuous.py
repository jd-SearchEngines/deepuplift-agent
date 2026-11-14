"""Push-context semi-synthetic continuous-frequency benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from deepuplift.contracts import AssignmentType, TreatmentType
from deepuplift.data.public.mrt import MRTDataset, load_mybpmylife_mrt
from deepuplift.data.schema import create_causal_dataset
from .continuous.datasets import SyntheticContinuousData, continuous_dose_cost
from .continuous.suite import run_continuous_suite
from .push_semisynthetic import _context_features


def generate_push_continuous(
    mrt: MRTDataset | str,
    *,
    shape: str = "fatigue",
    rows: int = 600,
    seed: int = 42,
    dose_grid: Any | None = None,
    noise_sd: float = 0.18,
    outcome_value: float = 1.0,
) -> SyntheticContinuousData:
    """Generate a known-truth frequency-dose DGP from MRT pre-treatment context."""

    if shape not in {"monotone", "saturating", "fatigue"}:
        raise ValueError("shape must be monotone, saturating, or fatigue")
    source = mrt if isinstance(mrt, MRTDataset) else load_mybpmylife_mrt(mrt)
    context, source_rows = _context_features(source, rows=rows, seed=seed)
    x = context.to_numpy(dtype="float64")
    rng = np.random.default_rng(seed + 31)
    grid = np.asarray(dose_grid if dose_grid is not None else np.linspace(0.0, 20.0, 21), dtype="float64")
    x0 = x[:, 0]
    x1 = x[:, min(1, x.shape[1] - 1)]
    x2 = x[:, min(2, x.shape[1] - 1)]
    baseline = 0.45 + 0.12 * x0 - 0.08 * x1 + 0.04 * x2
    amplitude = 0.75 + 0.18 * np.tanh(x0) - 0.08 * x1
    normalized = grid[None, :] / 20.0
    if shape == "monotone":
        effect = amplitude[:, None] * normalized
        assignment_center = 0.35 + 0.75 * x0 - 0.45 * x1
    elif shape == "saturating":
        effect = amplitude[:, None] * (1.0 - np.exp(-grid[None, :] / 6.0))
        assignment_center = 0.25 + 0.80 * x0 - 0.55 * x1
    else:
        curvature = 0.95 + 0.18 / (1.0 + np.exp(-x2))
        effect = amplitude[:, None] * normalized - curvature[:, None] * normalized**2
        assignment_center = 0.20 + 0.90 * x0 - 0.60 * x1 + 0.20 * x2**2
    expected = 20.0 / (1.0 + np.exp(-assignment_center))
    dose = np.clip(expected + rng.normal(0.0, 2.0, len(x)), 0.0, 20.0)
    if len(grid) > 2:
        right_idx = np.clip(np.searchsorted(grid, dose, side="right"), 1, len(grid) - 1)
        left_idx = right_idx - 1
        fraction = (dose - grid[left_idx]) / np.maximum(grid[right_idx] - grid[left_idx], 1e-8)
        row_idx = np.arange(len(x))
        observed_effect = effect[row_idx, left_idx] + fraction * (effect[row_idx, right_idx] - effect[row_idx, left_idx])
    else:
        observed_effect = effect[:, 0]
    observed_y = baseline + observed_effect + rng.normal(0.0, noise_sd, len(x))
    true_response = baseline[:, None] + effect
    true_effect = true_response - true_response[:, :1]
    true_opt_effect = grid[np.argmax(true_effect, axis=1)]
    utility = outcome_value * true_effect - continuous_dose_cost(grid)[None, :]
    utility[:, 0] = np.maximum(utility[:, 0], 0.0)
    true_opt_economic = grid[np.argmax(utility, axis=1)]
    frame = context.copy()
    frame.insert(0, "unit_id", [f"push-continuous-{seed}-{i}" for i in range(len(x))])
    frame["dose"] = dose
    frame["outcome"] = observed_y
    dataset = create_causal_dataset(
        frame,
        feature_cols=list(context.columns),
        treatment_col="dose",
        outcome_col="outcome",
        treatment_type=TreatmentType.CONTINUOUS,
        assignment_type=AssignmentType.OBSERVATIONAL,
        id_column="unit_id",
        metadata={
            "name": f"push_semisynthetic_continuous_{shape}",
            "source": "myBPmyLife MRT pre-treatment context distribution",
            "license": "local-context-only; raw upstream terms remain applicable",
            "evidence_class": "SEMI_SYNTHETIC_KNOWN_TRUTH",
            "shape": shape,
            "seed": seed,
            "source_row_count": int(len(source_rows)),
            "ground_truth_available": True,
            "assignment_mechanism": "dose = clip(20*sigmoid(context score) + Normal(0,2), 0,20)",
            "dose_is_not_randomized_in_real_mrt": True,
        },
    )
    return SyntheticContinuousData(
        dataset=dataset,
        dose_grid=grid,
        true_response_curves=true_response,
        true_effect_curves=true_effect,
        true_optimal_effect_dose=true_opt_effect,
        true_optimal_economic_dose=true_opt_economic,
        seed=seed,
        outcome_value=outcome_value,
        license="DeepUplift-generated semi-synthetic data from MRT context",
        cost_dose_range=20.0,
    )


def run_push_continuous_suite(
    mrt: MRTDataset | str,
    *,
    shapes: list[str] | None = None,
    rows: int = 600,
    seed: int = 42,
    model_names: list[str] | None = None,
    model_options: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    requested_shapes = shapes or ["monotone", "saturating", "fatigue"]
    results = []
    for index, shape in enumerate(requested_shapes):
        data = generate_push_continuous(mrt, shape=shape, rows=rows, seed=seed + index * 101)
        result = run_continuous_suite(
            data=data,
            seed=seed + index * 101,
            model_names=model_names,
            model_options=model_options,
            outcome_value=data.outcome_value,
        )
        results.append({"shape": shape, **result})
    return {
        "status": "PASS" if results and all(item["status"] == "PASS" for item in results) else "PARTIAL",
        "evidence_class": "SEMI_SYNTHETIC_KNOWN_TRUTH",
        "config": {"shapes": requested_shapes, "rows": rows, "seed": seed, "models": model_names},
        "shapes": results,
        "limitations": [
            "Real Push assignment is not a randomized continuous-dose experiment; these are context-derived semi-synthetic frequency scenarios.",
            "Dose-response metrics use generated truth and are not real business value.",
            "VC-Transformer is not claimed; TransTEE remains OPTIONAL_NOT_VALIDATED in the model catalog.",
        ],
    }


__all__ = ["generate_push_continuous", "run_push_continuous_suite"]
