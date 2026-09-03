from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from deepuplift.contracts import CausalDataset, EffectPrediction
from deepuplift.decision.evaluation.calibration import calibration_metrics
from deepuplift.decision.evaluation.ranking import ranking_metrics
from deepuplift.decision.policy.binary import build_binary_policy


def ground_truth_metrics(prediction: EffectPrediction, dataset: CausalDataset, *, true_effect_col: str = "true_effect") -> dict[str, Any]:
    frame = dataset.to_pandas().reset_index(drop=True)
    if true_effect_col not in frame:
        return {"available": False, "reason": "No ground-truth effect is available; PEHE is not computed."}
    truth = pd.to_numeric(frame[true_effect_col], errors="coerce").to_numpy(dtype="float64")
    estimate = np.asarray(prediction.cate, dtype="float64")
    valid = np.isfinite(truth) & np.isfinite(estimate)
    if not valid.any():
        return {"available": False, "reason": "Ground-truth effect has no valid rows."}
    top = max(1, int(valid.sum() * .1))
    return {"available": True, "pehe": float(np.sqrt(np.mean((estimate[valid] - truth[valid]) ** 2))), "ate_bias": float(np.mean(estimate[valid]) - np.mean(truth[valid])), "cate_correlation": float(np.corrcoef(estimate[valid], truth[valid])[0, 1]) if valid.sum() > 1 and np.std(estimate[valid]) > 0 and np.std(truth[valid]) > 0 else None, "top10_true_effect": float(np.mean(np.sort(truth[valid])[-top:]))}


def policy_metrics(prediction: EffectPrediction, dataset: CausalDataset, *, treatment_cost: float = 0.0, outcome_value: float = 1.0, budget: float | None = None) -> dict[str, Any]:
    policy = build_binary_policy(prediction, treatment_cost=treatment_cost, outcome_value=outcome_value, budget=budget)
    return {"target_count": policy.summary["target_count"], "incremental_outcome": policy.summary["expected_incremental_outcome"], "incremental_value": policy.summary["expected_incremental_value"], "treatment_cost": policy.summary["total_cost"], "net_value": policy.summary["expected_net_value"], "iROAS": policy.summary["iroas"], "budget_utilization": policy.summary["budget_utilization"], "offline_only": True}


def evaluate_prediction(prediction: EffectPrediction, dataset: CausalDataset, *, true_effect_col: str | None = None) -> dict[str, Any]:
    result = {"ranking": ranking_metrics(prediction, dataset), "calibration": calibration_metrics(prediction, dataset), "policy": policy_metrics(prediction, dataset)}
    if true_effect_col:
        result["ground_truth"] = ground_truth_metrics(prediction, dataset, true_effect_col=true_effect_col)
    else:
        result["ground_truth"] = {"available": False, "reason": "Ground truth is not available for this dataset."}
    return result


__all__ = ["ground_truth_metrics", "policy_metrics", "evaluate_prediction"]
