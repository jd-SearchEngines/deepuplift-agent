from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from deepuplift.contracts import CausalDataset, EffectPrediction


def _frame(prediction: EffectPrediction, dataset: CausalDataset) -> pd.DataFrame:
    result = prediction.to_frame()
    observed = dataset.to_pandas().reset_index(drop=True)
    result["treatment"] = observed[dataset.treatment_col].to_numpy()
    result["outcome"] = pd.to_numeric(observed[dataset.outcome_col], errors="coerce").to_numpy()
    return result.dropna(subset=["treatment", "outcome", "uplift"])


def qini_curve(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values("uplift", ascending=False).reset_index(drop=True)
    treated = ordered["treatment"].to_numpy() == 1
    outcomes = ordered["outcome"].to_numpy(dtype="float64")
    treated_n = np.cumsum(treated)
    control_n = np.cumsum(~treated)
    treated_y = np.cumsum(np.where(treated, outcomes, 0.0))
    control_y = np.cumsum(np.where(~treated, outcomes, 0.0))
    gain = treated_y - np.divide(treated_n, np.maximum(control_n, 1)) * control_y
    return pd.DataFrame({"fraction": (np.arange(len(ordered)) + 1) / max(len(ordered), 1), "qini": gain})


def _area(curve: pd.DataFrame, value_col: str) -> float | None:
    if curve.empty:
        return None
    x = np.concatenate([[0.0], curve["fraction"].to_numpy(dtype="float64")])
    y = np.concatenate([[0.0], curve[value_col].to_numpy(dtype="float64")])
    integrator = getattr(np, "trapezoid", np.trapz)
    return float(integrator(y, x))


def uplift_at_k(frame: pd.DataFrame, fraction: float = 0.1) -> float | None:
    if frame.empty:
        return None
    top = frame.sort_values("uplift", ascending=False).head(max(1, int(len(frame) * fraction)))
    treated = top[top["treatment"] == 1]["outcome"]
    control = top[top["treatment"] == 0]["outcome"]
    if treated.empty or control.empty:
        return None
    return float(treated.mean() - control.mean())


def ranking_metrics(prediction: EffectPrediction, dataset: CausalDataset, fractions: tuple[float, ...] = (0.05, 0.1, 0.2)) -> dict[str, Any]:
    frame = _frame(prediction, dataset)
    curve = qini_curve(frame)
    top_k = [{"fraction": fraction, "uplift": uplift_at_k(frame, fraction)} for fraction in fractions]
    return {
        "qini": _area(curve, "qini"),
        "auuc": _area(curve, "qini"),
        "uplift_at_k": top_k,
        "rows": int(len(frame)),
        "curve": curve.to_dict(orient="records"),
    }
