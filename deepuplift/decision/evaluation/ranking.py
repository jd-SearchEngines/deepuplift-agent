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
    treated_mean = np.divide(treated_y, treated_n, out=np.full_like(treated_y, np.nan, dtype="float64"), where=treated_n > 0)
    control_mean = np.divide(control_y, control_n, out=np.full_like(control_y, np.nan, dtype="float64"), where=control_n > 0)
    prefix_count = np.arange(len(ordered), dtype="float64") + 1.0
    curve = pd.DataFrame({
        "fraction": (np.arange(len(ordered)) + 1) / max(len(ordered), 1),
        "qini": gain,
        "uplift": treated_mean - control_mean,
        "auuc_gain": (treated_mean - control_mean) * prefix_count,
    })
    # A ranked prefix with only one treatment arm has no identifiable uplift
    # rate. Exclude it from both area calculations instead of treating the
    # missing arm as a zero outcome.
    return curve.loc[(treated_n > 0) & (control_n > 0)].reset_index(drop=True)


def _area(curve: pd.DataFrame, value_col: str, *, include_origin: bool = True) -> float | None:
    if curve.empty:
        return None
    if not include_origin and len(curve) < 2:
        return None
    x_values = curve["fraction"].to_numpy(dtype="float64")
    y_values = curve[value_col].to_numpy(dtype="float64")
    if include_origin:
        x = np.concatenate([[0.0], x_values])
        y = np.concatenate([[0.0], y_values])
    else:
        x, y = x_values, y_values
    integrator = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrator(y, x))


def _qini_and_auuc(curve: pd.DataFrame) -> tuple[float | None, float | None, float | None]:
    """Return (qini coefficient, AUUC, random-baseline area)."""
    qini_raw_area = _area(curve, "qini", include_origin=False)
    auuc = _area(curve, "auuc_gain", include_origin=False)
    if qini_raw_area is None or curve.empty:
        return None, auuc, None
    total_gain = float(curve["qini"].iloc[-1])
    support_start = float(curve["fraction"].iloc[0])
    random_area = 0.5 * total_gain * (1.0 - support_start**2)
    return float(qini_raw_area - random_area), float(auuc) if auuc is not None else None, float(random_area)


def uplift_at_k(frame: pd.DataFrame, fraction: float = 0.1) -> float | None:
    if frame.empty:
        return None
    top = frame.sort_values("uplift", ascending=False).head(max(1, int(len(frame) * fraction)))
    treated = top[top["treatment"] == 1]["outcome"]
    control = top[top["treatment"] == 0]["outcome"]
    if treated.empty or control.empty:
        return None
    return float(treated.mean() - control.mean())


def ranking_metrics(
    prediction: EffectPrediction,
    dataset: CausalDataset,
    fractions: tuple[float, ...] = (0.05, 0.1, 0.2),
    *,
    include_curves: bool = True,
    bootstrap_samples: int = 100,
    random_state: int = 42,
) -> dict[str, Any]:
    frame = _frame(prediction, dataset)
    curve = qini_curve(frame)
    qini, auuc, random_baseline_area = _qini_and_auuc(curve)
    top_k = [{"fraction": fraction, "uplift": uplift_at_k(frame, fraction)} for fraction in fractions]
    ordered = frame.sort_values("uplift", ascending=False).reset_index(drop=True)
    treated = (ordered["treatment"].to_numpy() == 1).astype("float64")
    balance_fraction = (np.arange(len(ordered)) + 1) / max(len(ordered), 1)
    balance_rate = np.cumsum(treated) / (np.arange(len(ordered)) + 1)
    rng = np.random.default_rng(random_state)
    bootstrap_qini = []
    bootstrap_auuc = []
    for _ in range(max(0, int(bootstrap_samples))):
        if len(frame) < 2:
            break
        sample = frame.iloc[rng.integers(0, len(frame), len(frame))]
        if sample["treatment"].nunique() == 2:
            sample_qini, sample_auuc, _ = _qini_and_auuc(qini_curve(sample))
            if sample_qini is not None and sample_auuc is not None:
                bootstrap_qini.append(sample_qini)
                bootstrap_auuc.append(sample_auuc)

    def _ci(values: list[float]) -> list[float | None]:
        if not values:
            return [None, None]
        return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]

    return {
        "qini": qini,
        "auuc": auuc,
        "qini_raw_area": float(qini + random_baseline_area) if qini is not None and random_baseline_area is not None else None,
        "random_baseline_area": random_baseline_area,
        "identifiable_support": {
            "fraction_start": float(curve["fraction"].iloc[0]) if not curve.empty else None,
            "fraction_end": float(curve["fraction"].iloc[-1]) if not curve.empty else None,
        },
        "definitions": {
            "qini": "area under the cumulative Qini gain curve minus the area of the straight random-targeting baseline",
            "auuc": "area under the cumulative uplift-rate curve over ranked fractions",
            "bootstrap": "row bootstrap over the observed evaluation sample",
        },
        "uplift_at_k": top_k,
        "uplift_at_10": uplift_at_k(frame, .1),
        "uplift_at_20": uplift_at_k(frame, .2),
        "rows": len(frame),
        "curve": curve.to_dict(orient="records") if include_curves else None,
        "gain_curve": curve.to_dict(orient="records") if include_curves else None,
        "treatment_balance_curve": [
            {"fraction": float(fraction), "treated_rate": float(rate)}
            for fraction, rate in zip(balance_fraction, balance_rate, strict=True)
        ] if include_curves else None,
        "bootstrap_stability": {
            "n": len(bootstrap_qini),
            "samples_requested": int(bootstrap_samples),
            "random_state": int(random_state),
            "mean_qini": float(np.mean(bootstrap_qini)) if bootstrap_qini else None,
            "std_qini": float(np.std(bootstrap_qini)) if bootstrap_qini else None,
            "qini_ci_95": _ci(bootstrap_qini),
            "mean_auuc": float(np.mean(bootstrap_auuc)) if bootstrap_auuc else None,
            "std_auuc": float(np.std(bootstrap_auuc)) if bootstrap_auuc else None,
            "auuc_ci_95": _ci(bootstrap_auuc),
        },
    }
