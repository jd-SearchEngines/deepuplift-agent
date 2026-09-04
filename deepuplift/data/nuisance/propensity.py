from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from deepuplift.contracts import CausalDataset, TreatmentType
from deepuplift.data.preprocessing import TabularPreprocessor

from .contracts import NuisanceResult


def _binary_treatment(values: pd.Series) -> tuple[np.ndarray, dict[Any, int]]:
    unique = values.dropna().unique().tolist()
    if len(unique) != 2:
        raise ValueError(f"Propensity estimation requires two treatment arms; got {unique}.")
    ordered = sorted(unique, key=lambda item: str(item))
    mapping = {ordered[0]: 0, ordered[1]: 1}
    return values.map(mapping).to_numpy(dtype="float64"), mapping


def _make_estimator(name: str, random_state: int, **kwargs: Any):
    key = name.lower().replace("_", "-")
    if key == "logistic":
        return LogisticRegression(max_iter=1000, random_state=random_state, **kwargs)
    if key in {"gradient-boosting", "gbm"}:
        return GradientBoostingClassifier(random_state=random_state, **kwargs)
    if key in {"hist-gradient-boosting", "histgb"}:
        return HistGradientBoostingClassifier(random_state=random_state, **kwargs)
    if key == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("lightgbm propensity estimation requires the optional 'lightgbm' extra.") from exc
        return LGBMClassifier(random_state=random_state, verbosity=-1, **kwargs)
    raise ValueError("Unknown propensity estimator. Choose logistic, gradient_boosting, hist_gradient_boosting, or lightgbm.")


def estimate_propensity(
    dataset: CausalDataset,
    *,
    estimator: str = "logistic",
    cross_fit: bool = True,
    n_splits: int = 5,
    random_state: int = 42,
    clipping_range: tuple[float, float] | None = None,
) -> NuisanceResult:
    """Estimate binary propensity and OOF outcome nuisances under one contract."""
    if dataset.treatment_type != TreatmentType.BINARY:
        raise ValueError("The unified propensity layer currently supports binary treatment only.")
    frame = dataset.to_pandas().reset_index(drop=True)
    treatment, mapping = _binary_treatment(frame[dataset.treatment_col])
    if np.isnan(treatment).any() or len(np.unique(treatment)) != 2:
        raise ValueError("Both treatment arms are required for propensity estimation.")
    y = pd.to_numeric(frame[dataset.outcome_col], errors="coerce").to_numpy(dtype="float64")
    if np.isnan(y).any():
        raise ValueError("Outcome must be numeric and non-missing for nuisance estimation.")
    preprocessor = TabularPreprocessor(dataset.feature_cols)
    x = preprocessor.fit_transform(frame[dataset.feature_cols])
    n = len(frame)
    if cross_fit:
        from sklearn.model_selection import StratifiedKFold

        if n_splits < 2:
            raise ValueError("n_splits must be >= 2 when cross_fit=True.")
        counts = np.bincount(treatment.astype(int))
        if counts.min() < n_splits:
            raise ValueError(f"n_splits={n_splits} exceeds the smallest treatment arm ({counts.min()}).")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = list(splitter.split(x, treatment))
    else:
        folds = [(np.arange(n), np.arange(n))]
    propensity = np.empty(n, dtype="float64")
    m0 = np.empty(n, dtype="float64")
    m1 = np.empty(n, dtype="float64")
    fold_ids = np.full(n, -1, dtype="int64")
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        fold_ids[test_idx] = fold_id
        model = _make_estimator(estimator, random_state + fold_id)
        model.fit(x.iloc[train_idx], treatment[train_idx])
        propensity[test_idx] = model.predict_proba(x.iloc[test_idx])[:, 1]
        for arm, output in ((0, m0), (1, m1)):
            arm_idx = train_idx[treatment[train_idx] == arm]
            if len(arm_idx) < 2:
                output[test_idx] = float(np.mean(y[arm_idx])) if len(arm_idx) else float(np.mean(y[train_idx]))
                continue
            outcome = GradientBoostingClassifier(random_state=random_state + fold_id) if set(np.unique(y[arm_idx])).issubset({0.0, 1.0}) else GradientBoostingRegressor(random_state=random_state + fold_id)
            try:
                outcome.fit(x.iloc[arm_idx], y[arm_idx])
                output[test_idx] = outcome.predict_proba(x.iloc[test_idx])[:, 1] if hasattr(outcome, "predict_proba") else outcome.predict(x.iloc[test_idx])
            except ValueError:
                output[test_idx] = float(np.mean(y[arm_idx]))
    # A full propensity model is retained only for predicting new rows.  The
    # estimates used to train effect learners remain OOF when cross_fit=True.
    full_model = _make_estimator(estimator, random_state)
    full_model.fit(x, treatment)
    raw = propensity.copy()
    if clipping_range is not None:
        low, high = clipping_range
        if not 0 < low < high < 1:
            raise ValueError("clipping_range must satisfy 0 < low < high < 1.")
        propensity = np.clip(propensity, low, high)
    diagnostics = _diagnostics(raw, treatment, propensity, m0, m1, clipping_range)
    return NuisanceResult(
        propensity_scores=propensity,
        propensity_model_metadata={"estimator": estimator, "treatment_mapping": {str(k): v for k, v in mapping.items()}, "cross_fit": cross_fit, "n_splits": n_splits if cross_fit else 1},
        outcome_control_oof=m0,
        outcome_treated_oof=m1,
        fold_ids=fold_ids,
        overlap_mask=diagnostics["overlap_mask"],
        trim_mask=np.ones(n, dtype=bool),
        sample_weights=np.ones(n, dtype="float64"),
        effective_sample_size={},
        clipping_range=clipping_range,
        diagnostics=diagnostics,
        metadata={"assignment_type": dataset.assignment_type.value, "provenance": "deepuplift unified nuisance layer"},
        propensity_model=full_model,
        propensity_preprocessor=preprocessor,
    )


def _diagnostics(raw: np.ndarray, treatment: np.ndarray, scores: np.ndarray, m0: np.ndarray, m1: np.ndarray, clipping_range: tuple[float, float] | None) -> dict[str, Any]:
    quantiles = {f"p{q:02d}": float(np.quantile(raw, q / 100)) for q in (1, 5, 25, 50, 75, 95, 99)}
    low, high = (clipping_range if clipping_range is not None else (0.05, 0.95))
    overlap = (raw >= low) & (raw <= high)
    return {"status": "estimated", "min": float(raw.min()), **quantiles, "median": float(np.median(raw)), "max": float(raw.max()), "mean": float(raw.mean()), "common_support_rate": float(overlap.mean()), "extreme_propensity_rate": float((~overlap).mean()), "warning": "Extreme propensity or limited common support requires review." if (~overlap).mean() > .05 else None, "treated_distribution": _distribution(raw[treatment == 1]), "control_distribution": _distribution(raw[treatment == 0]), "overlap_mask": overlap, "outcome_nuisance_oof": {"control_mean": float(m0.mean()), "treated_mean": float(m1.mean())}, "clipping": {"range": clipping_range, "applied": clipping_range is not None}}


def _distribution(values: np.ndarray) -> dict[str, float | None]:
    return {"min": float(values.min()) if len(values) else None, "p01": float(np.quantile(values, .01)) if len(values) else None, "median": float(np.median(values)) if len(values) else None, "p99": float(np.quantile(values, .99)) if len(values) else None, "max": float(values.max()) if len(values) else None}


from sklearn.ensemble import GradientBoostingRegressor

__all__ = ["estimate_propensity"]
