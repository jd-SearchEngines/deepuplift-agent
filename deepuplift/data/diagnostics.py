from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from deepuplift.contracts import CausalDataset, DataDiagnostics, TreatmentType

from .preprocessing import TabularPreprocessor


def _clean(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _balance(frame: pd.DataFrame, treatment_col: str, feature_cols: list[str]) -> dict[str, Any]:
    treatment = frame[treatment_col]
    values = treatment.dropna().unique().tolist()
    if len(values) != 2:
        return {"status": "not_applicable", "reason": "feature balance currently targets binary treatment"}
    ordered = sorted(values, key=lambda item: str(item))
    control, treated = ordered[0], ordered[1]
    result = {}
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(frame[col]):
            left = pd.to_numeric(frame.loc[treatment == control, col], errors="coerce").dropna()
            right = pd.to_numeric(frame.loc[treatment == treated, col], errors="coerce").dropna()
            pooled = np.sqrt((left.var() + right.var()) / 2) if len(left) and len(right) else 0.0
            result[col] = _clean((right.mean() - left.mean()) / pooled) if pooled else 0.0
        else:
            left = frame.loc[treatment == control, col].astype(str).value_counts(normalize=True)
            right = frame.loc[treatment == treated, col].astype(str).value_counts(normalize=True)
            levels = set(left.index) | set(right.index)
            result[col] = _clean(max(abs(right.get(level, 0.0) - left.get(level, 0.0)) for level in levels)) if levels else None
    return result


def _propensity(frame: pd.DataFrame, dataset: CausalDataset) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    if dataset.treatment_type != TreatmentType.BINARY:
        return {}, {"status": "not_applicable"}, ["Propensity diagnostics currently target binary treatment."]
    treatment = frame[dataset.treatment_col]
    values = sorted(treatment.dropna().unique().tolist(), key=lambda item: str(item))
    if len(values) != 2:
        return {}, {"status": "unavailable"}, ["Binary propensity diagnostics need exactly two treatment values."]
    encoded = treatment.map({values[0]: 0, values[1]: 1}).astype(float)
    preprocessor = TabularPreprocessor(dataset.feature_cols)
    x = preprocessor.fit_transform(frame[dataset.feature_cols])
    if encoded.nunique() < 2:
        return {}, {"status": "failed"}, ["Only one treatment arm is present."]
    try:
        estimator = LogisticRegression(max_iter=500, random_state=42)
        estimator.fit(x, encoded)
        scores = estimator.predict_proba(x)[:, 1]
    except Exception as exc:
        return {}, {"status": "failed", "error": str(exc)}, ["Propensity model could not be fit."]
    summary = {
        "status": "estimated",
        "min": _clean(scores.min()),
        "p05": _clean(np.quantile(scores, 0.05)),
        "median": _clean(np.median(scores)),
        "p95": _clean(np.quantile(scores, 0.95)),
        "max": _clean(scores.max()),
        "mean": _clean(scores.mean()),
    }
    overlap = {
        "status": "estimated",
        "common_support_rate": _clean(np.mean((scores >= 0.05) & (scores <= 0.95))),
        "extreme_propensity_rate": _clean(np.mean((scores < 0.05) | (scores > 0.95))),
        "positivity_warning": bool(np.mean((scores < 0.05) | (scores > 0.95)) > 0.05),
    }
    return summary, overlap, []


def diagnose_dataset(dataset: CausalDataset, *, post_treatment_cols: list[str] | None = None) -> DataDiagnostics:
    frame = dataset.to_pandas()
    feature_cols = list(dataset.feature_cols)
    missingness = {col: _clean(frame[col].isna().mean()) for col in [*feature_cols, dataset.treatment_col, dataset.outcome_col]}
    treatment_distribution = {
        str(key): int(value) for key, value in frame[dataset.treatment_col].value_counts(dropna=False).items()
    }
    warnings: list[str] = []
    leakage_warnings = []
    if post_treatment_cols:
        leakage_warnings.extend(f"Declared post-treatment feature included: {col}" for col in post_treatment_cols if col in feature_cols)
    suspicious_tokens = ("outcome", "conversion", "label", "post_treatment", "future")
    leakage_warnings.extend(
        f"Feature name may indicate post-treatment information: {col}"
        for col in feature_cols
        if any(token in col.lower() for token in suspicious_tokens)
    )
    if frame[dataset.outcome_col].isna().any():
        warnings.append("Outcome contains missing values; affected rows are excluded by model adapters.")
    if len(treatment_distribution) < 2:
        warnings.append("Only one treatment arm is present.")
    balance = _balance(frame, dataset.treatment_col, feature_cols)
    balance_values = [abs(float(value)) for value in balance.values() if isinstance(value, (int, float))]
    if balance_values and max(balance_values) > 0.25:
        warnings.append("At least one feature has a large standardized or categorical treatment imbalance.")

    propensity_summary: dict[str, Any] = {"status": "not_requested"}
    overlap_summary: dict[str, Any] = {"status": "not_requested"}
    if dataset.assignment_type.value == "observational":
        propensity_summary, overlap_summary, propensity_warnings = _propensity(frame, dataset)
        warnings.extend(propensity_warnings)
        warnings.append("Observational estimates rely on conditional ignorability and positivity assumptions.")

    readiness = "READY" if not warnings and not leakage_warnings else "REVIEW"
    return DataDiagnostics(
        sample_size=int(len(frame)),
        treatment_distribution=treatment_distribution,
        missingness=missingness,
        feature_balance=balance,
        propensity_summary=propensity_summary,
        overlap_summary=overlap_summary,
        leakage_warnings=sorted(set(leakage_warnings)),
        warnings=sorted(set(warnings)),
        readiness=readiness,
        metadata={
            "treatment_type": dataset.treatment_type.value,
            "assignment_type": dataset.assignment_type.value,
            "feature_count": len(feature_cols),
        },
    )
