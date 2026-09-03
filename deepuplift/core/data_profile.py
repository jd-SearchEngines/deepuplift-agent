from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _clean(value: Any):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if np.isnan(number) or np.isinf(number):
        return None
    return number


def build_feature_profile(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    features = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        series = df[col]
        missing_rate = float(series.isna().mean())
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            features[col] = {
                "type": "numeric",
                "missing_rate": missing_rate,
                "mean": _clean(numeric.mean()),
                "std": _clean(numeric.std(ddof=0)),
                "p10": _clean(numeric.quantile(0.10)),
                "p50": _clean(numeric.quantile(0.50)),
                "p90": _clean(numeric.quantile(0.90)),
            }
        else:
            values = series.astype("object").where(series.notna(), "__missing__").astype(str)
            freq = values.value_counts(normalize=True).head(20)
            features[col] = {
                "type": "categorical",
                "missing_rate": missing_rate,
                "top_values": {str(key): _clean(value) for key, value in freq.items()},
            }
    return {"rows": int(len(df)), "features": features}


def compare_feature_profile(reference: Dict[str, Any], scoring_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    if not reference:
        return {}
    reference_features = reference.get("features") or {}
    rows = []
    max_shift = 0.0
    high_shift = 0
    for col in feature_cols:
        if col not in scoring_df.columns or col not in reference_features:
            continue
        ref = reference_features[col]
        series = scoring_df[col]
        if ref.get("type") == "numeric":
            numeric = pd.to_numeric(series, errors="coerce")
            score_mean = float(numeric.mean()) if numeric.notna().any() else np.nan
            score_std = float(numeric.std(ddof=0)) if numeric.notna().any() else np.nan
            ref_mean = ref.get("mean")
            ref_std = ref.get("std") or 0.0
            denom = max(abs(float(ref_std or 0.0)), 1e-6)
            shift = abs(score_mean - float(ref_mean or 0.0)) / denom if not np.isnan(score_mean) else np.nan
            p10 = ref.get("p10")
            p90 = ref.get("p90")
            outside_rate = None
            if p10 is not None and p90 is not None:
                outside_rate = float(((numeric < float(p10)) | (numeric > float(p90))).mean())
            metric = "standardized_mean_shift"
            detail = {
                "reference_mean": ref_mean,
                "scoring_mean": _clean(score_mean),
                "reference_std": ref.get("std"),
                "scoring_std": _clean(score_std),
                "outside_reference_p10_p90_rate": _clean(outside_rate),
            }
        else:
            values = series.astype("object").where(series.notna(), "__missing__").astype(str)
            scoring_freq = values.value_counts(normalize=True)
            ref_freq = ref.get("top_values") or {}
            categories = set(ref_freq).union(set(scoring_freq.head(20).index))
            shift = 0.5 * sum(abs(float(ref_freq.get(cat, 0.0)) - float(scoring_freq.get(cat, 0.0))) for cat in categories)
            metric = "total_variation_shift"
            detail = {
                "reference_top_values": ref_freq,
                "scoring_top_values": {str(key): _clean(value) for key, value in scoring_freq.head(10).items()},
            }

        clean_shift = _clean(shift)
        if clean_shift is not None:
            max_shift = max(max_shift, float(clean_shift))
            if float(clean_shift) >= 0.5:
                high_shift += 1
        rows.append(
            {
                "feature": col,
                "type": ref.get("type"),
                "metric": metric,
                "shift": clean_shift,
                "reference_missing_rate": ref.get("missing_rate"),
                "scoring_missing_rate": _clean(series.isna().mean()),
                **detail,
            }
        )

    if max_shift >= 1.0 or high_shift >= 3:
        verdict = "high"
        message = "Scoring population differs materially from training data; treat uplift scores as high-risk until validated."
    elif max_shift >= 0.5 or high_shift:
        verdict = "medium"
        message = "Some scoring features shifted from training data; inspect affected segments before deployment."
    else:
        verdict = "low"
        message = "No large training-vs-scoring feature shift detected."

    return {
        "reference_rows": reference.get("rows"),
        "scoring_rows": int(len(scoring_df)),
        "verdict": verdict,
        "message": message,
        "max_shift": _clean(max_shift),
        "high_shift_features": int(high_shift),
        "rows": sorted(rows, key=lambda row: abs(row.get("shift") or 0), reverse=True),
    }
