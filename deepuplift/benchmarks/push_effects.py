"""Simple participant-held-out proximal-effect baselines for Push.

The model is a documented WCLS-style centered treatment regression. It is
intended as a stable reference, not as proof of individual CATE accuracy on
the real MRT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from deepuplift.data.preprocessing import TabularPreprocessor
from deepuplift.data.public.mrt import MRTDataset, MRT_DECLARED_ASSIGNMENT_PROBABILITIES, load_mybpmylife_mrt
from deepuplift.data.push import build_mrt_decision_points


_STATE_COLUMNS = (
    "TimeEnrollment_to_PrefDate_days",
    "DoesPrefDateHaveRepeat",
    "Phase",
    "Phase_1dayBefore",
    "TimeOfDayID",
    "TimeOfDay",
    "PrefRepeatRank",
    "IsPresent_HeartRate_30minBefore",
    "NumValues_HeartRate_30minBefore",
    "SumValues_WearTime_hr_30minBefore",
    "SumValues_Steps_30minBefore",
    "NumValues_LowSaltChoices_30minBefore",
    "SodiumIntakeScore",
)


def _validate_binary_propensity(values: Any, name: str) -> np.ndarray:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(array).all() or np.any(array <= 0.0) or np.any(array >= 1.0):
        raise ValueError(f"{name} must be finite and strictly between 0 and 1.")
    return array


def _design(preprocessed: pd.DataFrame, centered_treatment: np.ndarray) -> np.ndarray:
    base = np.c_[np.ones(len(preprocessed)), preprocessed.to_numpy(dtype="float64")]
    return np.c_[base, base * centered_treatment[:, None]]


@dataclass
class WCLSProximalEffectModel:
    """Centered treatment regression with known binary randomization probability."""

    feature_cols: list[str]
    alpha: float = 10.0
    preprocessor: TabularPreprocessor | None = None
    model: Ridge | None = None

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        treatment_col: str = "treatment",
        outcome_col: str = "outcome",
        propensity_col: str = "push_probability",
    ) -> "WCLSProximalEffectModel":
        missing = sorted(set([*self.feature_cols, treatment_col, outcome_col, propensity_col]) - set(frame.columns))
        if missing:
            raise ValueError(f"Proximal model frame is missing columns: {missing}")
        treatment = pd.to_numeric(frame[treatment_col], errors="coerce").to_numpy(dtype="float64")
        outcome = pd.to_numeric(frame[outcome_col], errors="coerce").to_numpy(dtype="float64")
        propensity = _validate_binary_propensity(frame[propensity_col], propensity_col)
        if not np.isin(treatment, [0.0, 1.0]).all() or not np.isfinite(outcome).all():
            raise ValueError("Proximal model requires finite binary treatment and outcome values.")
        self.preprocessor = TabularPreprocessor(self.feature_cols)
        x = self.preprocessor.fit_transform(frame[self.feature_cols])
        centered = treatment - propensity
        design = _design(x, centered)
        # Inverse treatment-variance weighting is the WCLS-style weighting
        # contract used here; the randomization probability is never fitted.
        sample_weight = 1.0 / np.maximum(propensity * (1.0 - propensity), 1e-8)
        self.model = Ridge(alpha=float(self.alpha), fit_intercept=False)
        self.model.fit(design, outcome, sample_weight=sample_weight)
        return self

    def predict_effect(self, frame: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None or self.model is None:
            raise RuntimeError("WCLSProximalEffectModel must be fitted before prediction.")
        x = self.preprocessor.transform(frame[self.feature_cols])
        base = np.c_[np.ones(len(x)), x.to_numpy(dtype="float64")]
        coefficients = self.model.coef_.reshape(-1)
        return np.asarray(base @ coefficients[len(base[0]) :], dtype="float64")


def fit_group_crossfit_proximal(
    frame: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    group_col: str = "user_id",
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    propensity_col: str = "push_probability",
    n_splits: int = 5,
    seed: int = 42,
    alpha: float = 10.0,
) -> dict[str, Any]:
    """Fit WCLS-style models on participant-held-out folds."""

    # GroupKFold is deterministic; seed remains in the report contract for
    # reproducibility even though it is not used to randomize the folds.
    data = frame.reset_index(drop=True).copy()
    groups = data[group_col].astype(str)
    unique_groups = groups.nunique()
    if unique_groups < 2:
        return {"status": "NOT_AVAILABLE", "reason": "At least two participant groups are required."}
    folds = min(int(n_splits), int(unique_groups))
    predictions = np.full(len(data), np.nan, dtype="float64")
    fold_rows = []
    splitter = GroupKFold(n_splits=folds)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(data, groups=groups)):
        model = WCLSProximalEffectModel(list(feature_cols), alpha=alpha)
        model.fit(
            data.iloc[train_idx],
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            propensity_col=propensity_col,
        )
        predictions[test_idx] = model.predict_effect(data.iloc[test_idx])
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_groups": int(groups.iloc[train_idx].nunique()),
                "test_groups": int(groups.iloc[test_idx].nunique()),
                "group_overlap": int(len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]))),
            }
        )
    return {
        "status": "PASS",
        "method": "participant_group_crossfit_wcls_style",
        "feature_cols": list(feature_cols),
        "alpha": float(alpha),
        "n_splits": folds,
        "seed": int(seed),
        "predicted_effect": predictions,
        "folds": fold_rows,
        "participant_overlap": int(sum(row["group_overlap"] for row in fold_rows)),
    }


def build_mrt_pairwise_frame(
    dataset: MRTDataset | str,
    *,
    outcome: str,
    arm: str,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Build an arm-vs-none proximal frame with pre-treatment history only."""

    if arm not in {"activity", "diet"}:
        raise ValueError("arm must be activity or diet")
    mrt = dataset if isinstance(dataset, MRTDataset) else load_mybpmylife_mrt(dataset)
    points = build_mrt_decision_points(mrt, outcome=outcome).to_pandas()
    outcome_column = "outcome"
    pair = points.loc[points["treatment_assignment"].isin(["none", arm])].copy()
    pair["treatment"] = pair["treatment_assignment"].eq(arm).astype(int)
    pair["push_probability"] = np.where(
        pair["treatment"].eq(1),
        MRT_DECLARED_ASSIGNMENT_PROBABILITIES[arm] / (MRT_DECLARED_ASSIGNMENT_PROBABILITIES[arm] + MRT_DECLARED_ASSIGNMENT_PROBABILITIES["none"]),
        MRT_DECLARED_ASSIGNMENT_PROBABILITIES[arm] / (MRT_DECLARED_ASSIGNMENT_PROBABILITIES[arm] + MRT_DECLARED_ASSIGNMENT_PROBABILITIES["none"]),
    )
    state = pd.json_normalize(pair["user_state"]).add_prefix("state_")
    history = pd.json_normalize(pair["user_history"])[["prior_assignment_count"]].rename(columns={"prior_assignment_count": "history_prior_assignment_count"})
    state.index = pair.index
    history.index = pair.index
    pair = pd.concat([pair, state, history], axis=1)
    feature_cols = [column for column in [*state.columns, *history.columns, "decision_sequence"] if column in pair.columns]
    pair["outcome"] = pd.to_numeric(pair[outcome_column], errors="coerce")
    pair["user_id"] = pair["user_id"].astype(str)
    metadata = {
        "outcome": outcome,
        "arm": arm,
        "estimand": f"complete-outcome arm assignment contrast: {arm} vs none, conditional on pre-treatment state/history",
        "assignment_probability": {
            "arm": MRT_DECLARED_ASSIGNMENT_PROBABILITIES[arm],
            "none": MRT_DECLARED_ASSIGNMENT_PROBABILITIES["none"],
            "conditional_arm_probability": float(pair["push_probability"].iloc[0]) if len(pair) else None,
        },
        "outcome_window": "60 minutes after decision" if outcome.startswith("activity") else "24 hours after decision",
        "truth_status": "NOT_AVAILABLE_ON_REAL_MRT",
    }
    return pair, feature_cols, metadata


def run_mrt_proximal_baseline(
    dataset: MRTDataset | str,
    *,
    outcome: str = "activity_steps_60min",
    arm: str = "activity",
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Run participant-held-out proximal effect modeling on a real MRT source."""

    frame, feature_cols, metadata = build_mrt_pairwise_frame(dataset, outcome=outcome, arm=arm)
    full_rows = len(frame)
    missing_by_arm = {
        str(key): int(value)
        for key, value in frame.assign(_missing=frame["outcome"].isna()).groupby("treatment")["_missing"].sum().items()
    }
    complete = frame.loc[frame["outcome"].notna()].copy().reset_index(drop=True)
    if complete["user_id"].nunique() < 2 or len(complete) < 6:
        return {
            "status": "NOT_AVAILABLE",
            "evidence_class": "REAL_RANDOMIZED_MRT",
            "metadata": metadata,
            "rows_before_outcome_filter": int(full_rows),
            "rows_complete_outcome": int(len(complete)),
            "missing_outcome_rows_by_arm": missing_by_arm,
            "reason": "Too few complete outcome rows or participant groups for held-out proximal model.",
        }
    fit = fit_group_crossfit_proximal(
        complete,
        feature_cols=feature_cols,
        n_splits=n_splits,
        seed=seed,
    )
    effects = fit.pop("predicted_effect")
    return {
        "status": fit["status"],
        "evidence_class": "REAL_RANDOMIZED_MRT",
        "metadata": metadata,
        "rows_before_outcome_filter": int(full_rows),
        "rows_complete_outcome": int(len(complete)),
        "participants_complete_outcome": int(complete["user_id"].nunique()),
        "missing_outcome_rows_by_arm": missing_by_arm,
        "missingness_estimand_warning": "Complete-outcome held-out model; full-population ITT is not claimed without a missingness assumption.",
        "model": fit,
        "held_out_predicted_effect_summary": {
            "mean": float(np.mean(effects)),
            "p10": float(np.quantile(effects, 0.10)),
            "p90": float(np.quantile(effects, 0.90)),
        },
        "real_data_pehe": "NOT_AVAILABLE",
    }


__all__ = ["WCLSProximalEffectModel", "build_mrt_pairwise_frame", "fit_group_crossfit_proximal", "run_mrt_proximal_baseline"]
