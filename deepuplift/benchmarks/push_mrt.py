"""P1 design-based baselines for the myBPmyLife Push MRT.

This module is intentionally separate from binary observational benchmarks.
The primary estimand is assignment-based ITT with the trial's declared arm
probabilities and participant-clustered uncertainty.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from deepuplift.data.preprocessing import TabularPreprocessor
from deepuplift.data.public.mrt import (
    MRTDataset,
    MRT_DECLARED_ASSIGNMENT_PROBABILITIES,
    audit_mrt_dataset,
    load_mybpmylife_mrt,
    participant_group_split,
)


_OUTCOMES = {
    "activity_steps_60min": "SumValues_Steps_60minAfter",
    "diet_low_salt_24hr": "NumValues_LowSaltChoices_24hrAfter",
}


def _complete(frame: pd.DataFrame, outcome_column: str) -> pd.DataFrame:
    result = frame.copy()
    result["_outcome"] = pd.to_numeric(result[outcome_column], errors="coerce")
    values = result["_outcome"].to_numpy(dtype="float64")
    return result.loc[result["_outcome"].notna() & np.isfinite(values)].copy()


def _ht_mean(frame: pd.DataFrame, arm: str) -> float:
    if frame.empty:
        return float("nan")
    indicator = frame["assignment_arm"].eq(arm).to_numpy(dtype="float64")
    values = frame["_outcome"].to_numpy(dtype="float64")
    return float(np.sum(indicator * values / MRT_DECLARED_ASSIGNMENT_PROBABILITIES[arm]) / len(frame))


def _contrast(frame: pd.DataFrame, left: str, right: str) -> float:
    return float(_ht_mean(frame, left) - _ht_mean(frame, right))


def _cluster_bootstrap_ci(
    frame: pd.DataFrame,
    left: str,
    right: str,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    participants = frame["participant_id"].dropna().unique()
    if len(participants) < 2 or repetitions <= 0:
        return {"repetitions": 0, "ci_95": [None, None], "standard_error": None}
    groups = {participant: group for participant, group in frame.groupby("participant_id", sort=False)}
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(int(repetitions)):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        sample = pd.concat([groups[participant] for participant in sampled], ignore_index=True)
        value = _contrast(sample, left, right)
        if np.isfinite(value):
            estimates.append(float(value))
    if not estimates:
        return {"repetitions": int(repetitions), "ci_95": [None, None], "standard_error": None}
    values = np.asarray(estimates, dtype="float64")
    return {
        "repetitions": int(repetitions),
        "finite_repetitions": int(len(values)),
        "ci_95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "standard_error": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "cluster_column": "participant_id",
    }


def _context_conditional_analysis(
    mrt: MRTDataset,
    outcome_name: str,
    *,
    seed: int,
    test_fraction: float,
) -> dict[str, Any]:
    outcome_dataset = mrt.to_causal_dataset(outcome_name)
    analysis = outcome_dataset.to_pandas().reset_index(drop=True)
    train_pos, test_pos = participant_group_split(mrt, test_fraction=test_fraction, seed=seed)
    analysis_frame = mrt.analysis_frame().reset_index(drop=True)
    train_participants = set(analysis_frame.iloc[train_pos]["participant_id"])
    test_participants = set(analysis_frame.iloc[test_pos]["participant_id"])
    complete = _complete(analysis, "mrt_outcome")
    train_mask = complete["participant_id"].isin(train_participants).to_numpy()
    test_mask = complete["participant_id"].isin(test_participants).to_numpy()
    if int(train_mask.sum()) < 6 or int(test_mask.sum()) < 1:
        return {"status": "NOT_AVAILABLE", "reason": "Participant-held-out context split is too small."}
    prep = TabularPreprocessor(outcome_dataset.feature_cols)
    train_x = prep.fit_transform(complete.loc[train_mask, outcome_dataset.feature_cols])
    test_x = prep.transform(complete.loc[test_mask, outcome_dataset.feature_cols])
    train_frame = complete.loc[train_mask].reset_index(drop=True)
    test_frame = complete.loc[test_mask].reset_index(drop=True)
    predictions: dict[str, np.ndarray] = {}
    model_status: dict[str, str] = {}
    for arm in MRT_DECLARED_ASSIGNMENT_PROBABILITIES:
        arm_mask = train_frame["assignment_arm"].eq(arm).to_numpy()
        if int(arm_mask.sum()) < 2:
            predictions[arm] = np.full(len(test_frame), float(train_frame["_outcome"].mean()))
            model_status[arm] = "fallback_train_mean"
            continue
        model = Ridge(alpha=10.0)
        model.fit(train_x.loc[arm_mask], train_frame.loc[arm_mask, "_outcome"])
        predictions[arm] = np.asarray(model.predict(test_x), dtype="float64")
        model_status[arm] = "ridge_on_pre_treatment_features"
    conditional = {}
    for arm in ("activity", "diet"):
        effect = predictions[arm] - predictions["none"]
        conditional[f"{arm}_vs_none"] = {
            "mean_predicted_effect": float(np.mean(effect)),
            "p10": float(np.quantile(effect, 0.10)),
            "p90": float(np.quantile(effect, 0.90)),
        }
    return {
        "status": "PASS",
        "estimand": "held_out_conditional_assignment_effect_surface; descriptive model-based heterogeneity, not individual counterfactual truth",
        "feature_cols": list(outcome_dataset.feature_cols),
        "future_or_post_treatment_features_rejected": True,
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_participants": int(len(train_participants)),
        "test_participants": int(len(test_participants)),
        "participant_overlap": int(len(train_participants & test_participants)),
        "models": model_status,
        "contrasts": conditional,
    }


def _outcome_result(
    mrt: MRTDataset,
    outcome_name: str,
    *,
    bootstrap_repetitions: int,
    seed: int,
    test_fraction: float,
) -> dict[str, Any]:
    column = _OUTCOMES[outcome_name]
    analysis_frame = mrt.analysis_frame()
    complete = _complete(analysis_frame, column)
    arms = {}
    for arm in MRT_DECLARED_ASSIGNMENT_PROBABILITIES:
        arm_frame = complete.loc[complete["assignment_arm"] == arm]
        all_arm_rows = analysis_frame.loc[analysis_frame["assignment_arm"] == arm]
        arms[arm] = {
            "rows": int(len(arm_frame)),
            "participants": int(arm_frame["participant_id"].nunique()),
            "observed_mean": float(arm_frame["_outcome"].mean()) if len(arm_frame) else None,
            "observed_sd": float(arm_frame["_outcome"].std(ddof=1)) if len(arm_frame) > 1 else None,
            "missing_rate_within_analysis_rows": float(all_arm_rows[column].isna().mean()) if len(all_arm_rows) else None,
        }
    contrasts = {}
    pairs = [("activity", "none"), ("diet", "none"), ("diet", "activity")]
    for left, right in pairs:
        name = f"{left}_vs_{right}"
        contrasts[name] = {
            "estimate": _contrast(complete, left, right),
            "estimand": "Horvitz-Thompson assignment ITT using declared MRT arm probabilities on outcome-complete rows",
            "left_arm_probability": MRT_DECLARED_ASSIGNMENT_PROBABILITIES[left],
            "right_arm_probability": MRT_DECLARED_ASSIGNMENT_PROBABILITIES[right],
            "cluster_bootstrap": _cluster_bootstrap_ci(complete, left, right, repetitions=bootstrap_repetitions, seed=seed + len(contrasts)),
        }
    delivery_rows = analysis_frame.loc[analysis_frame["assignment_arm"].isin(["activity", "diet"])]
    return {
        "outcome": outcome_name,
        "source_column": column,
        "outcome_window": "60 minutes after decision" if outcome_name.startswith("activity") else "24 hours after decision",
        "complete_rows": int(len(complete)),
        "complete_participants": int(complete["participant_id"].nunique()),
        "arm_descriptives": arms,
        "itt_contrasts": contrasts,
        "delivery_separate_from_assignment": {
            "notification_assignment_rows": int(len(delivery_rows)),
            "delivered_rows": int(delivery_rows["delivery_observed"].sum()),
            "delivery_rate": float(delivery_rows["delivery_observed"].mean()) if len(delivery_rows) else None,
            "causal_delivery_effect_estimated": False,
        },
        "context_conditional": _context_conditional_analysis(mrt, outcome_name, seed=seed, test_fraction=test_fraction),
        "ground_truth": {
            "individual_counterfactuals": False,
            "pehe": "NOT_AVAILABLE",
            "real_data_causal_claim": "No individual counterfactual error or online lift claim is made.",
        },
    }


def run_push_mrt_baseline(
    dataset: MRTDataset | str,
    *,
    outcomes: list[str] | None = None,
    bootstrap_repetitions: int = 200,
    seed: int = 42,
    test_fraction: float = 0.2,
) -> dict[str, Any]:
    """Run outcome-specific randomized MRT ITT baselines."""

    mrt = dataset if isinstance(dataset, MRTDataset) else load_mybpmylife_mrt(dataset)
    requested = list(outcomes or _OUTCOMES)
    unknown = sorted(set(requested) - set(_OUTCOMES))
    if unknown:
        raise ValueError(f"Unknown Push MRT outcomes: {unknown}; expected {sorted(_OUTCOMES)}")
    audit = audit_mrt_dataset(mrt, split_seed=seed, test_fraction=test_fraction)
    if audit["gate_status"] != "PASS":
        return {"status": "BLOCKED", "blocker": audit["failure_mode"], "audit": audit, "outcomes": []}
    results = [_outcome_result(mrt, name, bootstrap_repetitions=bootstrap_repetitions, seed=seed + index * 1000, test_fraction=test_fraction) for index, name in enumerate(requested)]
    return {
        "status": "PASS",
        "evidence_class": "REAL_RANDOMIZED_MRT",
        "estimand": "assignment-based intent-to-treat; participant-clustered uncertainty",
        "assignment_probabilities": dict(MRT_DECLARED_ASSIGNMENT_PROBABILITIES),
        "audit": audit,
        "config": {"seed": int(seed), "bootstrap_repetitions": int(bootstrap_repetitions), "test_fraction": float(test_fraction), "outcomes": requested},
        "outcomes": results,
        "limitations": [
            "Repeated decision rows are clustered by participant; they are not independent users.",
            "The MRT is randomized over notification assignment, not continuous notification frequency.",
            "Delivery/per-protocol effects are not estimated here.",
            "No PEHE or individual counterfactual metric is available on this real dataset.",
            "Results are offline design-based evidence and not business lift.",
        ],
    }


def run_push_multitreatment_baseline(
    dataset: MRTDataset | str,
    *,
    outcome: str = "activity_steps_60min",
    seed: int = 42,
    test_fraction: float = 0.2,
) -> dict[str, Any]:
    """Compare a multi-arm outcome model with known-probability ITT.

    This is deliberately not routed through the binary propensity contract.
    It reports model-based held-out arm surfaces and the design-based arm
    means; it does not claim individual counterfactual truth for real MRT.
    """

    mrt = dataset if isinstance(dataset, MRTDataset) else load_mybpmylife_mrt(dataset)
    if outcome not in _OUTCOMES:
        raise ValueError(f"Unknown Push MRT outcome {outcome!r}.")
    data = mrt.to_causal_dataset(outcome)
    frame = _complete(data.to_pandas().reset_index(drop=True), "mrt_outcome")
    analysis = mrt.analysis_frame().reset_index(drop=True)
    train_pos, test_pos = participant_group_split(mrt, test_fraction=test_fraction, seed=seed)
    train_participants = set(analysis.iloc[train_pos]["participant_id"])
    test_participants = set(analysis.iloc[test_pos]["participant_id"])
    train = frame[frame["participant_id"].isin(train_participants)].copy()
    test = frame[frame["participant_id"].isin(test_participants)].copy()
    preprocessor = TabularPreprocessor(data.feature_cols)
    train_x = preprocessor.fit_transform(train[data.feature_cols])
    test_x = preprocessor.transform(test[data.feature_cols])
    predictions: dict[str, np.ndarray] = {}
    models: dict[str, str] = {}
    for arm in MRT_DECLARED_ASSIGNMENT_PROBABILITIES:
        mask = train["assignment_arm"].eq(arm).to_numpy()
        if int(mask.sum()) < 2:
            predictions[arm] = np.full(len(test), float(train["_outcome"].mean()))
            models[arm] = "fallback_train_mean"
        else:
            model = Ridge(alpha=10.0)
            model.fit(train_x.loc[mask], train.loc[mask, "_outcome"])
            predictions[arm] = model.predict(test_x)
            models[arm] = "ridge_per_assignment_arm_on_pre_treatment_features"
    return {
        "status": "PASS",
        "outcome": outcome,
        "treatment_arms": list(MRT_DECLARED_ASSIGNMENT_PROBABILITIES),
        "declared_assignment_probabilities": dict(MRT_DECLARED_ASSIGNMENT_PROBABILITIES),
        "propensity_contract": "binary_propensity_not_used",
        "train_participants": len(train_participants),
        "test_participants": len(test_participants),
        "participant_overlap": len(train_participants & test_participants),
        "train_rows": len(train),
        "test_rows": len(test),
        "models": models,
        "held_out_predicted_arm_means": {arm: float(np.mean(values)) for arm, values in predictions.items()},
        "known_probability_itt": _outcome_result(mrt, outcome, bootstrap_repetitions=100, seed=seed, test_fraction=test_fraction)["itt_contrasts"],
        "ground_truth": "NOT_AVAILABLE_ON_REAL_MRT",
    }


__all__ = ["run_push_mrt_baseline", "run_push_multitreatment_baseline"]
