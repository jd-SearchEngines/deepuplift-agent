#!/usr/bin/env python3
"""Run a design-aware Hillstrom real-randomized benchmark.

Raw Hillstrom data must be supplied locally.  The script writes only aggregate
metrics and model metadata to the requested output path.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, train_test_split

from deepuplift.benchmarks.metrics import evaluate_prediction
from deepuplift.contracts import AssignmentType, CausalDataset, TreatmentType
from deepuplift.data.nuisance.contracts import NuisanceResult
from deepuplift.data.preprocessing import TabularPreprocessor
from deepuplift.data.public.datasets import load_hillstrom
from deepuplift.data.schema import create_causal_dataset
from deepuplift.models import build_model, model_info


def _known_randomized_nuisance(dataset: CausalDataset, *, p_treated: float, seed: int, binary_outcome: bool) -> NuisanceResult:
    """Use known randomization probability; only outcome nuisances are fitted."""
    frame = dataset.to_pandas().reset_index(drop=True)
    values = sorted(frame[dataset.treatment_col].unique().tolist(), key=lambda value: str(value))
    treatment = frame[dataset.treatment_col].map({values[0]: 0, values[1]: 1}).to_numpy(dtype=int)
    y = pd.to_numeric(frame[dataset.outcome_col], errors="coerce").to_numpy(dtype=float)
    prep = TabularPreprocessor(dataset.feature_cols)
    x = prep.fit_transform(frame[dataset.feature_cols])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    m0, m1 = np.empty(len(frame)), np.empty(len(frame))
    fold_ids = np.full(len(frame), -1, dtype=int)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, treatment)):
        fold_ids[test_idx] = fold
        for arm, output in ((0, m0), (1, m1)):
            arm_idx = train_idx[treatment[train_idx] == arm]
            model = LogisticRegression(max_iter=1000, random_state=seed + fold) if binary_outcome else Ridge(alpha=1.0)
            model.fit(x.iloc[arm_idx], y[arm_idx])
            if binary_outcome:
                output[test_idx] = model.predict_proba(x.iloc[test_idx])[:, 1]
            else:
                output[test_idx] = model.predict(x.iloc[test_idx])
    propensity = np.full(len(frame), float(p_treated), dtype=float)
    return NuisanceResult(
        propensity_scores=propensity,
        propensity_model_metadata={"assignment": "known randomized probability", "p_treated": p_treated, "estimator_fit": False},
        outcome_control_oof=m0,
        outcome_treated_oof=m1,
        fold_ids=fold_ids,
        overlap_mask=np.ones(len(frame), dtype=bool),
        trim_mask=np.ones(len(frame), dtype=bool),
        sample_weights=np.ones(len(frame), dtype=float),
        effective_sample_size={"overall": float(len(frame))},
        clipping_range=None,
        diagnostics={"known_randomized_assignment": True, "propensity_distribution": {"min": p_treated, "max": p_treated}},
        metadata={"assignment_type": AssignmentType.RANDOMIZED.value, "provenance": "Hillstrom design probability"},
        propensity_preprocessor=prep,
    )


def _pairwise(dataset: CausalDataset, active_arm: int) -> CausalDataset:
    frame = dataset.to_pandas().copy()
    frame = frame[frame[dataset.treatment_col].isin([0, active_arm])].reset_index(drop=True)
    frame["pair_treatment"] = (frame[dataset.treatment_col].to_numpy() == active_arm).astype(int)
    return create_causal_dataset(
        frame,
        feature_cols=dataset.feature_cols,
        treatment_col="pair_treatment",
        outcome_col=dataset.outcome_col,
        treatment_type=TreatmentType.BINARY,
        assignment_type=AssignmentType.RANDOMIZED,
        id_column=dataset.id_column,
        metadata={**dataset.metadata, "pairwise_active_arm": active_arm, "known_pairwise_probability": .5},
    )


def _design_policy_value(prediction, dataset: CausalDataset, p_treated: float = .5) -> dict[str, float | int | None]:
    frame = dataset.to_pandas().reset_index(drop=True)
    t = frame[dataset.treatment_col].to_numpy(dtype=int)
    y = pd.to_numeric(frame[dataset.outcome_col], errors="coerce").to_numpy(dtype=float)
    action = np.asarray(prediction.cate, dtype=float) > 0
    value = np.where(action, (t == 1) * y / p_treated, (t == 0) * y / (1.0 - p_treated))
    baseline = y[t == 0].mean() if np.any(t == 0) else np.nan
    return {"policy_value": float(np.mean(value)), "policy_incremental_value": float(np.mean(value) - baseline), "target_rate": float(np.mean(action)), "baseline_control_mean": float(baseline)}


def run(path: Path, *, seeds: list[int], outcome: str, include_forest: bool = False) -> dict:
    dataset = load_hillstrom(path, multi_treatment=True)
    if outcome not in {"conversion", "visit", "spend"}:
        raise ValueError("outcome must be conversion, visit or spend")
    frame = dataset.to_pandas().copy()
    frame["selected_outcome"] = pd.to_numeric(frame[outcome], errors="coerce")
    dataset = create_causal_dataset(frame, feature_cols=dataset.feature_cols, treatment_col=dataset.treatment_col, outcome_col="selected_outcome", treatment_type=TreatmentType.MULTI_DISCRETE, assignment_type=AssignmentType.RANDOMIZED, id_column=dataset.id_column, metadata={**dataset.metadata, "outcome_definition": outcome, "evidence_class": "REAL_RANDOMIZED_EXTERNAL"})
    model_names = ["S-Learner", "T-Learner", "X-Learner", "R-Learner", "DR-Learner"]
    if include_forest:
        model_names.append("CausalForestDML")
    rows = []
    for active_arm in (1, 2):
        pair = _pairwise(dataset, active_arm)
        for seed in seeds:
            ids = np.arange(len(pair.to_pandas()))
            train_idx, test_idx = train_test_split(ids, test_size=.25, random_state=seed, stratify=pair.treatment)
            train, test = pair.subset(train_idx), pair.subset(test_idx)
            test_frame = test.to_pandas()
            t = test_frame[test.treatment_col].to_numpy(dtype=int)
            y = test_frame[test.outcome_col].to_numpy(dtype=float)
            itt = float(y[t == 1].mean() - y[t == 0].mean())
            for name in model_names:
                started = time.perf_counter()
                try:
                    info = model_info(name)
                    nuisance = _known_randomized_nuisance(train, p_treated=.5, seed=seed, binary_outcome=outcome != "spend") if name in {"R-Learner", "DR-Learner"} else None
                    model = build_model(name, task="classification" if outcome != "spend" else "regression", random_state=seed)
                    model.fit(train, nuisance=nuisance)
                    prediction = model.predict(test)
                    metrics = evaluate_prediction(prediction, test)
                    policy = _design_policy_value(prediction, test)
                    rows.append({"status": "PASS", "dataset": "Hillstrom", "evidence_class": "REAL_RANDOMIZED_EXTERNAL", "outcome": outcome, "active_arm": active_arm, "seed": seed, "model": name, "maturity": info["maturity"], "runtime_seconds": time.perf_counter() - started, "itt_test": itt, "qini": metrics["ranking"].get("qini"), "auuc": metrics["ranking"].get("auuc"), "uplift_at_10": metrics["ranking"].get("uplift_at_10"), "calibration_mae": metrics["calibration"].get("mae"), **policy, "ground_truth": "NOT_AVAILABLE: randomized aggregate outcome does not reveal individual counterfactuals"})
                except Exception as exc:  # noqa: BLE001 - preserve per-model failure evidence
                    rows.append({"status": "FAIL", "dataset": "Hillstrom", "evidence_class": "REAL_RANDOMIZED_EXTERNAL", "outcome": outcome, "active_arm": active_arm, "seed": seed, "model": name, "runtime_seconds": time.perf_counter() - started, "error_type": type(exc).__name__, "error": str(exc)[:1000]})
    return {"dataset": "Hillstrom", "evidence_class": "REAL_RANDOMIZED_EXTERNAL", "source_path": str(path), "raw_data_committed": False, "outcome": outcome, "rows": rows, "assignment": {"arms": {"0": "No E-Mail", "1": "Mens E-Mail", "2": "Womens E-Mail"}, "pairwise_probability": .5, "original_probability_each_arm": 1 / 3}, "status": "PASS" if rows and all(row["status"] == "PASS" for row in rows) else "PARTIAL"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outcome", choices=["conversion", "visit", "spend"], default="conversion")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62, 72, 82])
    parser.add_argument("--include-forest", action="store_true")
    args = parser.parse_args()
    payload = {"environment": {"python": platform.python_version(), "platform": platform.platform()}, "config": {"seeds": args.seeds, "outcome": args.outcome}, "hillstrom": run(args.path, seeds=args.seeds, outcome=args.outcome, include_forest=args.include_forest)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)) + "\n", encoding="utf-8")
    print(f"output={args.output} rows={len(payload['hillstrom']['rows'])} status={payload['hillstrom']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
