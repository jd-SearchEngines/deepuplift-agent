#!/usr/bin/env python3
"""Run a streamed, design-aware Criteo v2.1 binary benchmark.

The raw Criteo file is never copied into the repository.  The runner first
draws an exact uniform row sample from the compressed CSV, then reuses the
same sample for every model and seed at a given size.  ``exposure`` is kept
for audit metadata and is never used as a pre-treatment feature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepuplift.benchmarks.metrics import evaluate_prediction
from deepuplift.contracts import AssignmentType, CausalDataset, TreatmentType
from deepuplift.data.nuisance.contracts import NuisanceResult
from deepuplift.data.preprocessing import TabularPreprocessor
from deepuplift.data.schema import create_causal_dataset
from deepuplift.models import build_model, model_info

DEFAULT_TOTAL_ROWS = 13_979_592
FEATURES = [f"f{i}" for i in range(12)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _uniform_sample(path: Path, *, rows: int, seed: int, total_rows: int) -> pd.DataFrame:
    if rows > total_rows:
        raise ValueError(f"rows={rows} exceeds total_rows={total_rows}")
    selected = np.sort(np.random.default_rng(seed).choice(total_rows, size=rows, replace=False))
    parts: list[pd.DataFrame] = []
    offset = 0
    for chunk in pd.read_csv(path, compression="infer", chunksize=250_000):
        end = offset + len(chunk)
        left = np.searchsorted(selected, offset, side="left")
        right = np.searchsorted(selected, end, side="left")
        if right > left:
            parts.append(chunk.iloc[selected[left:right] - offset].copy())
        offset = end
    if offset != total_rows:
        raise ValueError(f"expected {total_rows} rows but streamed {offset}")
    return pd.concat(parts, ignore_index=True)


def _dataset(frame: pd.DataFrame, outcome: str) -> CausalDataset:
    if outcome not in {"conversion", "visit"}:
        raise ValueError("outcome must be conversion or visit")
    selected = frame.copy()
    selected["selected_outcome"] = pd.to_numeric(selected[outcome], errors="coerce")
    return create_causal_dataset(
        selected,
        feature_cols=FEATURES,
        treatment_col="treatment",
        outcome_col="selected_outcome",
        treatment_type=TreatmentType.BINARY,
        assignment_type=AssignmentType.RANDOMIZED,
        id_column=None,
        metadata={
            "name": "criteo_uplift_v2_1",
            "evidence_class": "REAL_RANDOMIZED_EXTERNAL",
            "outcome_definition": outcome,
            "randomization_probability": 0.85,
            "exposure_excluded_from_features": True,
        },
    )


def _known_randomized_nuisance(dataset: CausalDataset, *, p_treated: float, seed: int) -> NuisanceResult:
    """Cross-fit outcome nuisances while keeping the known design propensity."""
    frame = dataset.to_pandas().reset_index(drop=True)
    treatment = frame[dataset.treatment_col].to_numpy(dtype="int64")
    outcome = frame[dataset.outcome_col].to_numpy(dtype="float64")
    prep = TabularPreprocessor(dataset.feature_cols)
    x = prep.fit_transform(frame[dataset.feature_cols])
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    m0, m1 = np.empty(len(frame)), np.empty(len(frame))
    fold_ids = np.full(len(frame), -1, dtype="int64")
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, treatment)):
        fold_ids[test_idx] = fold
        for arm, output in ((0, m0), (1, m1)):
            arm_idx = train_idx[treatment[train_idx] == arm]
            estimator = LogisticRegression(max_iter=200, random_state=seed + fold)
            estimator.fit(x.iloc[arm_idx], outcome[arm_idx])
            output[test_idx] = estimator.predict_proba(x.iloc[test_idx])[:, 1]
    return NuisanceResult(
        propensity_scores=np.full(len(frame), p_treated, dtype="float64"),
        propensity_model_metadata={"assignment": "known randomized probability", "p_treated": p_treated, "estimator_fit": False},
        outcome_control_oof=m0,
        outcome_treated_oof=m1,
        fold_ids=fold_ids,
        overlap_mask=np.ones(len(frame), dtype=bool),
        trim_mask=np.ones(len(frame), dtype=bool),
        sample_weights=np.ones(len(frame), dtype="float64"),
        effective_sample_size={"overall": float(len(frame))},
        clipping_range=None,
        diagnostics={"known_randomized_assignment": True, "propensity_distribution": {"min": p_treated, "max": p_treated}},
        metadata={"assignment_type": AssignmentType.RANDOMIZED.value, "provenance": "Criteo published randomization ratio"},
        propensity_preprocessor=prep,
    )


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024 * 1024) if platform.system() == "Darwin" else value / 1024


def _run_one(dataset: CausalDataset, *, model_name: str, seed: int, p_treated: float) -> dict:
    frame = dataset.to_pandas().reset_index(drop=True)
    indices = np.arange(len(frame))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed, stratify=frame[dataset.treatment_col])
    train, test = dataset.subset(train_idx), dataset.subset(test_idx)
    started = time.perf_counter()
    try:
        nuisance = _known_randomized_nuisance(train, p_treated=p_treated, seed=seed) if model_name in {"R-Learner", "DR-Learner"} else None
        model = build_model(model_name, task="classification", random_state=seed, outcome_model="hist-gradient-boosting")
        model.fit(train, nuisance=nuisance)
        prediction = model.predict(test)
        test_frame = test.to_pandas().reset_index(drop=True)
        treatment = test_frame[dataset.treatment_col].to_numpy(dtype="int64")
        outcome = test_frame[dataset.outcome_col].to_numpy(dtype="float64")
        factual = np.where(treatment == 1, prediction.y1, prediction.y0)
        action = np.asarray(prediction.cate) > 0
        design_value = np.mean(np.where(action, treatment * outcome / p_treated, (1 - treatment) * outcome / (1 - p_treated)))
        control_mean = outcome[treatment == 0].mean()
        metrics = evaluate_prediction(prediction, test, include_ranking_curves=False)
        return {
            "status": "PASS",
            "model": model_name,
            "seed": seed,
            "rows": len(frame),
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "maturity": model_info(model_name)["maturity"],
            "runtime_seconds": time.perf_counter() - started,
            "peak_rss_mb": _peak_rss_mb(),
            "itt_test": float(outcome[treatment == 1].mean() - control_mean),
            "factual_auc": float(roc_auc_score(outcome, factual)),
            "factual_logloss": float(log_loss(outcome, np.clip(factual, 1e-6, 1 - 1e-6))),
            "qini": metrics["ranking"].get("qini"),
            "auuc": metrics["ranking"].get("auuc"),
            "uplift_at_10": metrics["ranking"].get("uplift_at_10"),
            "calibration_mae": metrics["calibration"].get("mae"),
            "policy_value_ipw": float(design_value),
            "policy_increment_vs_control": float(design_value - control_mean),
            "target_rate": float(np.mean(action)),
            "evidence_class": "REAL_RANDOMIZED_EXTERNAL",
            "ground_truth": "NOT_AVAILABLE: randomized aggregate outcomes do not reveal individual counterfactuals",
        }
    except Exception as exc:  # noqa: BLE001 - preserve per-model failure evidence
        return {
            "status": "FAIL",
            "model": model_name,
            "seed": seed,
            "rows": len(frame),
            "runtime_seconds": time.perf_counter() - started,
            "peak_rss_mb": _peak_rss_mb(),
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
            "failure_class": "IMPLEMENTATION_FAILURE" if isinstance(exc, (ValueError, FloatingPointError)) else "UNKNOWN",
        }


def run(path: Path, *, outcomes: list[str], sizes: list[int], seeds: list[int], models: list[str], total_rows: int) -> dict:
    raw_hash = _sha256(path)
    all_rows = []
    for size in sizes:
        sampled = _uniform_sample(path, rows=size, seed=42 + size, total_rows=total_rows)
        for outcome in outcomes:
            dataset = _dataset(sampled, outcome)
            for seed in seeds:
                for model_name in models:
                    row = _run_one(dataset, model_name=model_name, seed=seed, p_treated=0.85)
                    row.update({"dataset": "Criteo Uplift v2.1", "outcome": outcome, "evidence_class": "REAL_RANDOMIZED_EXTERNAL"})
                    all_rows.append(row)
    return {
        "status": "PASS" if all(row["status"] == "PASS" for row in all_rows) else "PARTIAL",
        "dataset": "Criteo Uplift v2.1",
        "source_path": str(path),
        "source_sha256": raw_hash,
        "raw_data_committed": False,
        "config": {"outcomes": outcomes, "sizes": sizes, "seeds": seeds, "models": models, "total_rows": total_rows, "features": FEATURES, "exposure_used_as_feature": False, "split": "stratified row split; same sampled rows and test split per model within seed"},
        "rows": all_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outcomes", nargs="+", choices=["conversion", "visit"], default=["conversion"])
    parser.add_argument("--sizes", type=int, nargs="+", default=[100_000, 500_000, 1_000_000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    parser.add_argument("--models", nargs="+", default=["S-Learner", "T-Learner", "X-Learner", "R-Learner", "DR-Learner"])
    parser.add_argument("--total-rows", type=int, default=DEFAULT_TOTAL_ROWS)
    args = parser.parse_args()
    payload = run(args.path, outcomes=args.outcomes, sizes=args.sizes, seeds=args.seeds, models=args.models, total_rows=args.total_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"output={args.output} rows={len(payload['rows'])} status={payload['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
