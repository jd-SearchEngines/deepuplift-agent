"""Known-truth Push-derived semi-synthetic bias benchmark.

The covariate distribution is sampled from the real MRT's pre-treatment
context, while potential outcomes are generated before assignment and sample
selection.  Truth is kept in ``SemiSyntheticTruth`` and never added to the
model-facing ``CausalDataset``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from deepuplift.benchmarks.metrics import evaluate_prediction
from deepuplift.contracts import AssignmentType, CausalDataset, TreatmentType
from deepuplift.data.nuisance import estimate_nuisance
from deepuplift.data.preprocessing import TabularPreprocessor
from deepuplift.data.public.mrt import MRTDataset, load_mybpmylife_mrt
from deepuplift.data.schema import create_causal_dataset
from deepuplift.models import build_model

SCENARIOS = (
    "measured_confounding",
    "sample_selection_mar",
    "overlap_failure",
    "hidden_confounding",
    "sample_selection_mnar",
)


@dataclass
class SemiSyntheticTruth:
    full_frame: pd.DataFrame
    y0: np.ndarray
    y1: np.ndarray
    true_effect: np.ndarray
    propensity: np.ndarray
    observed_x_propensity: np.ndarray
    selection_probability: np.ndarray
    selected_mask: np.ndarray
    scenario: str
    seed: int
    parameters: dict[str, Any]

    @property
    def true_ate(self) -> float:
        return float(np.mean(self.true_effect))

    @property
    def selected_indices(self) -> np.ndarray:
        return np.flatnonzero(self.selected_mask)


@dataclass
class SemiSyntheticBenchmark:
    dataset: CausalDataset
    truth: SemiSyntheticTruth


def _context_features(mrt: MRTDataset, *, rows: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    frame = mrt.analysis_frame().reset_index(drop=True)
    if rows < 40:
        raise ValueError("Semi-synthetic Push benchmarks need at least 40 rows.")
    rng = np.random.default_rng(seed)
    take = int(rows)
    indices = rng.choice(len(frame), size=take, replace=take > len(frame))
    sampled = frame.iloc[indices].reset_index(drop=True)
    features = [
        column
        for column in (
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
        if column in sampled.columns
    ]
    raw = sampled[features].copy()
    categorical = [column for column in features if not pd.api.types.is_numeric_dtype(raw[column])]
    numeric = [column for column in features if column not in categorical]
    for column in numeric:
        values = pd.to_numeric(raw[column], errors="coerce")
        raw[column] = values.fillna(values.median() if values.notna().any() else 0.0)
    if categorical:
        raw[categorical] = raw[categorical].fillna("__missing__").astype(str)
    x = pd.get_dummies(raw, columns=categorical, dtype="float64")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    values = x.to_numpy(dtype="float64")
    scale = np.std(values, axis=0)
    values = (values - np.mean(values, axis=0)) / np.where(scale > 1e-8, scale, 1.0)
    columns = [f"x{i}" for i in range(values.shape[1])]
    return pd.DataFrame(values, columns=columns), indices


def generate_push_semisynthetic(
    mrt: MRTDataset | str,
    *,
    scenario: str,
    rows: int = 5000,
    seed: int = 42,
    outcome_noise: float = 0.15,
) -> SemiSyntheticBenchmark:
    """Generate a binary push-vs-none known-truth benchmark from MRT context."""

    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; expected {SCENARIOS}.")
    source = mrt if isinstance(mrt, MRTDataset) else load_mybpmylife_mrt(mrt)
    x_frame, source_indices = _context_features(source, rows=rows, seed=seed)
    x = x_frame.to_numpy(dtype="float64")
    rng = np.random.default_rng(seed + 17)
    u = rng.normal(size=len(x))
    x0 = x[:, 0]
    x1 = x[:, min(1, x.shape[1] - 1)]
    x2 = x[:, min(2, x.shape[1] - 1)]
    baseline = 1.0 + 0.15 * x0 - 0.10 * x1 + 0.06 * x2
    true_effect = 0.12 + 0.10 * np.tanh(x0) - 0.06 * x1 + 0.03 * x2
    y0 = baseline.copy()
    if scenario in {"hidden_confounding", "sample_selection_mnar"}:
        y0 = y0 + 0.40 * u
    y1 = y0 + true_effect
    observed_score = -0.15 + 0.85 * x0 - 0.60 * x1 + 0.25 * x2
    if scenario == "overlap_failure":
        observed_score = 4.5 * x0 + 1.5 * x1
    measured_score = observed_score.copy()
    if scenario in {"hidden_confounding", "sample_selection_mnar"}:
        measured_score = measured_score + 1.0 * u
    propensity = np.clip(1.0 / (1.0 + np.exp(-measured_score)), 0.01, 0.99)
    observed_x_propensity = np.clip(1.0 / (1.0 + np.exp(-observed_score)), 0.01, 0.99)
    treatment = rng.binomial(1, propensity)
    selection_score = np.zeros(len(x), dtype="float64")
    if scenario == "sample_selection_mar":
        selection_score = 0.15 + 0.95 * x0 - 0.55 * x1
    elif scenario == "sample_selection_mnar":
        selection_score = 0.15 + 0.75 * x0 + 1.10 * u
    selection_probability = np.clip(1.0 / (1.0 + np.exp(-selection_score)), 0.10, 0.95)
    selected_mask = rng.binomial(1, selection_probability).astype(bool) if scenario.startswith("sample_selection") else np.ones(len(x), dtype=bool)
    observed_y = np.where(treatment == 1, y1, y0) + rng.normal(0.0, outcome_noise, len(x))
    full = x_frame.copy()
    full["treatment"] = treatment
    full["outcome"] = observed_y
    full["mrt_source_row_index"] = source_indices
    full["source_row_index"] = np.arange(len(full), dtype="int64")
    observed = full.loc[selected_mask].reset_index(drop=True).copy()
    dataset = create_causal_dataset(
        observed,
        feature_cols=list(x_frame.columns),
        treatment_col="treatment",
        outcome_col="outcome",
        treatment_type=TreatmentType.BINARY,
        assignment_type=AssignmentType.OBSERVATIONAL,
        id_column="source_row_index",
        metadata={
            "name": f"push_semisynthetic_{scenario}",
            "source": "myBPmyLife MRT pre-treatment context distribution",
            "evidence_class": "SEMI_SYNTHETIC_KNOWN_TRUTH",
            "scenario": scenario,
            "seed": int(seed),
            "rows_full": len(full),
            "rows_observed": len(observed),
            "truth_not_in_model_frame": True,
            "assignment_mechanism": "known logistic score; hidden-U scenarios add U to assignment",
            "selection_mechanism": "known logistic selection probability; MNAR uses hidden U",
        },
    )
    truth = SemiSyntheticTruth(
        full_frame=full,
        y0=y0,
        y1=y1,
        true_effect=true_effect,
        propensity=propensity,
        observed_x_propensity=observed_x_propensity,
        selection_probability=selection_probability,
        selected_mask=selected_mask,
        scenario=scenario,
        seed=seed,
        parameters={
            "baseline": "1 + 0.15*x0 - 0.10*x1 + 0.06*x2",
            "effect": "0.12 + 0.10*tanh(x0) - 0.06*x1 + 0.03*x2",
            "outcome_hidden_u": 0.40 if scenario in {"hidden_confounding", "sample_selection_mnar"} else 0.0,
            "assignment_score": "-0.15 + 0.85*x0 - 0.60*x1 + 0.25*x2" if scenario != "overlap_failure" else "4.5*x0 + 1.5*x1",
            "observed_x_assignment_score": "same as assignment score except hidden-U term is omitted",
            "selection_score": "0.15 + 0.95*x0 - 0.55*x1" if scenario == "sample_selection_mar" else "0.15 + 0.75*x0 + 1.10*U" if scenario == "sample_selection_mnar" else "none",
            "outcome_noise_sd": outcome_noise,
        },
    )
    return SemiSyntheticBenchmark(dataset=dataset, truth=truth)


def _fit_outcome_models(frame: pd.DataFrame, feature_cols: list[str], *, sample_weight: np.ndarray | None = None):
    prep = TabularPreprocessor(feature_cols)
    x = prep.fit_transform(frame[feature_cols])
    models: dict[int, Any] = {}
    for arm in (0, 1):
        mask = frame["treatment"].to_numpy() == arm
        model = Ridge(alpha=10.0)
        model.fit(x.loc[mask], frame.loc[mask, "outcome"], sample_weight=sample_weight[mask] if sample_weight is not None else None)
        models[arm] = model
    return prep, models


def _ess(weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    return float(total * total / np.sum(weights * weights)) if np.sum(weights * weights) > 0 else 0.0


def _weighted_smd(x: np.ndarray, treatment: np.ndarray, weights: np.ndarray | None = None) -> float:
    weights = np.ones(len(x), dtype="float64") if weights is None else np.asarray(weights, dtype="float64")
    rows = []
    for column in range(x.shape[1]):
        left = treatment == 0
        right = treatment == 1
        if not left.any() or not right.any():
            continue
        mean_left = np.average(x[left, column], weights=weights[left])
        mean_right = np.average(x[right, column], weights=weights[right])
        var_left = np.average((x[left, column] - mean_left) ** 2, weights=weights[left])
        var_right = np.average((x[right, column] - mean_right) ** 2, weights=weights[right])
        scale = np.sqrt((var_left + var_right) / 2.0)
        rows.append(abs((mean_right - mean_left) / scale) if scale > 1e-8 else 0.0)
    return float(max(rows)) if rows else 0.0


def _evaluate_method(benchmark: SemiSyntheticBenchmark, method: str) -> dict[str, Any]:
    dataset, truth = benchmark.dataset, benchmark.truth
    frame = dataset.to_pandas().reset_index(drop=True)
    x = frame[dataset.feature_cols].to_numpy(dtype="float64")
    treatment = frame[dataset.treatment_col].to_numpy(dtype="int64")
    observed_indices = frame["source_row_index"].to_numpy(dtype="int64")
    oracle_assignment = method == "oracle_selection_weighted_dr"
    p = (truth.propensity if oracle_assignment else truth.observed_x_propensity)[observed_indices]
    s = truth.selection_probability[observed_indices]
    selection_weight = 1.0 / s
    if method == "unadjusted":
        estimate = float(frame.loc[treatment == 1, "outcome"].mean() - frame.loc[treatment == 0, "outcome"].mean())
        analysis_weights = np.ones(len(frame))
    else:
        use_selection = method.startswith("selection_weighted") or method == "oracle_selection_weighted_dr"
        base_weight = selection_weight if use_selection else np.ones(len(frame))
        prep, models = _fit_outcome_models(frame, dataset.feature_cols, sample_weight=base_weight if use_selection else None)
        x_observed = prep.transform(frame[dataset.feature_cols])
        m0 = models[0].predict(x_observed)
        m1 = models[1].predict(x_observed)
        if method == "outcome_regression":
            estimate = float(np.average(m1 - m0, weights=base_weight))
            analysis_weights = base_weight
        elif method.endswith("ipw"):
            weights = base_weight * (treatment / p + (1 - treatment) / (1 - p))
            treated_mean = np.sum(weights * treatment * frame["outcome"].to_numpy()) / max(np.sum(weights * treatment), 1e-8)
            control_mean = np.sum(weights * (1 - treatment) * frame["outcome"].to_numpy()) / max(np.sum(weights * (1 - treatment)), 1e-8)
            estimate = float(treated_mean - control_mean)
            analysis_weights = weights
        elif method == "stabilized_ipw":
            marginal = float(np.mean(treatment))
            weights = treatment * marginal / p + (1 - treatment) * (1 - marginal) / (1 - p)
            estimate = float(np.sum(weights * treatment * frame["outcome"]) / np.sum(weights * treatment) - np.sum(weights * (1 - treatment) * frame["outcome"]) / np.sum(weights * (1 - treatment)))
            analysis_weights = weights
        elif method == "overlap_weighting":
            weights = treatment * (1 - p) + (1 - treatment) * p
            estimate = float(np.sum(weights * treatment * frame["outcome"]) / np.sum(weights * treatment) - np.sum(weights * (1 - treatment) * frame["outcome"]) / np.sum(weights * (1 - treatment)))
            analysis_weights = weights
        else:
            residual = m1 - m0 + treatment / p * (frame["outcome"].to_numpy() - m1) - (1 - treatment) / (1 - p) * (frame["outcome"].to_numpy() - m0)
            estimate = float(np.average(residual, weights=base_weight))
            analysis_weights = base_weight
    weights = np.asarray(analysis_weights, dtype="float64")
    treated_weights = weights * treatment
    control_weights = weights * (1 - treatment)
    return {
        "method": method,
        "ate_estimate": estimate,
        "true_ate": truth.true_ate,
        "ate_bias": float(estimate - truth.true_ate),
        "ate_squared_error": float((estimate - truth.true_ate) ** 2),
        "estimand": "ATE",
        "cate_pehe_observed": None,
        "policy_value": None,
        "oracle_policy_value": None,
        "policy_regret": None,
        "cate_policy_surface": "NOT_EVALUATED; ATE estimator output is not a CATE learner",
        "effective_sample_size": {"treated": _ess(treated_weights), "control": _ess(control_weights)},
        "weight_max": float(np.max(weights)),
        "weight_variance": float(np.var(weights)),
        "balance": {"smd_max_before": _weighted_smd(x, treatment), "smd_max_after": _weighted_smd(x, treatment, weights)},
    }


CATE_LEARNERS = ("S-Learner", "T-Learner", "X-Learner", "R-Learner", "DR-Learner")


def _prediction_fingerprint(values: np.ndarray) -> str:
    rounded = np.asarray(values, dtype="float64").round(10)
    return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]


def _evaluate_cate_learner(benchmark: SemiSyntheticBenchmark, method: str) -> dict[str, Any]:
    """Evaluate one learner on a held-out observed-row split.

    This path intentionally does not reuse the ATE estimator's Ridge surface.
    R/DR receive a nuisance result fitted on the training partition only.
    """
    if method not in CATE_LEARNERS:
        raise ValueError(f"Unknown CATE learner {method!r}; expected {CATE_LEARNERS}.")
    dataset, truth = benchmark.dataset, benchmark.truth
    frame = dataset.to_pandas().reset_index(drop=True)
    treatment = frame[dataset.treatment_col].to_numpy(dtype="int64")
    indices = np.arange(len(frame))
    train_idx, test_idx = train_test_split(indices, test_size=0.25, random_state=truth.seed, stratify=treatment)
    train = dataset.subset(train_idx)
    test = dataset.subset(test_idx)
    nuisance = None
    if method in {"R-Learner", "DR-Learner"}:
        arm_counts = np.bincount(treatment[train_idx].astype(int), minlength=2)
        n_splits = min(5, int(arm_counts.min()))
        if n_splits < 2:
            raise ValueError(f"{method} needs at least two rows per treatment arm in the training split.")
        nuisance = estimate_nuisance(train, estimator="logistic", cross_fit=True, n_splits=n_splits, random_state=truth.seed)
    model = build_model(method, task="regression", random_state=truth.seed)
    model.fit(train, nuisance=nuisance)
    prediction = model.predict(test)
    observed_indices = frame.iloc[test_idx]["source_row_index"].to_numpy(dtype="int64")
    true_effect = truth.true_effect[observed_indices]
    y0 = truth.y0[observed_indices]
    y1 = truth.y1[observed_indices]
    estimated_effect = np.asarray(prediction.cate, dtype="float64")
    policy_action = estimated_effect > 0
    policy_value = float(np.mean(y0 + policy_action * true_effect))
    oracle_value = float(np.mean(np.maximum(y0, y1)))
    evaluated = evaluate_prediction(prediction, test)
    return {
        "method": method,
        "estimand": "CATE",
        "status": "PASS",
        "target_population": "observed_test_rows",
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "prediction_fingerprint": _prediction_fingerprint(estimated_effect),
        "ate_estimate_from_cate": float(np.mean(estimated_effect)),
        "true_ate_test": float(np.mean(true_effect)),
        "ate_bias_from_cate": float(np.mean(estimated_effect) - np.mean(true_effect)),
        "cate_pehe_test": float(np.sqrt(np.mean((estimated_effect - true_effect) ** 2))),
        "policy_value": policy_value,
        "oracle_policy_value": oracle_value,
        "policy_regret": float(oracle_value - policy_value),
        "ranking": evaluated["ranking"],
        "calibration": evaluated["calibration"],
        "model_metadata": prediction.metadata,
        "split": "stratified row split; model-specific prediction on held-out rows",
    }


def run_push_semisynthetic_scenario(
    mrt: MRTDataset | str,
    *,
    scenario: str,
    rows: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run one scenario and all supported identification baselines."""

    benchmark = generate_push_semisynthetic(mrt, scenario=scenario, rows=rows, seed=seed)
    methods = ["unadjusted", "outcome_regression", "ipw", "stabilized_ipw", "overlap_weighting", "dr"]
    if scenario == "sample_selection_mar":
        methods.extend(["selection_weighted_ipw", "selection_weighted_dr"])
    if scenario == "sample_selection_mnar":
        methods.extend(["selection_weighted_dr", "oracle_selection_weighted_dr"])
    ate_results = [_evaluate_method(benchmark, method) for method in methods]
    cate_results = [_evaluate_cate_learner(benchmark, method) for method in CATE_LEARNERS]
    truth = benchmark.truth
    return {
        "status": "PASS",
        "evidence_class": "SEMI_SYNTHETIC_KNOWN_TRUTH",
        "scenario": scenario,
        "seed": int(seed),
        "rows_full": len(truth.full_frame),
        "rows_observed": len(benchmark.dataset.to_pandas()),
        "selection_rate": float(np.mean(truth.selected_mask)),
        "propensity_extreme_rate_true": float(np.mean((truth.propensity < 0.05) | (truth.propensity > 0.95))),
        "propensity_extreme_rate_observed_x": float(np.mean((truth.observed_x_propensity < 0.05) | (truth.observed_x_propensity > 0.95))),
        "true_ate": truth.true_ate,
        "dgp_parameters": truth.parameters,
        "methods": ate_results,
        "ate_estimators": ate_results,
        "cate_learners": cate_results,
        "method_separation": {
            "ate_estimators": "ATE estimates, bias and weighting diagnostics only",
            "cate_learners": "held-out method-specific CATE, ranking, calibration and oracle policy diagnostics",
        },
        "hidden_confounding_limit": scenario in {"hidden_confounding", "sample_selection_mnar"},
        "mnar_interpretation": "Observed-X correction is not identified; oracle selection weighting is a sensitivity upper bound, not an estimable production method." if scenario == "sample_selection_mnar" else None,
        "deep_candidates": {"status": "NOT_RUN", "reason": "P2 gate compares auditable classical identification baselines; deep candidates are not silently treated as causal corrections."},
    }


def run_push_semisynthetic_suite(
    mrt: MRTDataset | str,
    *,
    scenarios: list[str] | None = None,
    rows: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    requested = scenarios or list(SCENARIOS)
    return {
        "status": "PASS",
        "evidence_class": "SEMI_SYNTHETIC_KNOWN_TRUTH",
        "config": {"rows": rows, "seed": seed, "scenarios": requested},
        "scenarios": [run_push_semisynthetic_scenario(mrt, scenario=scenario, rows=rows, seed=seed + i * 101) for i, scenario in enumerate(requested)],
        "limitations": [
            "Semi-synthetic results use known generated truth and are not real Push business outcomes.",
            "MRT context is repeated-decision data; the generator samples context rows and does not claim iid users.",
            "Hidden confounding remains non-identifiable from observed X alone.",
        ],
    }


__all__ = ["SCENARIOS", "SemiSyntheticBenchmark", "SemiSyntheticTruth", "generate_push_semisynthetic", "run_push_semisynthetic_scenario", "run_push_semisynthetic_suite"]
