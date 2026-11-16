"""Reproducible known-truth Push decision-chain benchmark.

This lane is deliberately independent of the blocked myBPmyLife archive. It
uses a documented Push-shaped DGP with user heterogeneity, treatment selection,
time windows, fatigue and carryover. Results are semi-synthetic evidence only;
they are not real Push lift.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from deepuplift.benchmarks.experiment_a import run_multi_comparison
from deepuplift.benchmarks.metrics import evaluate_prediction
from deepuplift.contracts import AssignmentType, TreatmentType
from deepuplift.data.schema import create_causal_dataset
from deepuplift.decision import build_push_candidate_bundle, optimize_push_candidate_bundle
from deepuplift.decision.optimization import AllocationConstraints
from deepuplift.models import build_model


@dataclass(frozen=True)
class PushE2EWorld:
    frame: pd.DataFrame
    seed: int
    horizon: int


def generate_push_e2e_world(*, users: int = 300, horizon: int = 4, seed: int = 42) -> PushE2EWorld:
    if users < 40 or horizon < 2:
        raise ValueError("users must be >=40 and horizon must be >=2")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(users, 3))
    rows: list[dict[str, Any]] = []
    for user in range(users):
        previous = 0
        pushes = 0
        for slot in range(horizon):
            x0, x1, x2 = x[user]
            baseline = 0.30 + 0.08 * x1 + 0.02 * slot + 0.05 * previous
            fatigue = 0.035 * pushes + 0.02 * max(0, pushes - 1)
            tau_push = 0.14 + 0.09 * np.tanh(x0) - 0.04 * x1 - fatigue + 0.045 * previous
            tau_rich = 0.19 - 0.05 * x1 + 0.04 * np.tanh(x2) - 0.055 * pushes + 0.02 * previous
            # Three-arm selection depends on observed covariates and time. The
            # probabilities are retained so IPS/SNIPS have an explicit design.
            logits = np.array([0.0, -0.15 + 0.35 * x0 - 0.05 * slot, 0.05 - 0.25 * x1 + 0.04 * slot])
            logits -= logits.max()
            probs = np.exp(logits) / np.exp(logits).sum()
            action = int(rng.choice(3, p=probs))
            potential = np.array([baseline, baseline + tau_push, baseline + tau_rich])
            outcome = float(potential[action] + rng.normal(0.0, 0.08))
            available = bool(rng.random() > 0.08)
            rows.append(
                {
                    "user_id": f"push-e2e-{seed}-{user}",
                    "decision_time": slot,
                    "x0": x0,
                    "x1": x1,
                    "x2": x2,
                    "previous_push": previous,
                    "push_count": pushes,
                    "available": available,
                    "treatment": action,
                    "outcome": outcome,
                    "logged_probability_none": probs[0],
                    "logged_probability_push": probs[1],
                    "logged_probability_rich": probs[2],
                    "potential_outcome_none": potential[0],
                    "potential_outcome_push": potential[1],
                    "potential_outcome_rich": potential[2],
                    "true_effect_push": tau_push,
                    "true_effect_rich": tau_rich,
                    "fatigue_load": fatigue,
                    "carryover_state": previous,
                    "evidence_class": "SEMI_SYNTHETIC_KNOWN_TRUTH",
                }
            )
            previous = int(action > 0)
            pushes += previous
    return PushE2EWorld(pd.DataFrame(rows), seed, horizon)


def _dataset(frame: pd.DataFrame, *, seed: int):
    features = ["x0", "x1", "x2", "previous_push", "push_count", "decision_time", "available"]
    return create_causal_dataset(
        frame,
        feature_cols=features,
        treatment_col="treatment",
        outcome_col="outcome",
        treatment_type=TreatmentType.MULTI_DISCRETE,
        assignment_type=AssignmentType.OBSERVATIONAL,
        id_column="user_id",
        metadata={
            "name": "push_e2e_v2_binary_view",
            "evidence_class": "SEMI_SYNTHETIC_KNOWN_TRUTH",
            "seed": seed,
            "ground_truth_available": True,
        },
    )


def _binary_view(frame: pd.DataFrame, *, seed: int, arm: int = 1):
    view = frame.copy()
    view["treatment_binary"] = (view["treatment"].to_numpy() == arm).astype(int)
    view["true_effect"] = view["true_effect_push" if arm == 1 else "true_effect_rich"]
    features = ["x0", "x1", "x2", "previous_push", "push_count", "decision_time", "available"]
    return create_causal_dataset(
        view,
        feature_cols=features,
        treatment_col="treatment_binary",
        outcome_col="outcome",
        treatment_type=TreatmentType.BINARY,
        assignment_type=AssignmentType.OBSERVATIONAL,
        id_column="user_id",
        metadata={"name": f"push_e2e_v2_arm_{arm}", "evidence_class": "SEMI_SYNTHETIC_KNOWN_TRUTH", "seed": seed},
    )


def _split_by_user(frame: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    users = np.asarray(sorted(frame["user_id"].unique()))
    rng = np.random.default_rng(seed + 991)
    rng.shuffle(users)
    cut = max(2, int(0.75 * len(users)))
    train_users = set(users[:cut])
    mask = frame["user_id"].isin(train_users).to_numpy()
    return np.flatnonzero(mask), np.flatnonzero(~mask)


def _policy_metrics(frame: pd.DataFrame, selected: np.ndarray, *, policy_name: str) -> dict[str, Any]:
    selected = np.asarray(selected, dtype=bool)
    truth = frame["true_effect_push"].to_numpy(dtype=float)
    baseline = frame["potential_outcome_none"].to_numpy(dtype=float)
    value = float(np.mean(baseline + selected * truth))
    oracle = float(np.mean(baseline + np.maximum(truth, 0.0)))
    return {
        "policy": policy_name,
        "selected_count": int(selected.sum()),
        "policy_value_truth": value,
        "oracle_value_truth": oracle,
        "regret": oracle - value,
        "selection_rate": float(selected.mean()),
    }


def _ope(frame: pd.DataFrame, selected: np.ndarray) -> dict[str, float]:
    action = (frame["treatment"].to_numpy() == 1).astype(float)
    y = frame["outcome"].to_numpy(dtype=float)
    p = frame["logged_probability_push"].to_numpy(dtype=float)
    action_prob = np.where(action == 1, p, 1.0 - p)
    observed_policy_prob = np.where(selected, action, 1.0 - action)
    logged_prob = np.where(selected, action_prob, 1.0 - action_prob)
    ratio = observed_policy_prob / np.clip(logged_prob, 1e-6, None)
    ips_terms = ratio * y
    return {
        "ips": float(np.mean(ips_terms)),
        "snips": float(np.sum(ips_terms) / max(np.sum(ratio), 1e-8)),
        "ess": float(np.sum(ratio) ** 2 / max(np.sum(ratio * ratio), 1e-8)),
        "max_weight": float(np.max(ratio)),
    }


def run_push_e2e_v2(
    *,
    users: int = 300,
    horizon: int = 4,
    seeds: Iterable[int] = (42, 52, 62, 72, 82),
    models: Iterable[str] = ("S-Learner", "T-Learner", "X-Learner", "DR-Learner"),
) -> dict[str, Any]:
    started = time.perf_counter()
    model_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    ope_rows: list[dict[str, Any]] = []
    seeds = tuple(int(seed) for seed in seeds)
    for seed in seeds:
        world = generate_push_e2e_world(users=users, horizon=horizon, seed=seed)
        train_idx, test_idx = _split_by_user(world.frame, seed)
        train_frame, test_frame = world.frame.iloc[train_idx].reset_index(drop=True), world.frame.iloc[test_idx].reset_index(drop=True)
        for name in models:
            model_started = time.perf_counter()
            try:
                train = _binary_view(train_frame, seed=seed)
                test = _binary_view(test_frame, seed=seed)
                model = build_model(name, task="regression", random_state=seed)
                model.fit(train)
                prediction = model.predict(test)
                metrics = evaluate_prediction(prediction, test, true_effect_col="true_effect")
                truth = metrics["ground_truth"]
                model_rows.append({
                    "seed": seed,
                    "model": name,
                    "status": "PASS",
                    "n_train": len(train_frame),
                    "n_test": len(test_frame),
                    "ate_bias": truth.get("ate_bias"),
                    "pehe": truth.get("pehe"),
                    "qini": metrics["ranking"].get("qini"),
                    "auuc": metrics["ranking"].get("auuc"),
                    "calibration_mae": metrics["calibration"].get("mae"),
                    "runtime_seconds": time.perf_counter() - model_started,
                    "backend": "DeepUplift native model registry",
                })
                if name == "T-Learner":
                    selected_effect = np.asarray(prediction.cate, dtype=float)
                    predictions = test_frame[["user_id", "decision_time", "available"]].rename(columns={"decision_time": "time_slot", "available": "eligible"})
                    predictions["action"] = "push"
                    predictions["predicted_incremental_effect"] = selected_effect
                    value_multiplier = 10.0
                    bundle = build_push_candidate_bundle(
                        predictions,
                        outcome="synthetic_push_outcome",
                        model_version="push_e2e_v2_t_learner",
                        eligibility_col="eligible",
                        value_multiplier=value_multiplier,
                    )
                    constraints = AllocationConstraints(
                        budget=float(max(1, int(0.20 * len(test_frame)))),
                        slot_capacity={slot: max(1, int(len(test_frame) / (horizon * 3))) for slot in range(horizon)},
                        user_frequency_cap=1,
                        minimum_interval=2,
                    )
                    optimized = optimize_push_candidate_bundle(bundle, constraints, include_exact=False)
                    greedy_selected = None
                    for solver, result in optimized["solvers"].items():
                        selected = np.zeros(len(test_frame), dtype=bool)
                        selected[np.asarray(result["selected_indices"], dtype=int)] = True
                        if solver == "greedy":
                            greedy_selected = selected.copy()
                        policy_rows.append(_policy_metrics(test_frame, selected, policy_name=solver))
                        ope_rows.append({"seed": seed, "policy": solver, **_ope(test_frame, selected)})
                        solver_rows.append({
                            "seed": seed,
                            "solver": solver,
                            "objective_net_value": result.get("objective_net_value"),
                            "selected_count": result.get("selected_count"),
                            "feasible": result.get("independent_feasibility", result.get("feasibility", {})).get("feasible"),
                            "stopping_reason": result.get("stopping_reason"),
                            "converged": result.get("converged"),
                            "runtime_seconds": result.get("runtime_seconds"),
                            "constraint_diagnostics": result.get("independent_feasibility", result.get("feasibility", {})),
                        })
                    # Policy baselines use the same held-out rows and budget
                    # count. They are kept outside the optimizer rows so a
                    # baseline cannot be mistaken for a feasible constrained
                    # solution.
                    count = int(greedy_selected.sum()) if greedy_selected is not None else 0
                    eligible = np.flatnonzero(test_frame["available"].to_numpy(dtype=bool))
                    random_selected = np.zeros(len(test_frame), dtype=bool)
                    if count and len(eligible):
                        random_selected[np.random.default_rng(seed + 7001).choice(eligible, size=min(count, len(eligible)), replace=False)] = True
                    no_push = np.zeros(len(test_frame), dtype=bool)
                    all_push = test_frame["available"].to_numpy(dtype=bool)
                    oracle_bundle = build_push_candidate_bundle(
                        predictions.assign(predicted_incremental_effect=test_frame["true_effect_push"].to_numpy()),
                        outcome="synthetic_push_outcome",
                        model_version="push_e2e_v2_oracle_diagnostic",
                        eligibility_col="eligible",
                        value_multiplier=value_multiplier,
                    )
                    oracle_result = optimize_push_candidate_bundle(oracle_bundle, constraints, solvers=("greedy",))['solvers']['greedy']
                    oracle_selected = np.zeros(len(test_frame), dtype=bool)
                    oracle_selected[np.asarray(oracle_result["selected_indices"], dtype=int)] = True
                    for policy_name, selected in (("random_same_count", random_selected), ("treat_none", no_push), ("treat_all_eligible", all_push), ("oracle_constrained", oracle_selected)):
                        policy_rows.append(_policy_metrics(test_frame, selected, policy_name=policy_name))
                        ope_rows.append({"seed": seed, "policy": policy_name, **_ope(test_frame, selected)})
            except Exception as exc:  # record and continue across model/seed failures
                model_rows.append({"seed": seed, "model": name, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}", "runtime_seconds": time.perf_counter() - model_started})

    multi = run_multi_comparison(sizes=(users,), seeds=seeds)
    feasible = all(row.get("feasible") is True for row in solver_rows)
    model_pass = any(row.get("status") == "PASS" for row in model_rows)
    return {
        "run_id": f"push_e2e_v2_{int(time.time())}",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "evidence_class": "SEMI_SYNTHETIC_KNOWN_TRUTH",
        "status": "PASS" if model_pass and feasible else "PARTIAL",
        "config": {"users": users, "horizon": horizon, "seeds": list(seeds), "models": list(models), "scenarios": ["heterogeneity", "selection", "fatigue", "carryover"], "context_source": "documented Push-shaped DGP; MRT archive not used", "value_multiplier": 10.0, "value_mapping_unit": "mapped_value_units; not revenue"},
        "model_rows": model_rows,
        "policy_rows": policy_rows,
        "ope_rows": ope_rows,
        "solver_rows": solver_rows,
        "three_arm": multi,
        "runtime_seconds": time.perf_counter() - started,
        "gates": {
            "code_pass": True,
            "test_pass": None,
            "experiment_pass": "PASS" if model_pass else "PARTIAL",
            "research_gate_pass": "PARTIAL",
            "real_mrt_lane": "BLOCKED_LICENSE_AND_LOCAL_ARCHIVE",
            "full_sequential_dr": "NOT_AVAILABLE_CONTINUATION_MODEL_MISSING",
        },
        "limitations": [
            "This is known-truth semi-synthetic evidence, not real Push business lift.",
            "The local myBPmyLife archive remains unavailable/unlicensed; no raw data was downloaded.",
            "The OPE lane is one-step binary IPS/SNIPS; it is not full sequential DR.",
            "Three-arm comparison reuses the existing DeepUplift synthetic multi-treatment runner and remains separate from the binary candidate optimizer contract.",
        ],
        "next_tasks": [
            "Add a continuation-value nuisance model before claiming full sequential DR.",
            "Reopen real MRT lane only after archive fingerprint and license are verified.",
            "Add a multi-arm candidate/OPE adapter using actual logged arm probabilities.",
        ],
    }


def render_push_e2e_report(report: dict[str, Any]) -> str:
    lines = [
        "# Batch 3 Push E2E v2 Report",
        "",
        f"Run ID: `{report['run_id']}`  ",
        f"Commit: `{report['git_sha']}`  ",
        f"Evidence class: `{report['evidence_class']}`  ",
        f"Overall status: **{report['status']}**",
        "",
        "## Scope and gates",
        "",
        "This report is a known-truth semi-synthetic DGP. It is not a claim about real Push lift or a licensed myBPmyLife analysis.",
        "",
        "| Gate | Status |",
        "|---|---|",
    ]
    lines += [f"| {key} | {value} |" for key, value in report["gates"].items()]
    lines += ["", "## Model rows", "", "| Seed | Model | Status | PEHE | ATE bias | Qini | AUUC | Runtime (s) |", "|---:|---|---|---:|---:|---:|---:|---:|"]
    for row in report["model_rows"]:
        lines.append("| {seed} | {model} | {status} | {pehe} | {ate_bias} | {qini} | {auuc} | {runtime_seconds:.3f} |".format(**{**{"pehe": "-", "ate_bias": "-", "qini": "-", "auuc": "-", "runtime_seconds": 0.0}, **row}))
    lines += ["", "## Optimizer and OPE", "", "| Seed | Solver | Feasible | Objective | Stopping reason | Converged | Runtime (s) |", "|---:|---|---|---:|---|---|---:|"]
    for row in report["solver_rows"]:
        lines.append(f"| {row['seed']} | {row['solver']} | {row['feasible']} | {row.get('objective_net_value')} | {row.get('stopping_reason')} | {row.get('converged')} | {row.get('runtime_seconds')} |")
    lines += ["", "## Policy truth value", "", "| Policy | Selected | Truth value | Oracle regret |", "|---|---:|---:|---:|"]
    for row in report["policy_rows"]:
        lines.append(f"| {row['policy']} | {row['selected_count']} | {row['policy_value_truth']:.6f} | {row['regret']:.6f} |")
    lines += ["", "## Limitations and next tasks", ""]
    lines += [f"- {item}" for item in report["limitations"]]
    lines += ["", "Next tasks:"]
    lines += [f"- {item}" for item in report["next_tasks"]]
    return "\n".join(lines) + "\n"


__all__ = ["PushE2EWorld", "generate_push_e2e_world", "run_push_e2e_v2", "render_push_e2e_report"]
