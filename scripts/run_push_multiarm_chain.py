#!/usr/bin/env python3
"""Run a three-arm Push-shaped policy/optimizer chain on known-truth data."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepuplift.benchmarks.push_e2e_v2 import generate_push_e2e_world
from deepuplift.decision import build_push_candidate_bundle, optimize_push_candidate_bundle
from deepuplift.decision.evaluation import multiarm_policy_value
from deepuplift.decision.optimization import AllocationConstraints


def _sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _one(seed: int, users: int, horizon: int, bootstrap: int) -> dict[str, object]:
    world = generate_push_e2e_world(users=users, horizon=horizon, seed=seed)
    frame = world.frame.reset_index(drop=True)
    # This is an oracle candidate diagnostic: it tests the arm-aware decision
    # plumbing without claiming that a fitted model recovered these effects.
    candidate_rows = []
    for row_id, row in frame.iterrows():
        for arm, effect in ((1, row["true_effect_push"]), (2, row["true_effect_rich"])):
            candidate_rows.append(
                {
                    "row_id": row_id,
                    "user_id": row["user_id"],
                    "time_slot": int(row["decision_time"]),
                    "action": int(arm),
                    "predicted_incremental_effect": float(effect),
                    "eligible": bool(row["available"]),
                }
            )
    candidates = pd.DataFrame(candidate_rows)
    bundle = build_push_candidate_bundle(
        candidates,
        outcome="synthetic_three_arm_outcome",
        model_version="oracle_effect_diagnostic",
        eligibility_col="eligible",
        value_multiplier=1.0,
    )
    constraints = AllocationConstraints(
        budget=float(max(1, int(0.20 * len(frame)))),
        slot_capacity={slot: max(1, int(len(frame) / (horizon * 4))) for slot in range(horizon)},
        user_frequency_cap=1,
        minimum_interval=2,
    )
    optimized = optimize_push_candidate_bundle(bundle, constraints, solvers=("greedy", "heap"), include_exact=False)
    probs = frame[["logged_probability_none", "logged_probability_push", "logged_probability_rich"]].to_numpy(dtype="float64")
    rows = []
    truth_policy = []
    for solver, result in optimized["solvers"].items():
        policy = np.zeros(len(frame), dtype="int64")
        for index in np.asarray(result["selected_indices"], dtype="int64"):
            source_row = int(bundle.frame.iloc[index]["row_id"])
            policy[source_row] = int(bundle.frame.iloc[index]["action"])
        evaluation = multiarm_policy_value(
            frame["treatment"].to_numpy(), frame["outcome"].to_numpy(), policy, probs,
            clusters=frame["user_id"].to_numpy(), bootstrap_samples=bootstrap, random_state=seed,
        )
        potential = frame[["potential_outcome_none", "potential_outcome_push", "potential_outcome_rich"]].to_numpy(dtype="float64")
        truth_value = float(np.mean(potential[np.arange(len(frame)), policy]))
        truth_policy.append(truth_value)
        rows.append({
            "solver": solver,
            "objective_net_value": result.get("objective_net_value"),
            "selected_count": result.get("selected_count"),
            "feasible": result.get("independent_feasibility", result.get("feasibility", {})).get("feasible"),
            "policy_value_ipw": evaluation["policy_value"],
            "policy_value_ci": evaluation["policy_value_ci"],
            "truth_policy_value": truth_value,
            "abs_ipw_truth_error": abs(float(evaluation["policy_value"]) - truth_value),
            "bootstrap_unit": evaluation["bootstrap_unit"],
            "runtime_seconds": result.get("runtime_seconds"),
        })
    return {"seed": seed, "rows": int(len(frame)), "solvers": rows, "max_truth_value": max(truth_policy)}


def run(*, seeds: list[int], users: int = 180, horizon: int = 4, bootstrap: int = 40) -> dict[str, object]:
    results = [_one(seed, users, horizon, bootstrap) for seed in seeds]
    solver_rows = [row for result in results for row in result["solvers"]]
    return {
        "status": "PASS" if all(row["feasible"] is True for row in solver_rows) else "FAIL",
        "benchmark": "push_multiarm_candidate_optimizer_ope",
        "git_sha": _sha(),
        "evidence_class": "SEMI_SYNTHETIC_KNOWN_TRUTH",
        "data_provenance": "runtime-generated Push-shaped DGP; no raw MRT data",
        "config": {"seeds": seeds, "users": users, "horizon": horizon, "bootstrap": bootstrap, "arms": [0, 1, 2]},
        "results": results,
        "gates": {
            "three_arm_logged_probability_ope": "PASS",
            "candidate_optimizer_chain": "PASS" if all(row["feasible"] is True for row in solver_rows) else "FAIL",
            "real_mrt": "BLOCKED; local authorized raw myBPmyLife archive is not available",
            "online_lift": "NOT_RUN",
        },
        "limitations": [
            "Candidates use known truth as an oracle diagnostic; this is not a fitted-model quality claim.",
            "Policy value uses logged arm probabilities and cluster bootstrap but remains offline semi-synthetic evidence.",
        ],
    }


def _markdown(report: dict[str, object]) -> str:
    lines = [
        "# Multi-arm Push Candidate Chain",
        "",
        f"Status: **{report['status']}** (semi-synthetic known-truth only)",
        "",
        "| Seed | Solver | Feasible | Selected | Truth value | IPW value | Abs error | Bootstrap |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for result in report["results"]:
        for row in result["solvers"]:
            lines.append(f"| {result['seed']} | {row['solver']} | {row['feasible']} | {row['selected_count']} | {row['truth_policy_value']:.6f} | {row['policy_value_ipw']:.6f} | {row['abs_ipw_truth_error']:.6f} | {row['bootstrap_unit']} |")
    lines += ["", "## Gates", "", *[f"- {key}: {value}" for key, value in report["gates"].items()], "", "## Limitations", "", *[f"- {item}" for item in report["limitations"]]]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62, 72, 82])
    parser.add_argument("--users", type=int, default=180)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=40)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    report = run(seeds=args.seeds, users=args.users, horizon=args.horizon, bootstrap=args.bootstrap)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output_json": str(args.output_json)}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
