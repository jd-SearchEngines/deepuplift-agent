#!/usr/bin/env python3
"""Run the known-truth Sequential DR benchmark without downloading data."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Keep the documented ``python3 scripts/...`` invocation usable from a clean
# checkout where the package has not been installed into the active venv.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepuplift.core.ope import evaluate_sequential_dr_policy, evaluate_sequential_ope_policy


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _world(seed: int, *, users: int = 1200, horizon: int = 4) -> tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=users)
    rows: list[dict[str, object]] = []
    target_return = []
    for user in range(users):
        target_returns = []
        for time in range(horizon):
            target = "treat" if x[user] + 0.10 * time > 0 else "control"
            logged = "treat" if rng.random() < 0.5 else "control"
            base = 0.35 + 0.10 * x[user] + 0.02 * time
            effect = 0.22 + 0.04 * x[user] - 0.01 * time
            target_mean = base + (effect if target == "treat" else 0.0)
            logged_mean = base + (effect if logged == "treat" else 0.0)
            reward = logged_mean + rng.normal(0.0, 0.08)
            # V_t is the target-policy continuation value. Q_t is the value of
            # the observed action followed by the target policy thereafter.
            future_target = sum(
                0.35 + 0.10 * x[user] + 0.02 * future_time
                + ((0.22 + 0.04 * x[user] - 0.01 * future_time) if (x[user] + 0.10 * future_time > 0) else 0.0)
                for future_time in range(time + 1, horizon)
            )
            target_value = target_mean + future_target
            logged_q = logged_mean + future_target
            rows.append(
                {
                    "user": str(user),
                    "time": time,
                    "logged": logged,
                    "target": target,
                    "reward": reward,
                    "propensity": 0.5,
                    "target_reward": target_mean,
                    "logged_reward": logged_mean,
                    "target_value": target_value,
                    "logged_q": logged_q,
                }
            )
            target_returns.append(target_mean)
        target_return.append(float(np.sum(target_returns)))
    return pd.DataFrame(rows), float(np.mean(target_return))


def run(*, seeds: list[int], users: int = 1200, horizon: int = 4, bootstrap: int = 100) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for seed in seeds:
        frame, truth = _world(seed, users=users, horizon=horizon)
        ordinary = evaluate_sequential_ope_policy(
            logged_action=frame["logged"],
            reward=frame["reward"],
            logged_propensity=frame["propensity"],
            target_action=frame["target"],
            cluster_id=frame["user"],
            decision_time=frame["time"],
            target_reward_pred=frame["target_reward"],
            logged_reward_pred=frame["logged_reward"],
            n_bootstrap=bootstrap,
            random_state=seed,
        )
        sequential = evaluate_sequential_dr_policy(
            logged_action=frame["logged"],
            reward=frame["reward"],
            logged_propensity=frame["propensity"],
            target_action=frame["target"],
            cluster_id=frame["user"],
            decision_time=frame["time"],
            target_value_pred=frame["target_value"],
            logged_action_value_pred=frame["logged_q"],
            n_bootstrap=bootstrap,
            random_state=seed,
        )
        results.append(
            {
                "seed": seed,
                "users": users,
                "horizon": horizon,
                "truth_policy_value": truth,
                "ordinary_per_decision_dr": ordinary["metrics"].get("per_decision_dr"),
                "sequential_dr": sequential["metrics"].get("sequential_dr"),
                "trajectory_ips": sequential["metrics"].get("trajectory_ips"),
                "ordinary_diagnostics": ordinary["diagnostics"],
                "sequential_diagnostics": sequential["diagnostics"],
                "sequential_bootstrap": sequential["bootstrap"],
                "sequential_abs_error": abs(float(sequential["metrics"]["sequential_dr"]) - truth),
                "ordinary_abs_error": abs(float(ordinary["metrics"]["per_decision_dr"]) - truth),
            }
        )
    errors = [float(row["sequential_abs_error"]) for row in results]
    return {
        "status": "PASS" if max(errors) < 0.20 else "FAIL",
        "benchmark": "sequential_dr_known_truth",
        "git_sha": _git_sha(),
        "data_provenance": "runtime-generated semi-synthetic trajectories; no raw MRT data",
        "seeds": seeds,
        "users_per_seed": users,
        "horizon": horizon,
        "bootstrap_replicates": bootstrap,
        "estimand": "finite-horizon deterministic target-policy value",
        "results": results,
        "gates": {
            "known_truth_sequential_dr": "PASS" if max(errors) < 0.20 else "FAIL",
            "ordinary_dr_separate_from_sequential_dr": "PASS",
            "real_mrt": "BLOCKED; local authorized raw myBPmyLife archive is not available",
            "online_lift": "NOT_RUN",
        },
    }


def _markdown(report: dict[str, object]) -> str:
    rows = report["results"]
    lines = [
        "# Sequential DR Benchmark",
        "",
        f"Status: **{report['status']}** (known-truth semi-synthetic only)",
        "",
        "This report validates a finite-horizon sequential doubly robust estimator using supplied target-policy V and logged-action Q continuation values. It does not claim real MRT performance or online lift.",
        "",
        "| Seed | Truth | Ordinary per-decision DR | Sequential DR | Sequential abs error | Trajectory ESS | Max trajectory weight |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        diag = row["sequential_diagnostics"]
        lines.append(
            f"| {row['seed']} | {row['truth_policy_value']:.6f} | {row['ordinary_per_decision_dr']:.6f} | {row['sequential_dr']:.6f} | {row['sequential_abs_error']:.6f} | {diag['trajectory_ess']:.2f} | {diag['trajectory_weight_max']:.2f} |"
        )
    lines += [
        "",
        "## Gates",
        "",
        f"- Known-truth Sequential DR: {report['gates']['known_truth_sequential_dr']}",
        "- Ordinary DR and Sequential DR are reported as separate estimands: PASS",
        f"- Real MRT: {report['gates']['real_mrt']}",
        "- Online lift: NOT_RUN",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62, 72, 82])
    parser.add_argument("--users", type=int, default=1200)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    report = run(seeds=args.seeds, users=args.users, horizon=args.horizon, bootstrap=args.bootstrap)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "seeds": report["seeds"], "output_json": str(args.output_json)}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
