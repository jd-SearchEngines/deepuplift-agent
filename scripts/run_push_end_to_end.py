#!/usr/bin/env python3
"""Run the semi-synthetic Push model -> candidates -> optimizer chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepuplift.benchmarks import fit_group_crossfit_proximal, generate_push_longitudinal_dgp
from deepuplift.data import load_mybpmylife_mrt
from deepuplift.decision import build_push_candidate_bundle, optimize_push_candidate_bundle
from deepuplift.decision.optimization import AllocationConstraints


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Local MRT source used only for semi-synthetic context")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--value-multiplier", type=float, default=10.0)
    args = parser.parse_args()

    world = generate_push_longitudinal_dgp(load_mybpmylife_mrt(args.path), users=args.users, horizon=args.horizon, seed=args.seed)
    logged = world.rollout("logged")
    feature_cols = ["x0", "x1", "x2", "previous_treatments", "time_since_last_push", "fatigue_load"]
    fit = fit_group_crossfit_proximal(
        logged,
        feature_cols=feature_cols,
        group_col="user_id",
        treatment_col="treatment_assignment",
        outcome_col="observed_outcome",
        propensity_col="push_probability",
        n_splits=5,
        seed=args.seed,
    )
    predictions = logged[["user_id", "decision_time", "availability"]].rename(columns={"decision_time": "time_slot"})
    predictions["action"] = "push"
    predictions["predicted_incremental_effect"] = fit["predicted_effect"]
    bundle = build_push_candidate_bundle(
        predictions,
        outcome="synthetic_push_outcome",
        model_version="wcls_group_oof_v1",
        eligibility_col="availability",
        value_multiplier=args.value_multiplier,
    )
    constraints = AllocationConstraints(
        budget=args.budget,
        slot_capacity={slot: max(1, args.budget // max(args.horizon, 1)) for slot in range(args.horizon)},
        user_frequency_cap=2,
        minimum_interval=2,
    )
    optimized = optimize_push_candidate_bundle(bundle, constraints)
    report = _jsonable(
        {
            "status": optimized["status"],
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "evidence_class": "SEMI_SYNTHETIC_KNOWN_TRUTH",
            "config": {
                "users": args.users,
                "horizon": args.horizon,
                "seed": args.seed,
                "value_multiplier": args.value_multiplier,
                "budget": args.budget,
                "feature_cols": feature_cols,
            },
            "model": {key: value for key, value in fit.items() if key != "predicted_effect"},
            "candidate_metadata": bundle.metadata,
            "optimizer": optimized,
            "limitations": [
                "Synthetic generated-truth evidence only; not real MRT long-term effect or online lift.",
                "value_multiplier is an explicit outcome-unit mapping, not observed revenue or profit.",
            ],
        }
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
