"""Known-truth longitudinal Push laboratory with fatigue and carryover.

This module is intentionally semi-synthetic. It samples user context from the
MRT distribution, then simulates a sequential world with explicit potential
outcomes and state transitions. It is not evidence about real MRT long-term
effects or online business lift.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from deepuplift.data.public.mrt import MRTDataset, load_mybpmylife_mrt
from .push_semisynthetic import _context_features


LONGITUDINAL_SCENARIOS = (
    "no_fatigue_no_carryover",
    "fatigue",
    "heterogeneous_fatigue",
    "rolling_fatigue",
    "decayed_fatigue",
    "spacing",
    "carryover",
    "temporal_drift",
    "negative_effect",
    "positivity_failure",
)
_POLICIES = (
    "logged",
    "no_push",
    "random_push",
    "rule_based",
    "fatigue_aware",
    "oracle",
    "myopic_oracle",
    "dynamic_oracle",
)


@dataclass(frozen=True)
class LongitudinalPushWorld:
    """Reusable exogenous context and truth parameters for policy rollouts."""

    user_context: pd.DataFrame
    availability: np.ndarray
    logging_uniform: np.ndarray
    random_policy_uniform: np.ndarray
    scenario: str
    horizon: int
    seed: int
    parameters: dict[str, Any]

    @property
    def users(self) -> int:
        return int(len(self.user_context))

    def _state(
        self,
        user_index: int,
        time_index: int,
        prior_pushes: int,
        last_push: int,
        previous_push: int,
        push_history: tuple[int, ...] = (),
    ) -> dict[str, float]:
        row = self.user_context.iloc[user_index]
        gap = time_index - last_push if last_push >= 0 else self.horizon + 1
        x0 = float(row["x0"])
        x1 = float(row["x1"])
        x2 = float(row["x2"])
        cumulative_fatigue = float(len(push_history))
        rolling_window = int(self.parameters.get("rolling_window", 3))
        rolling_fatigue = float(sum(time_index - push_time < rolling_window for push_time in push_history))
        decay_rate = float(self.parameters.get("decay_rate", 0.5))
        decayed_fatigue = float(sum(np.exp(-decay_rate * (time_index - push_time)) for push_time in push_history))
        fatigue_mode = self.parameters.get("fatigue_mode", "cumulative")
        fatigue_load = {
            "none": 0.0,
            "cumulative": cumulative_fatigue,
            "rolling": rolling_fatigue,
            "decayed": decayed_fatigue,
        }[fatigue_mode]
        baseline = 0.30 + 0.08 * x1 + 0.03 * time_index
        baseline += float(self.parameters.get("carryover_baseline_coef", 0.0)) * previous_push
        effect = 0.16 + 0.10 * np.tanh(x0) - 0.04 * x1 + 0.02 * x2
        fatigue_coef = float(self.parameters["fatigue_coef"])
        if self.scenario == "heterogeneous_fatigue":
            fatigue_coef *= 1.0 + 0.5 * np.tanh(x0)
        effect -= fatigue_coef * fatigue_load
        effect += float(self.parameters["carryover_coef"]) * previous_push
        if self.parameters["spacing_coef"]:
            effect += float(self.parameters["spacing_coef"]) * (1.0 if gap >= 3 else -0.5)
        effect += float(self.parameters["temporal_drift_coef"]) * (time_index / max(self.horizon - 1, 1))
        if self.parameters["negative_user_coef"]:
            effect -= float(self.parameters["negative_user_coef"]) * max(0.0, -x0)
        return {
            "baseline_no_current_push": baseline,
            "true_proximal_effect": effect,
            "time_since_last_push": float(gap),
            "prior_push_count": float(prior_pushes),
            "cumulative_fatigue": cumulative_fatigue,
            "rolling_fatigue": rolling_fatigue,
            "decayed_fatigue": decayed_fatigue,
            "fatigue_load": fatigue_load,
        }

    def _dynamic_oracle_actions(self, user_index: int) -> tuple[int, ...]:
        """Return the globally best binary action sequence for one user."""

        @lru_cache(maxsize=None)
        def solve(time_index: int, push_history: tuple[int, ...]) -> tuple[float, tuple[int, ...]]:
            if time_index >= self.horizon:
                return 0.0, ()
            last_push = push_history[-1] if push_history else -1
            previous_push = int(last_push == time_index - 1)
            state = self._state(
                user_index,
                time_index,
                len(push_history),
                last_push,
                previous_push,
                push_history,
            )
            future_zero, actions_zero = solve(time_index + 1, push_history)
            candidates = [(state["baseline_no_current_push"] + future_zero, (0, *actions_zero))]
            if self.availability[user_index, time_index]:
                next_history = (*push_history, time_index)
                future_one, actions_one = solve(time_index + 1, next_history)
                candidates.append(
                    (
                        state["baseline_no_current_push"] + state["true_proximal_effect"] + future_one,
                        (1, *actions_one),
                    )
                )
            return max(candidates, key=lambda item: item[0])

        return solve(0, ())[1]

    def rollout(self, policy: str, *, noise_sd: float = 0.0) -> pd.DataFrame:
        """Simulate one policy with common exogenous availability and truth."""

        if policy not in _POLICIES:
            raise ValueError(f"Unknown policy {policy!r}; expected {_POLICIES}")
        rows: list[dict[str, Any]] = []
        for user_index in range(self.users):
            prior_pushes = 0
            last_push = -1
            previous_push = 0
            push_history: list[int] = []
            dynamic_actions = self._dynamic_oracle_actions(user_index) if policy == "dynamic_oracle" else None
            for time_index in range(self.horizon):
                state = self._state(
                    user_index,
                    time_index,
                    prior_pushes,
                    last_push,
                    previous_push,
                    tuple(push_history),
                )
                available = bool(self.availability[user_index, time_index])
                if policy == "no_push":
                    action = 0
                elif not available:
                    action = 0
                elif policy == "logged":
                    action = int(self.logging_uniform[user_index, time_index] < self._assignment_probability(user_index, time_index))
                elif policy == "random_push":
                    action = int(self.random_policy_uniform[user_index, time_index] < 0.30)
                elif policy == "rule_based":
                    action = int(float(self.user_context.iloc[user_index]["x0"]) > -0.25 and state["time_since_last_push"] >= 2)
                elif policy == "fatigue_aware":
                    action = int(
                        float(self.user_context.iloc[user_index]["x0"]) > -0.25
                        and state["prior_push_count"] < 2
                        and state["time_since_last_push"] >= 3
                    )
                elif policy in {"oracle", "myopic_oracle"}:  # truth-only diagnostic.
                    action = int(state["true_proximal_effect"] > 0.0)
                else:  # dynamic_oracle is allowed only because this is known-truth data.
                    action = int(dynamic_actions[time_index])
                y0 = state["baseline_no_current_push"]
                y1 = y0 + state["true_proximal_effect"]
                observed = y1 if action else y0
                if noise_sd > 0:
                    observed += float(np.random.default_rng(self.seed + user_index * 1009 + time_index).normal(0.0, noise_sd))
                push_probability = self._assignment_probability(user_index, time_index)
                action_probability = push_probability if action else 1.0 - push_probability
                rows.append(
                    {
                        "user_id": self.user_context.iloc[user_index]["user_id"],
                        "decision_time": time_index,
                        "availability": available,
                        "treatment_assignment": action,
                        "push_probability": push_probability,
                        "assignment_probability": action_probability if policy == "logged" else np.nan,
                        "logged_action_probability": action_probability if policy == "logged" else np.nan,
                        "previous_treatments": prior_pushes,
                        "time_since_last_push": state["time_since_last_push"],
                        "baseline_y0": y0,
                        "potential_outcome_no_push": y0,
                        "potential_outcome_push": y1,
                        "true_proximal_effect": state["true_proximal_effect"],
                        "observed_outcome": observed,
                        "fatigue_state": prior_pushes,
                        "cumulative_fatigue": state["cumulative_fatigue"],
                        "rolling_fatigue": state["rolling_fatigue"],
                        "decayed_fatigue": state["decayed_fatigue"],
                        "fatigue_load": state["fatigue_load"],
                        "carryover_state": previous_push,
                        "evidence_class": "SEMI_SYNTHETIC_KNOWN_TRUTH",
                    }
                )
                if action:
                    prior_pushes += 1
                    last_push = time_index
                    push_history.append(time_index)
                previous_push = action
        return pd.DataFrame(rows)

    def _assignment_probability(self, user_index: int, time_index: int) -> float:
        x0 = float(self.user_context.iloc[user_index]["x0"])
        if self.scenario == "positivity_failure":
            value = 1.0 / (1.0 + np.exp(-(2.8 * x0 + 0.15 * time_index)))
        else:
            value = 0.30 + 0.08 * np.tanh(x0) + 0.03 * np.sin(time_index)
        return float(np.clip(value, 0.05, 0.95))


def generate_push_longitudinal_dgp(
    mrt: MRTDataset | str,
    *,
    scenario: str = "fatigue",
    users: int = 100,
    horizon: int = 8,
    seed: int = 42,
) -> LongitudinalPushWorld:
    """Generate a sequential known-truth Push world from MRT context."""

    if scenario not in LONGITUDINAL_SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; expected {LONGITUDINAL_SCENARIOS}")
    if users < 2 or horizon < 2:
        raise ValueError("users must be at least 2 and horizon must be at least 2")
    source = mrt if isinstance(mrt, MRTDataset) else load_mybpmylife_mrt(mrt)
    # Reuse the semi-synthetic context contract, including its minimum
    # resampling size, then retain the requested number of synthetic users.
    context, _ = _context_features(source, rows=max(users, 40), seed=seed)
    context = context.iloc[:users].copy().reset_index(drop=True)
    context.insert(0, "user_id", [f"push-user-{seed}-{index}" for index in range(users)])
    rng = np.random.default_rng(seed + 313)
    availability_probability = np.clip(0.82 + 0.08 * np.tanh(context["x0"].to_numpy(dtype="float64")), 0.55, 0.98)
    availability = rng.random((users, horizon)) < availability_probability[:, None]
    logging_uniform = rng.random((users, horizon))
    random_policy_uniform = rng.random((users, horizon))
    parameters = {
        "fatigue_mode": {
            "no_fatigue_no_carryover": "none",
            "rolling_fatigue": "rolling",
            "decayed_fatigue": "decayed",
        }.get(scenario, "cumulative"),
        "fatigue_coef": 0.0 if scenario == "no_fatigue_no_carryover" else (0.035 if scenario != "heterogeneous_fatigue" else 0.06),
        "carryover_coef": 0.045 if scenario == "carryover" else 0.0,
        "carryover_baseline_coef": 0.10 if scenario == "carryover" else 0.0,
        "spacing_coef": 0.045 if scenario == "spacing" else 0.0,
        "temporal_drift_coef": 0.06 if scenario == "temporal_drift" else 0.0,
        "negative_user_coef": 0.14 if scenario == "negative_effect" else 0.0,
        "rolling_window": 3,
        "decay_rate": 0.5,
    }
    if scenario == "heterogeneous_fatigue":
        parameters["fatigue_heterogeneity"] = "0.06 * (1 + 0.5*tanh(x0))"
    else:
        parameters["fatigue_heterogeneity"] = None
    return LongitudinalPushWorld(
        user_context=context,
        availability=availability,
        logging_uniform=logging_uniform,
        random_policy_uniform=random_policy_uniform,
        scenario=scenario,
        horizon=horizon,
        seed=seed,
        parameters=parameters,
    )


def run_push_longitudinal_suite(
    mrt: MRTDataset | str,
    *,
    scenarios: list[str] | None = None,
    policies: list[str] | None = None,
    users: int = 100,
    horizon: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare baseline policies against generated cumulative truth."""

    requested_scenarios = scenarios or list(LONGITUDINAL_SCENARIOS)
    requested_policies = policies or list(_POLICIES)
    results = []
    for scenario_index, scenario in enumerate(requested_scenarios):
        world = generate_push_longitudinal_dgp(
            mrt,
            scenario=scenario,
            users=users,
            horizon=horizon,
            seed=seed + scenario_index * 101,
        )
        no_push = world.rollout("no_push")
        no_push_value = float(no_push["baseline_y0"].sum())
        policy_rows = []
        for policy in requested_policies:
            rollout = world.rollout(policy)
            truth_value = float((rollout["baseline_y0"] + rollout["treatment_assignment"] * rollout["true_proximal_effect"]).sum())
            pushes = rollout["treatment_assignment"].to_numpy(dtype="int64")
            policy_rows.append(
                {
                    "policy": policy,
                    "cumulative_policy_value": truth_value,
                    "incremental_vs_no_push": truth_value - no_push_value,
                    "push_count": int(pushes.sum()),
                    "user_coverage": float(rollout.loc[rollout["treatment_assignment"].eq(1), "user_id"].nunique() / world.users),
                    "negative_effect_pushes": int(((pushes == 1) & (rollout["true_proximal_effect"].to_numpy() < 0)).sum()),
                    "fatigue_related_utility_loss": float(max(0.0, no_push_value + (rollout["true_proximal_effect"].clip(lower=0.0).sum()) - truth_value)),
                }
            )
        results.append(
            {
                "scenario": scenario,
                "seed": world.seed,
                "users": world.users,
                "horizon": world.horizon,
                "rows": int(world.users * world.horizon),
                "evidence_class": "SEMI_SYNTHETIC_KNOWN_TRUTH",
                "parameters": world.parameters,
                "policies": policy_rows,
            }
        )
    return {
        "status": "PASS" if results else "NOT_RUN",
        "evidence_class": "SEMI_SYNTHETIC_KNOWN_TRUTH",
        "config": {"scenarios": requested_scenarios, "policies": requested_policies, "users": users, "horizon": horizon, "seed": seed},
        "scenarios": results,
        "limitations": [
            "All long-term policy values are generated-truth evidence, not real MRT causal effects or online business lift.",
            "The current laboratory uses binary push versus no-push; multi-treatment timing remains a separate gate.",
            "Oracle policy is an upper-bound diagnostic and must not be trained or evaluated as a deployable policy.",
        ],
    }


__all__ = ["LONGITUDINAL_SCENARIOS", "LongitudinalPushWorld", "generate_push_longitudinal_dgp", "run_push_longitudinal_suite"]
