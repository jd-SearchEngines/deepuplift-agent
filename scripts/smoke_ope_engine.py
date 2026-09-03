from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.artifacts import save_json
from deepuplift.core.ope import clipping_sensitivity, evaluate_ope_policy


def build_synthetic_logged_policy(seed: int = 7, rows: int = 2500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=rows)
    x1 = rng.normal(size=rows)
    actions = np.array(["cheap", "strong", "human"])
    base_logits = np.stack(
        [
            0.4 - 0.2 * x0,
            0.1 + 0.5 * x0 + 0.2 * x1,
            -0.2 + 0.3 * x1,
        ],
        axis=1,
    )
    exp_logits = np.exp(base_logits - base_logits.max(axis=1, keepdims=True))
    propensity_matrix = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    logged_idx = np.array([rng.choice(len(actions), p=propensity_matrix[i]) for i in range(rows)])
    logged_action = actions[logged_idx]
    logged_propensity = propensity_matrix[np.arange(rows), logged_idx]

    reward_mean = pd.DataFrame(
        {
            "cheap": 0.20 + 0.05 * x0,
            "strong": 0.24 + 0.10 * x0 + 0.03 * x1,
            "human": 0.18 + 0.05 * x0 + 0.12 * x1,
        }
    ).clip(0.01, 0.95)
    target_action = reward_mean.idxmax(axis=1).to_numpy()
    logged_reward_pred = reward_mean.to_numpy()[np.arange(rows), logged_idx]
    target_idx = np.array([list(actions).index(action) for action in target_action])
    target_reward_pred = reward_mean.to_numpy()[np.arange(rows), target_idx]
    reward = rng.binomial(1, logged_reward_pred)
    oracle_value = float(np.mean(target_reward_pred))

    return pd.DataFrame(
        {
            "logged_action": logged_action,
            "target_action": target_action,
            "reward": reward,
            "logged_propensity": logged_propensity,
            "logged_reward_pred": logged_reward_pred,
            "target_reward_pred": target_reward_pred,
            "oracle_target_value": oracle_value,
        }
    )


def main() -> None:
    df = build_synthetic_logged_policy()
    result = evaluate_ope_policy(
        logged_action=df["logged_action"],
        reward=df["reward"],
        logged_propensity=df["logged_propensity"],
        target_action=df["target_action"],
        target_reward_pred=df["target_reward_pred"],
        logged_reward_pred=df["logged_reward_pred"],
        n_bootstrap=30,
    )
    sensitivity = clipping_sensitivity(
        logged_action=df["logged_action"],
        reward=df["reward"],
        logged_propensity=df["logged_propensity"],
        target_action=df["target_action"],
        target_reward_pred=df["target_reward_pred"],
        logged_reward_pred=df["logged_reward_pred"],
    )
    oracle = float(df["oracle_target_value"].iloc[0])
    failures = []
    if result.get("status") != "ok":
        failures.append("OPE result status is not ok")
    if (result.get("diagnostics") or {}).get("effective_sample_size", 0) <= 0:
        failures.append("effective sample size should be positive")
    dr = (result.get("metrics") or {}).get("dr")
    if dr is None or abs(float(dr) - oracle) > 0.08:
        failures.append(f"DR estimate {dr} is too far from oracle {oracle}")
    payload = {
        "status": "fail" if failures else "ok",
        "oracle_target_value": oracle,
        "ope": result,
        "clipping_sensitivity": sensitivity,
        "failures": failures,
    }
    path = ROOT / "reports/ope_engine_smoke_latest.json"
    path.parent.mkdir(exist_ok=True)
    save_json(path, payload)
    print(json.dumps({"status": payload["status"], "path": str(path.relative_to(ROOT)), "failures": failures}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
