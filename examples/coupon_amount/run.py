from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.data import TreatmentType, create_causal_dataset
from deepuplift.decision import build_multi_policy
from deepuplift.models import build_model


def run(rows: int = 1200, seed: int = 42):
    rng = np.random.default_rng(seed)
    treatment = rng.choice([0, 5, 10, 20], rows)
    x = rng.normal(size=rows)
    effects = {0: np.zeros(rows), 5: 0.12 + .02 * (x > 0), 10: 0.18 + .02 * (x > 0), 20: 0.24 + .02 * (x > 0)}
    probability = np.clip(.18 + .03 * x + np.array([effects[int(value)][i] for i, value in enumerate(treatment)]), .01, .95)
    frame = pd.DataFrame({"user_id": [f"coupon-{i}" for i in range(rows)], "x": x, "amount": treatment, "conversion": rng.binomial(1, probability)})
    dataset = create_causal_dataset(frame, feature_cols=["x"], treatment_col="amount", outcome_col="conversion", treatment_type=TreatmentType.MULTI_DISCRETE, id_column="user_id", metadata={"name": "coupon_amount", "source": "DeepUplift deterministic multi-treatment scenario"})
    model = build_model("MultiTreatmentOutcome")
    model.fit(dataset)
    prediction = model.predict(dataset)
    policy = build_multi_policy(prediction, treatment_costs={5: 1.0, 10: 3.0, 20: 10.0}, outcome_value=100.0, budget=rows * 2.0)
    print("DeepUplift Multi Coupon Amount Demo")
    print(f"MODEL      {model.name}")
    print(f"DECISION   targets={policy.summary['target_count']} net_value={policy.summary['expected_net_value']:.2f} cost={policy.summary['total_cost']:.2f}")
    print("STATUS     EXPERIMENTAL / OFFLINE_ONLY")
    return dataset, prediction, policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1200)
    args = parser.parse_args()
    run(args.rows)
