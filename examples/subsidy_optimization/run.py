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
from deepuplift.decision import build_continuous_policy
from deepuplift.models import build_model


def run(rows: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    dose = rng.uniform(0, 20, rows)
    x = rng.normal(size=rows)
    outcome = .25 + .05 * dose - .002 * dose**2 + .04 * x + rng.normal(0, .04, rows)
    frame = pd.DataFrame({"user_id": [f"subsidy-{i}" for i in range(rows)], "x": x, "dose": dose, "outcome": outcome})
    dataset = create_causal_dataset(frame, feature_cols=["x"], treatment_col="dose", outcome_col="outcome", treatment_type=TreatmentType.CONTINUOUS, id_column="user_id", metadata={"name": "subsidy_optimization", "source": "DeepUplift deterministic continuous reference scenario"})
    model = build_model("DoseResponseGBM")
    model.fit(dataset)
    prediction = model.predict(dataset)
    policy = build_continuous_policy(prediction, dose_cost=lambda value: .01 * value, outcome_value=1.0, budget=rows)
    print("DeepUplift Subsidy Optimization Demo")
    print(f"MODEL      {model.name}")
    print(f"DECISION   targets={policy.summary['target_count']} net_value={policy.summary['expected_net_value']:.2f} cost={policy.summary['total_cost']:.2f}")
    print("STATUS     EXPERIMENTAL / OFFLINE_ONLY")
    return dataset, prediction, policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1000)
    args = parser.parse_args()
    run(args.rows)
