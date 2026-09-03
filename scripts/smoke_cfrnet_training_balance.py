from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.trainer import train_uplift_model


def _synthetic_binary_uplift(n: int = 720, seed: int = 20260517) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 6))
    propensity = 1.0 / (1.0 + np.exp(-(0.9 * x[:, 0] - 0.7 * x[:, 1] + 0.25 * x[:, 2])))
    propensity = np.clip(propensity, 0.08, 0.92)
    treatment = rng.binomial(1, propensity)
    base = -1.35 + 0.55 * x[:, 0] - 0.35 * x[:, 3]
    cate = 0.55 + 0.45 * (x[:, 1] > 0).astype(float) - 0.35 * (x[:, 4] > 0.7).astype(float)
    outcome_prob = 1.0 / (1.0 + np.exp(-(base + treatment * cate)))
    outcome = rng.binomial(1, outcome_prob)
    data = {f"f{i}": x[:, i] for i in range(x.shape[1])}
    data.update({"treatment": treatment, "outcome": outcome, "true_uplift": cate})
    return pd.DataFrame(data)


def main() -> None:
    df = _synthetic_binary_uplift()
    feature_cols = [f"f{i}" for i in range(6)]
    result = train_uplift_model(
        data_frame=df,
        config={
            "treatment_col": "treatment",
            "outcome_col": "outcome",
            "feature_cols": feature_cols,
            "model_name": "CFRNet",
            "task": "classification",
            "test_size": 0.25,
            "valid_perc": 0.2,
            "epochs": 2,
            "batch_size": 96,
            "learning_rate": 0.001,
            "random_state": 20260517,
            "artifacts_dir": "reports/cfrnet_balance_training_runs",
            "run_name": "cfrnet-balance-smoke",
            "model_params": {
                "share_dim": 8,
                "share_hidden_dims": [24],
                "base_hidden_dims": [16],
                "alpha": 0.2,
                "ipm_mode": "mmd_rbf",
            },
            "bootstrap_samples": 0,
            "sensitivity_samples": 0,
        },
    )
    history = result.train_history
    failures: list[str] = []
    if len(history) < 2:
        failures.append("expected at least two CFRNet training epochs")
    balance_terms = [float(row.get("train_treatment_loss", 0.0)) for row in history]
    if not any(abs(value) > 1e-8 for value in balance_terms):
        failures.append("CFRNet balance term stayed at zero")
    qini = result.eval_metrics.get("qini_score")
    if qini is None:
        failures.append("missing qini_score from training smoke metrics")

    out = {
        "status": "fail" if failures else "ok",
        "run_id": result.run_id,
        "run_dir": result.run_dir,
        "history": history,
        "balance_terms": balance_terms,
        "metrics": {
            "qini_score": qini,
            "auuc_score": result.eval_metrics.get("auuc_score"),
            "oracle_top10_recall": result.eval_metrics.get("oracle_top10_recall"),
        },
        "artifacts": result.artifacts,
        "failures": failures,
    }
    Path("reports").mkdir(exist_ok=True)
    path = Path("reports/cfrnet_training_balance_smoke_latest.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": out["status"], "path": str(path), "run_id": result.run_id, "failures": failures}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
