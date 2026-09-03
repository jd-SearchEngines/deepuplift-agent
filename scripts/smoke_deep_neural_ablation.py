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


def _synthetic_ablation_data(n: int = 640, seed: int = 20260517) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 31)
    x = rng.normal(size=(n, 6))
    propensity = 1.0 / (1.0 + np.exp(-(0.75 * x[:, 0] - 0.55 * x[:, 1] + 0.25 * x[:, 2])))
    propensity = np.clip(propensity, 0.10, 0.90)
    treatment = rng.binomial(1, propensity)
    base = -1.1 + 0.5 * x[:, 0] - 0.4 * x[:, 3]
    cate = 0.35 + 0.55 * (x[:, 1] > 0).astype(float) - 0.35 * (x[:, 4] > 0.75).astype(float)
    outcome_prob = 1.0 / (1.0 + np.exp(-(base + treatment * cate)))
    outcome = rng.binomial(1, outcome_prob)
    data = {f"f{i}": x[:, i] for i in range(x.shape[1])}
    data.update({"treatment": treatment, "outcome": outcome, "true_uplift": cate})
    return pd.DataFrame(data)


def _config(model_name: str, feature_cols: list[str]) -> dict:
    params = {
        "share_dim": 8,
        "share_hidden_dims": [24],
        "base_hidden_dims": [16],
    }
    if model_name == "CFRNet":
        params.update({"alpha": 0.2, "ipm_mode": "mmd_rbf"})
    if model_name == "ContrastiveUpliftNet":
        params.update({"lambda_contrastive": 0.03, "temperature": 0.25, "propensity": 0.5})
    return {
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "model_name": model_name,
        "task": "classification",
        "test_size": 0.25,
        "valid_perc": 0.2,
        "epochs": 2,
        "batch_size": 96,
        "learning_rate": 0.001,
        "random_state": 20260517,
        "artifacts_dir": "reports/deep_neural_ablation_runs",
        "run_name": f"deep-ablation-{model_name.lower()}",
        "model_params": params,
        "bootstrap_samples": 0,
        "sensitivity_samples": 0,
    }


def main() -> None:
    df = _synthetic_ablation_data()
    feature_cols = [f"f{i}" for i in range(6)]
    rows = []
    failures: list[str] = []
    for model_name in ["TarNet", "CFRNet", "ContrastiveUpliftNet"]:
        result = train_uplift_model(data_frame=df, config=_config(model_name, feature_cols))
        metric_row = {
            "model": model_name,
            "run_id": result.run_id,
            "run_dir": result.run_dir,
            "qini_score": result.eval_metrics.get("qini_score"),
            "auuc_score": result.eval_metrics.get("auuc_score"),
            "oracle_top10_recall": result.eval_metrics.get("oracle_top10_recall"),
            "train_aux_loss_last": result.train_history[-1].get("train_treatment_loss") if result.train_history else None,
        }
        rows.append(metric_row)
        if metric_row["qini_score"] is None:
            failures.append(f"{model_name} missing qini_score")

    ranked = sorted(rows, key=lambda row: float(row.get("qini_score") or -1e18), reverse=True)
    out = {
        "status": "fail" if failures else "ok",
        "rows": rows,
        "ranking_by_qini": ranked,
        "best_model": ranked[0]["model"] if ranked else None,
        "failures": failures,
        "interview_line": "This ablation compares factual neural baseline, representation balance and contrastive representation under the same data/metric gate.",
    }
    Path("reports").mkdir(exist_ok=True)
    path = Path("reports/deep_neural_ablation_latest.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": out["status"], "path": str(path), "best_model": out["best_model"], "failures": failures}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
