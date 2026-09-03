from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core import UpliftConfig, train_uplift_model
from deepuplift.core.model_cards import build_model_card
from deepuplift.core.registry import MODEL_REGISTRY, missing_dependencies


MODEL_NAME = "UTBoostGBM"


def tiny_dataset(rows: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(20260517)
    x0 = rng.normal(size=rows)
    x1 = rng.normal(size=rows)
    x2 = rng.binomial(1, 0.45, size=rows)
    treatment_prob = 1 / (1 + np.exp(-(0.2 * x0 - 0.1 * x1)))
    treatment = rng.binomial(1, treatment_prob)
    uplift = 0.04 + 0.08 * (x0 > 0.4) - 0.04 * (x1 < -0.8)
    base = 1 / (1 + np.exp(-(-1.7 + 0.25 * x0 + 0.2 * x2)))
    outcome_prob = np.clip(base + treatment * uplift, 0.01, 0.85)
    outcome = rng.binomial(1, outcome_prob)
    return pd.DataFrame(
        {
            "treatment": treatment,
            "outcome": outcome,
            "x0": x0,
            "x1": x1,
            "x2": x2,
            "true_uplift": uplift,
        }
    )


def main() -> None:
    failures: list[str] = []
    if MODEL_NAME not in MODEL_REGISTRY:
        failures.append(f"{MODEL_NAME} missing from registry")
    card = build_model_card(MODEL_NAME) if MODEL_NAME in MODEL_REGISTRY else {}
    for field in ["integration_gate", "source_evidence", "promotion_rule"]:
        if not card.get(field):
            failures.append(f"{MODEL_NAME} model card missing {field}")
    if "scripts/smoke_utboost_guarded_adapter.py" not in str(card.get("integration_gate", "")):
        failures.append(f"{MODEL_NAME} integration gate does not reference guarded smoke")
    if "deepuplift/core/external_models.py" not in str(card.get("source_evidence", "")):
        failures.append(f"{MODEL_NAME} source evidence does not reference external adapter")

    missing = missing_dependencies(MODEL_NAME) if MODEL_NAME in MODEL_REGISTRY else ["registry"]
    payload = {
        "model": MODEL_NAME,
        "registered": MODEL_NAME in MODEL_REGISTRY,
        "model_card": {
            "status": card.get("status"),
            "source": card.get("source"),
            "integration_gate": card.get("integration_gate"),
            "source_evidence": card.get("source_evidence"),
            "promotion_rule": card.get("promotion_rule"),
        },
        "missing_dependencies": missing,
        "status": "fail" if failures else "guarded_skip" if missing else "train_smoke_pending",
        "run_id": None,
        "metrics": {},
        "failures": failures,
    }

    if not failures and not missing:
        result = train_uplift_model(
            data_frame=tiny_dataset(),
            config=UpliftConfig(
                treatment_col="treatment",
                outcome_col="outcome",
                feature_cols=["x0", "x1", "x2"],
                model_name=MODEL_NAME,
                task="classification",
                max_rows=360,
                test_size=0.25,
                bootstrap_samples=0,
                sensitivity_samples=1,
                run_name="utboost-guarded-smoke",
                model_params={"iterations": 8, "max_depth": 3, "learning_rate": 0.05},
            ),
        )
        payload.update(
            {
                "status": "trained",
                "run_id": result.run_id,
                "run_dir": result.run_dir,
                "metrics": {
                    "qini_score": result.eval_metrics.get("qini_score"),
                    "auuc_score": result.eval_metrics.get("auuc_score"),
                    "top_k_rows": len(result.eval_metrics.get("top_k") or []),
                },
            }
        )

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    path = reports_dir / "utboost_guarded_adapter_smoke_latest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "path": str(path.relative_to(ROOT)), "missing_dependencies": missing, "failures": failures}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
