from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core import UpliftConfig  # noqa: E402
from deepuplift.core.doseresponse import continuous_treatment_policy_optimizer, train_dose_response_model  # noqa: E402


DATASET = ROOT / "examples" / "datasets" / "synthetic_continuous_bid_discount_uplift_5k.csv"


def _model_gate_rows() -> list[dict[str, str]]:
    return [
        {
            "model": "DoseResponseGBM",
            "status": "ready",
            "role": "local practical baseline",
            "gate": "train + score + cost-aware optimizer smoke",
            "limitation": "S-learner style outcome model; must check dose support and marginal ROI",
        },
        {
            "model": "DRNet",
            "status": "guarded",
            "role": "neural dose-response plugin path",
            "gate": "requires stable PyTorch adapter, dose encoding, factual loss and support diagnostics before promotion",
            "limitation": "paper-level architecture is valuable, but unsupported dose extrapolation can be worse than a GBM baseline",
        },
        {
            "model": "VCNet",
            "status": "guarded",
            "role": "varying-coefficient continuous-treatment network",
            "gate": "requires continuous-treatment benchmark, validation curve and optimizer evidence",
            "limitation": "smoothness and positivity assumptions must be validated for subsidy/bid/discount dose",
        },
    ]


def main() -> None:
    failures: list[dict[str, object]] = []
    if not DATASET.exists():
        failures.append({"check": "dataset_exists", "path": str(DATASET)})
        df = pd.DataFrame()
    else:
        df = pd.read_csv(DATASET).head(1200)

    if not df.empty:
        feature_cols = [
            "baseline_intent",
            "price_elasticity",
            "supply_pressure",
            "competitor_pressure",
            "margin_rate",
            "channel",
            "market",
            "saturation",
        ]
        cfg = UpliftConfig(
            treatment_col="treatment_strength",
            outcome_col="outcome",
            feature_cols=feature_cols,
            categorical_cols=["channel", "market"],
            model_name="DoseResponseGBM",
            task="classification",
            max_rows=1000,
            test_size=0.25,
            control_value=0.05,
            run_name="smoke-continuous-treatment-policy",
            random_state=7,
        )
        result = train_dose_response_model(df, cfg, model_family="gbm", grid_size=15)
        predictions = pd.read_csv(result["artifacts"]["predictions"])
        optimizer = continuous_treatment_policy_optimizer(
            predictions,
            value_per_outcome=75.0,
            cost_per_dose=18.0,
            budget=3500.0,
        )
        rows = optimizer.get("rows") or []
        best = optimizer.get("best") or {}
        if result["metrics"].get("dose_grid_size", 0) < 5:
            failures.append({"check": "dose_grid_size", "value": result["metrics"].get("dose_grid_size")})
        if len(rows) < 5:
            failures.append({"check": "optimizer_rows", "value": len(rows)})
        if best.get("net_value") is None:
            failures.append({"check": "best_net_value"})
        if optimizer.get("support", {}).get("unsafe_extrapolation_rate") is None:
            failures.append({"check": "unsafe_extrapolation_rate"})
    else:
        result = {"run_id": None, "run_dir": None, "metrics": {}, "artifacts": {}}
        optimizer = {"rows": [], "best": None, "support": {}}

    payload = {
        "status": "fail" if failures else "ok",
        "dataset": str(DATASET),
        "run_id": result.get("run_id"),
        "run_dir": result.get("run_dir"),
        "metrics": result.get("metrics"),
        "policy_rows": optimizer.get("rows") or [],
        "best_policy": optimizer.get("best"),
        "support": optimizer.get("support"),
        "diagnostic": optimizer.get("diagnostic"),
        "model_gate_rows": _model_gate_rows(),
        "failures": failures,
    }
    out = ROOT / "reports" / "continuous_treatment_policy_smoke_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_out = ROOT / "reports" / "continuous_treatment_policy_rows_latest.csv"
    if payload["policy_rows"]:
        pd.DataFrame(payload["policy_rows"]).to_csv(csv_out, index=False)
        payload["policy_csv"] = str(csv_out)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
