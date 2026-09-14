#!/usr/bin/env python3
"""Run the bounded synthetic benchmark used as the continuous CI gate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from deepuplift.benchmarks.continuous import run_continuous_suite, synthetic_dose_response


def main() -> int:
    result = run_continuous_suite(
        data=synthetic_dose_response(rows=360, seed=20260914),
        model_names=["DoseResponseGBM", "DRNet", "VCNet", "GIKS-DRNet", "GIKS-VCNet"],
        seed=20260914,
        model_options={
            "DRNet": {"epochs": 5, "hidden_dim": 12, "batch_size": 128},
            "VCNet": {"epochs": 5, "hidden_dim": 12, "batch_size": 128},
            "GIKS-DRNet": {"factual_epochs": 4, "giks_epochs": 3, "model_kwargs": {"hidden_dim": 12, "batch_size": 128}},
            "GIKS-VCNet": {"factual_epochs": 4, "giks_epochs": 3, "model_kwargs": {"hidden_dim": 12, "batch_size": 128}},
        },
    )
    Path("continuous-ci-benchmark.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    Path("CONTINUOUS_CI_REPORT.md").write_text(result["report"], encoding="utf-8")
    expected = {"DoseResponseGBM", "DRNet", "VCNet", "GIKS-DRNet", "GIKS-VCNet"}
    rows = {row["model"]: row for row in result["models"]}
    if set(rows) != expected:
        raise SystemExit(f"CI benchmark model set mismatch: {sorted(rows)}")
    failures = []
    for name in sorted(expected):
        row = rows[name]
        metrics = row.get("metrics") or {}
        if row.get("status") != "PASS":
            failures.append(f"{name}: status={row.get('status')} error={row.get('error')}")
        if not row.get("policy", {}).get("rows"):
            failures.append(f"{name}: no policy rows")
        if not np.isfinite(metrics.get("mise", np.nan)) or not np.isfinite(metrics.get("economic_policy_regret", np.nan)):
            failures.append(f"{name}: MISE/economic regret is not finite")
        prediction_meta = row.get("prediction_metadata", {})
        if len(row.get("curve_preview", [])) == 0 or len(row.get("curve_preview", [{}])[0].get("dose_grid", [])) < 3:
            failures.append(f"{name}: missing full dose-response curve")
        if name.startswith("GIKS-") and not prediction_meta.get("giks", {}).get("augmentation_executed"):
            failures.append(f"{name}: GIKS augmentation did not execute")
        if name.startswith("GIKS-") and prediction_meta.get("giks", {}).get("accepted_pseudo_label_count", 0) <= 0:
            failures.append(f"{name}: no accepted pseudo labels")
    if failures:
        raise SystemExit("Continuous CI benchmark gate failed:\n- " + "\n- ".join(failures))
    print("Continuous CI synthetic benchmark PASS")
    for name in sorted(expected):
        metrics = rows[name]["metrics"]
        timing = rows[name]
        print(f"{name}: MISE={metrics['mise']:.6f} economic_regret={metrics['economic_policy_regret']:.6f} runtime_s={timing['train_seconds'] + timing['predict_curve_seconds'] + timing['decision_seconds']:.3f}")
    return 0


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
