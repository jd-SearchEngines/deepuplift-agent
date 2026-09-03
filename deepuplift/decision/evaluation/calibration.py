from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from deepuplift.contracts import CausalDataset, EffectPrediction


def calibration_metrics(
    prediction: EffectPrediction,
    dataset: CausalDataset,
    *,
    buckets: int = 10,
) -> dict[str, Any]:
    frame = prediction.to_frame()
    observed = dataset.to_pandas().reset_index(drop=True)
    frame["treatment"] = observed[dataset.treatment_col].to_numpy()
    frame["outcome"] = pd.to_numeric(observed[dataset.outcome_col], errors="coerce").to_numpy()
    frame = frame.dropna(subset=["uplift", "treatment", "outcome"]).copy()
    if frame.empty:
        return {"mae": None, "buckets": []}
    frame["bucket"] = pd.qcut(frame["uplift"], q=min(buckets, len(frame)), duplicates="drop")
    rows = []
    for bucket, group in frame.groupby("bucket", observed=False):
        treated = group[group["treatment"] == 1]["outcome"]
        control = group[group["treatment"] == 0]["outcome"]
        observed_uplift = float(treated.mean() - control.mean()) if len(treated) and len(control) else None
        predicted = float(group["uplift"].mean())
        rows.append({"bucket": str(bucket), "predicted": predicted, "observed": observed_uplift, "rows": len(group)})
    errors = [abs(row["predicted"] - row["observed"]) for row in rows if row["observed"] is not None]
    paired = [(row["predicted"], row["observed"]) for row in rows if row["observed"] is not None]
    slope = None
    if len(paired) >= 3 and np.std([item[0] for item in paired]) > 1e-12:
        slope = float(np.polyfit([item[0] for item in paired], [item[1] for item in paired], 1)[0])
    # A bucket-level bootstrap interval is intentionally reported as a
    # diagnostic, not a formal clustered confidence interval.
    bootstrap = []
    rng = np.random.default_rng(42)
    for _ in range(50):
        if len(errors) < 2:
            break
        bootstrap.append(float(np.mean(rng.choice(errors, size=len(errors), replace=True))))
    return {"mae": float(np.mean(errors)) if errors else None, "calibration_slope": slope, "bootstrap_interval": [float(np.quantile(bootstrap, .025)), float(np.quantile(bootstrap, .975))] if bootstrap else None, "buckets": rows}
