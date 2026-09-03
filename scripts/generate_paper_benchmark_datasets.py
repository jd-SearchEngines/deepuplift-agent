from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "examples" / "datasets"
MANIFEST_PATH = DATASET_DIR / "manifest.json"


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def _update_manifest(new_entries: list[dict]) -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    else:
        manifest = {"datasets": []}
    existing = {row.get("id"): row for row in manifest.get("datasets", [])}
    for entry in new_entries:
        existing[entry["id"]] = entry
    ordered = []
    seen = set()
    for row in manifest.get("datasets", []):
        dataset_id = row.get("id")
        if dataset_id in existing and dataset_id not in seen:
            ordered.append(existing[dataset_id])
            seen.add(dataset_id)
    for entry in new_entries:
        if entry["id"] not in seen:
            ordered.append(entry)
            seen.add(entry["id"])
    MANIFEST_PATH.write_text(json.dumps({"datasets": ordered}, ensure_ascii=False, indent=2), encoding="utf-8")


def write_acic_style_known_cate(rows: int = 1600, seed: int = 202605201) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, 18))
    nonlinear = np.column_stack(
        [
            x[:, 0] * x[:, 1],
            np.square(x[:, 2]),
            np.sin(x[:, 3]),
            (x[:, 4] > 0).astype(float),
            (x[:, 5] * x[:, 6] > 0.25).astype(float),
            rng.normal(size=rows),
        ]
    )
    features = np.column_stack([x, nonlinear])
    feature_cols = [f"x{i}" for i in range(1, features.shape[1] + 1)]

    base_logit = -1.15 + 0.45 * features[:, 0] - 0.35 * features[:, 2] + 0.18 * features[:, 19]
    tau = np.clip(
        0.18
        + 0.34 * (features[:, 1] > 0).astype(float)
        - 0.24 * (features[:, 4] > 1.0).astype(float)
        + 0.16 * np.tanh(features[:, 18])
        + 0.08 * features[:, 20],
        -0.32,
        0.62,
    )
    mu0 = _sigmoid(base_logit)
    mu1 = np.clip(mu0 + tau, 0.01, 0.98)
    true_uplift = mu1 - mu0
    propensity = _sigmoid(-0.25 + 0.55 * features[:, 0] - 0.42 * features[:, 1] + 0.30 * features[:, 5])
    propensity = np.clip(propensity, 0.08, 0.92)
    treatment = rng.binomial(1, propensity)
    outcome_prob = np.where(treatment == 1, mu1, mu0)
    outcome = rng.binomial(1, outcome_prob)
    oracle_policy_value = true_uplift * 12.0 - 0.15

    df = pd.DataFrame(features, columns=feature_cols)
    df["propensity"] = propensity.round(6)
    df["mu0"] = mu0.round(6)
    df["mu1"] = mu1.round(6)
    df["true_uplift"] = true_uplift.round(6)
    df["treatment"] = treatment
    df["outcome"] = outcome
    df["oracle_policy_value"] = oracle_policy_value.round(6)
    output = DATASET_DIR / "acic_style_synthetic_known_cate_1k6.csv"
    df.to_csv(output, index=False)
    return {
        "id": "acic_style_synthetic_known_cate_1k6",
        "name": "ACIC-Style Synthetic Known-CATE 1.6k",
        "kind": "paper_level_synthetic_acic_style",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "ACIC-style semi-synthetic benchmark with nonlinear response surfaces, non-random treatment assignment, known mu0/mu1 and known CATE. This is not an official ACIC raw file; it is a guarded local simulation for default regression.",
        "source_url": "https://acic.berkeley.edu/",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["TLearnerGBM", "DRLearnerGBM", "CFRNet", "DragonNet", "EFIN", "DESCN"],
        "quick_train": {"epochs": 2, "batch_size": 128, "learning_rate": 0.001, "max_rows": rows},
    }


def write_synthetic_known_cate_shift(rows: int = 1800, seed: int = 202605202) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, 10))
    segment = rng.choice(["new", "active", "lapsed"], rows, p=[0.25, 0.48, 0.27])
    high_margin = (x[:, 0] + 0.4 * x[:, 3] > 0.45).astype(float)
    lapsed = (segment == "lapsed").astype(float)
    base_logit = -1.35 + 0.55 * x[:, 0] - 0.30 * x[:, 2] + 0.22 * high_margin - 0.18 * lapsed
    tau = np.clip(
        0.04 + 0.30 * lapsed + 0.20 * high_margin + 0.16 * np.maximum(x[:, 1], 0) - 0.22 * (x[:, 4] > 1.0),
        -0.24,
        0.58,
    )
    mu0 = _sigmoid(base_logit)
    mu1 = np.clip(mu0 + tau, 0.01, 0.98)
    propensity = _sigmoid(-0.10 + 0.70 * lapsed + 0.38 * high_margin - 0.25 * x[:, 2] + 0.20 * x[:, 5])
    propensity = np.clip(propensity, 0.06, 0.94)
    treatment = rng.binomial(1, propensity)
    outcome = rng.binomial(1, np.where(treatment == 1, mu1, mu0))
    df = pd.DataFrame({f"f{i}": x[:, i] for i in range(x.shape[1])})
    df["segment"] = segment
    df["high_margin"] = high_margin
    df["propensity"] = propensity.round(6)
    df["mu0"] = mu0.round(6)
    df["mu1"] = mu1.round(6)
    df["true_uplift"] = (mu1 - mu0).round(6)
    df["oracle_policy_value"] = ((mu1 - mu0) * 20.0 - 0.25 - 0.10 * (segment == "new")).round(6)
    df["treatment"] = treatment
    df["outcome"] = outcome
    output = DATASET_DIR / "synthetic_known_cate_shift_1k8.csv"
    df.to_csv(output, index=False)
    feature_cols = [f"f{i}" for i in range(x.shape[1])] + ["segment", "high_margin"]
    return {
        "id": "synthetic_known_cate_shift_1k8",
        "name": "Synthetic Known-CATE Shift 1.8k",
        "kind": "paper_level_synthetic_known_cate",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "Known-CATE benchmark with treatment assignment shift, segment heterogeneity, mu0/mu1 and oracle policy value.",
        "source_url": "docs/UPLIFT_DEEP_MODEL_TRAINING_EVIDENCE.md",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["TLearnerGBM", "DRLearnerGBM", "CFRNet", "DragonNet", "EFIN", "DESCN"],
        "quick_train": {"epochs": 2, "batch_size": 128, "learning_rate": 0.001, "max_rows": rows},
    }


def main() -> None:
    entries = [write_acic_style_known_cate(), write_synthetic_known_cate_shift()]
    _update_manifest(entries)
    print(json.dumps({"status": "ok", "datasets": entries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
