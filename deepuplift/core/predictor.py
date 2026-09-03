from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .artifacts import load_json, load_pickle
from .data_profile import compare_feature_profile
from .policy import default_policy_fractions
from .registry import build_model


@dataclass
class UpliftScoringResult:
    run_id: str
    run_dir: str
    model_name: str
    task: str
    predictions: pd.DataFrame
    top_counts: List[Dict[str, Any]]
    shift_diagnostics: Dict[str, Any]


def _tensor_to_array(value: Any) -> np.ndarray:
    mean_attr = getattr(value, "mean", None)
    if mean_attr is not None and not callable(mean_attr) and not isinstance(value, (np.ndarray, pd.Series, pd.DataFrame)):
        value = mean_attr
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().reshape(-1)
    except ImportError:
        pass
    return np.asarray(value).reshape(-1)


def _load_model(run_dir: Path, config: Dict[str, Any]):
    sklearn_path = run_dir / "model.pkl"
    torch_path = run_dir / "model.pt"

    if sklearn_path.exists():
        payload = load_pickle(sklearn_path)
        if isinstance(payload, dict) and "model" in payload:
            return payload["model"], payload.get("metadata", {})
        return payload, {}

    if torch_path.exists():
        import torch

        checkpoint = torch.load(torch_path, map_location="cpu")
        metadata = checkpoint.get("metadata", {})
        input_dim = metadata.get("input_dim")
        if input_dim is None:
            input_dim = len(metadata.get("encoded_feature_cols", []))
        if not input_dim:
            raise ValueError("Saved torch model metadata does not include input_dim.")
        model_name = metadata.get("model_name", config.get("model_name"))
        task = metadata.get("task", config.get("task", "classification"))
        model_params = config.get("model_params", {})
        model, _ = build_model(model_name, int(input_dim), task, model_params)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model, metadata

    raise FileNotFoundError(f"No model.pkl or model.pt found in {run_dir}.")


def score_uplift_run(
    run_dir: str | Path,
    data_frame: pd.DataFrame,
    top_fractions: List[float] | None = None,
) -> UpliftScoringResult:
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    preprocessor_path = run_dir / "preprocessor.pkl"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {run_dir}.")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Missing preprocessor.pkl in {run_dir}.")

    config = load_json(config_path)
    profile_path = run_dir / "feature_profile.json"
    feature_profile = load_json(profile_path) if profile_path.exists() else {}
    preprocessor = load_pickle(preprocessor_path)
    feature_cols = list(config.get("feature_cols") or preprocessor.feature_cols)
    missing = [col for col in feature_cols if col not in data_frame.columns]
    if missing:
        raise ValueError(f"Scoring data is missing feature columns: {missing}")

    model, metadata = _load_model(run_dir, config)
    x = preprocessor.transform(data_frame)
    dummy_treatment = np.zeros(len(x), dtype="float32")
    _, y_preds, *_ = model.predict(x, dummy_treatment)
    y0_pred = _tensor_to_array(y_preds[0])
    y1_pred = _tensor_to_array(y_preds[1])

    predictions = data_frame.copy().reset_index(drop=True)
    predictions["y0_pred"] = y0_pred
    predictions["y1_pred"] = y1_pred
    predictions["uplift_score"] = predictions["y1_pred"] - predictions["y0_pred"]
    predictions = predictions.sort_values("uplift_score", ascending=False).reset_index(drop=True)
    predictions["uplift_rank"] = np.arange(1, len(predictions) + 1)
    predictions["uplift_percentile"] = predictions["uplift_rank"] / len(predictions)

    top_counts = []
    for fraction in top_fractions or [0.05, 0.1, 0.2]:
        n = max(1, int(len(predictions) * fraction))
        top = predictions.head(n)
        top_counts.append(
            {
                "top_fraction": float(fraction),
                "rows": int(n),
                "mean_predicted_uplift": float(top["uplift_score"].mean()),
            }
        )

    return UpliftScoringResult(
        run_id=run_dir.name,
        run_dir=str(run_dir),
        model_name=metadata.get("model_name", config.get("model_name")),
        task=metadata.get("task", config.get("task", "classification")),
        predictions=predictions,
        top_counts=top_counts,
        shift_diagnostics=compare_feature_profile(feature_profile, data_frame, feature_cols),
    )


def score_top_fraction(predictions: pd.DataFrame, fraction: float) -> pd.DataFrame:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")
    n = max(1, int(len(predictions) * fraction))
    return predictions.head(n).copy()


def default_scoring_policy_fractions() -> List[float]:
    return default_policy_fractions(max_fraction=1.0, step=0.05)
