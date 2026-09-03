from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .artifacts import create_run_dir, load_json, load_pickle, save_dataframe, save_json, save_pickle, save_text
from .preprocess import TabularPreprocessor
from .schema import UpliftConfig
from .sklearn_models import _default_outcome_estimator, _fit_estimator, _predict_outcome
from .trainer import _infer_task, _plain_value, _prepare_target, _set_seed


def _dose_grid(values: pd.Series, grid_size: int = 21, dose_min: float | None = None, dose_max: float | None = None) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype("float64")
    if numeric.empty:
        raise ValueError("Continuous treatment column has no numeric values.")
    lower = float(numeric.quantile(0.01)) if dose_min is None else float(dose_min)
    upper = float(numeric.quantile(0.99)) if dose_max is None else float(dose_max)
    if lower == upper:
        lower, upper = float(numeric.min()), float(numeric.max())
    if lower == upper:
        raise ValueError("Continuous treatment needs at least two distinct numeric values.")
    return np.linspace(lower, upper, int(max(3, grid_size)))


def _append_dose(x: pd.DataFrame, dose: float) -> pd.DataFrame:
    enriched = x.copy()
    enriched["__dose__"] = float(dose)
    return enriched


def _observed_by_dose_bin(df: pd.DataFrame, treatment_col: str, outcome_col: str, bins: int = 10) -> list[dict]:
    work = df[[treatment_col, outcome_col]].copy()
    try:
        work["dose_bin"] = pd.qcut(work[treatment_col], bins, duplicates="drop")
    except ValueError:
        return []
    rows = (
        work.groupby("dose_bin", observed=True)
        .agg(rows=(outcome_col, "size"), treatment_mean=(treatment_col, "mean"), outcome_mean=(outcome_col, "mean"))
        .reset_index()
    )
    rows["dose_bin"] = rows["dose_bin"].astype(str)
    return rows.to_dict(orient="records")


def _predict_dose_grid(model: Any, x: pd.DataFrame, dose_values: np.ndarray, task: str) -> pd.DataFrame:
    predictions = {}
    for dose in dose_values:
        predictions[f"y_pred_dose_{dose:.6g}"] = _predict_outcome(model, _append_dose(x, float(dose)), task)
    return pd.DataFrame(predictions)


def _score_dose_response(
    source_df: pd.DataFrame,
    x: pd.DataFrame,
    model: Any,
    dose_values: np.ndarray,
    task: str,
    baseline_dose: float,
) -> pd.DataFrame:
    grid_predictions = _predict_dose_grid(model, x, dose_values, task)
    values = grid_predictions.to_numpy()
    best_idx = values.argmax(axis=1)
    baseline_idx = int(np.argmin(np.abs(dose_values - baseline_dose)))

    predictions = source_df.copy().reset_index(drop=True)
    predictions["recommended_dose"] = dose_values[best_idx]
    predictions["recommended_outcome_pred"] = values[np.arange(len(values)), best_idx]
    predictions["baseline_dose"] = float(dose_values[baseline_idx])
    predictions["baseline_outcome_pred"] = values[:, baseline_idx]
    predictions["predicted_gain_vs_baseline"] = predictions["recommended_outcome_pred"] - predictions["baseline_outcome_pred"]
    predictions = predictions.sort_values("predicted_gain_vs_baseline", ascending=False).reset_index(drop=True)
    predictions["dose_policy_rank"] = np.arange(1, len(predictions) + 1)
    predictions["dose_policy_percentile"] = predictions["dose_policy_rank"] / len(predictions)
    return pd.concat([predictions, grid_predictions.reset_index(drop=True)], axis=1)


def _build_dose_note(config_payload: dict, metrics: dict) -> str:
    best = metrics.get("average_best_dose") or {}
    lines = [
        "# DeepUplift Dose-Response Run Note",
        "",
        "## Configuration",
        "",
        f"- Model: `{config_payload.get('model_name')}`",
        f"- Task: `{config_payload.get('task')}`",
        f"- Treatment: `{config_payload.get('treatment_col')}`",
        f"- Outcome: `{config_payload.get('outcome_col')}`",
        f"- Dose grid size: {len(config_payload.get('dose_grid') or [])}",
        "",
        "## Policy Summary",
        "",
        f"- Baseline dose: {metrics.get('baseline_dose')}",
        f"- Mean recommended dose: {best.get('mean_recommended_dose')}",
        f"- Mean predicted gain vs baseline: {best.get('mean_predicted_gain_vs_baseline')}",
        "",
        "## Caution",
        "",
        "This is a dose-response policy model based on observed continuous treatment support. Validate overlap across dose ranges before deployment.",
        "",
    ]
    return "\n".join(lines)


def continuous_treatment_policy_optimizer(
    predictions: pd.DataFrame,
    value_per_outcome: float = 100.0,
    cost_per_dose: float = 20.0,
    budget: float | None = None,
    baseline_col: str = "baseline_dose",
    recommended_col: str = "recommended_dose",
    gain_col: str = "predicted_gain_vs_baseline",
    observed_dose_col: str = "treatment",
) -> Dict[str, Any]:
    """Turn dose-response predictions into a cost-aware continuous policy.

    This is the practical optimizer behind the DRNet/VCNet plugin path: before
    swapping in a neural dose-response estimator, the platform needs a stable
    policy layer that can explain marginal gain, budget and support risk.
    """

    if predictions.empty:
        return {"status": "empty", "rows": [], "best": None, "support": {}}
    required = {baseline_col, recommended_col, gain_col}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Dose predictions missing required columns: {missing}")

    work = predictions.copy()
    baseline = pd.to_numeric(work[baseline_col], errors="coerce").fillna(0.0)
    recommended = pd.to_numeric(work[recommended_col], errors="coerce").fillna(baseline)
    gain = pd.to_numeric(work[gain_col], errors="coerce").fillna(0.0)
    incremental_dose = (recommended - baseline).clip(lower=0.0)
    incremental_cost = incremental_dose * float(cost_per_dose)
    gross_value = gain * float(value_per_outcome)
    net_value = gross_value - incremental_cost
    work["dose_increment"] = incremental_dose
    work["dose_policy_cost"] = incremental_cost
    work["dose_policy_gross_value"] = gross_value
    work["dose_policy_net_value"] = net_value
    work["dose_marginal_roi"] = gross_value / incremental_cost.replace(0.0, np.nan)
    work["dose_marginal_roi"] = work["dose_marginal_roi"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    support: dict[str, Any] = {}
    if observed_dose_col in work.columns:
        observed = pd.to_numeric(work[observed_dose_col], errors="coerce").dropna()
        if not observed.empty:
            p05 = float(observed.quantile(0.05))
            p95 = float(observed.quantile(0.95))
            unsafe_mask = (recommended < p05) | (recommended > p95)
            work["unsafe_extrapolation"] = unsafe_mask
            support = {
                "observed_p05": p05,
                "observed_p95": p95,
                "unsafe_extrapolation_rate": float(unsafe_mask.mean()),
            }
        else:
            work["unsafe_extrapolation"] = False
    else:
        work["unsafe_extrapolation"] = False

    ranked = work.sort_values("dose_policy_net_value", ascending=False).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    fractions = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
    for fraction in fractions:
        top = ranked.head(max(1, int(len(ranked) * fraction)))
        cost_sum = float(top["dose_policy_cost"].sum())
        if budget is not None and float(budget) > 0:
            within_budget = top[top["dose_policy_cost"].cumsum() <= float(budget)]
            if not within_budget.empty:
                top = within_budget
                cost_sum = float(top["dose_policy_cost"].sum())
        gross_sum = float(top["dose_policy_gross_value"].sum())
        net_sum = float(top["dose_policy_net_value"].sum())
        rows.append(
            {
                "top_fraction": fraction,
                "selected_rows": int(len(top)),
                "mean_recommended_dose": float(top[recommended_col].mean()),
                "mean_dose_increment": float(top["dose_increment"].mean()),
                "gross_value": gross_sum,
                "dose_cost": cost_sum,
                "net_value": net_sum,
                "roi": gross_sum / cost_sum if cost_sum > 1e-9 else None,
                "unsafe_extrapolation_rate": float(top["unsafe_extrapolation"].mean()) if len(top) else None,
            }
        )

    valid_rows = [row for row in rows if row.get("net_value") is not None]
    best = max(valid_rows, key=lambda row: row["net_value"]) if valid_rows else None
    return {
        "status": "ok",
        "rows": rows,
        "best": best,
        "support": support,
        "value_per_outcome": float(value_per_outcome),
        "cost_per_dose": float(cost_per_dose),
        "budget": None if budget is None else float(budget),
        "diagnostic": "High unsafe_extrapolation_rate means DRNet/VCNet or any dose-response model is extrapolating outside observed support.",
    }


def train_dose_response_model(
    data_frame: pd.DataFrame,
    config: UpliftConfig | Dict[str, Any],
    model_family: str = "gbm",
    grid_size: int = 21,
) -> Dict[str, Any]:
    if isinstance(config, dict):
        config = UpliftConfig.from_dict(config)

    _set_seed(config.random_state)
    df = data_frame.copy()
    if config.max_rows:
        df = df.head(config.max_rows).copy()

    required_cols = [config.treatment_col, config.outcome_col] + list(config.feature_cols)
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")
    if not pd.api.types.is_numeric_dtype(df[config.treatment_col]):
        raise ValueError("Dose-response workflow requires a numeric continuous treatment column.")

    task = _infer_task(df[config.outcome_col], config.task)
    working = df[required_cols].dropna(subset=[config.treatment_col, config.outcome_col]).copy()
    working["dose"] = pd.to_numeric(working[config.treatment_col], errors="coerce")
    working = working.dropna(subset=["dose"]).copy()
    working["outcome"], outcome_mapping = _prepare_target(working[config.outcome_col], task, config.outcome_col)

    dose_values = _dose_grid(working["dose"], grid_size=grid_size)
    baseline_dose = float(config.control_value) if config.control_value is not None else float(dose_values.min())

    split_df = working[config.feature_cols + ["dose", "outcome"]].copy()
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(split_df, test_size=config.test_size, random_state=config.random_state)
    preprocessor = TabularPreprocessor(config.feature_cols, categorical_cols=config.categorical_cols)
    X_train_features = preprocessor.fit_transform(train_df).reset_index(drop=True)
    X_test_features = preprocessor.transform(test_df).reset_index(drop=True)
    X_train = X_train_features.copy()
    X_train["__dose__"] = train_df["dose"].reset_index(drop=True).astype("float32")
    y_train = train_df["outcome"]

    forest = model_family == "forest"
    outcome_estimator = _default_outcome_estimator(task, config.model_params, forest=forest)
    model = _fit_estimator(outcome_estimator, X_train, y_train, task)

    prediction_source = pd.DataFrame(
        {
            "treatment": test_df["dose"].reset_index(drop=True),
            "outcome": test_df["outcome"].reset_index(drop=True),
        }
    )
    prediction_source = pd.concat([prediction_source, test_df[config.feature_cols].reset_index(drop=True)], axis=1)
    prediction_df = _score_dose_response(
        prediction_source,
        X_test_features,
        model,
        dose_values=dose_values,
        task=task,
        baseline_dose=baseline_dose,
    )

    curve_rows = []
    grid_cols = [col for col in prediction_df.columns if col.startswith("y_pred_dose_")]
    for dose, col in zip(dose_values, grid_cols):
        curve_rows.append({"dose": float(dose), "mean_predicted_outcome": float(prediction_df[col].mean())})

    best_summary = {
        "mean_recommended_dose": float(prediction_df["recommended_dose"].mean()),
        "median_recommended_dose": float(prediction_df["recommended_dose"].median()),
        "mean_predicted_gain_vs_baseline": float(prediction_df["predicted_gain_vs_baseline"].mean()),
        "top10_mean_predicted_gain_vs_baseline": float(prediction_df.head(max(1, int(len(prediction_df) * 0.1)))["predicted_gain_vs_baseline"].mean()),
    }
    policy_optimizer = continuous_treatment_policy_optimizer(prediction_df)
    metrics = {
        "model_name": config.model_name,
        "task": task,
        "strategy": "dose_response_s_learner",
        "baseline_dose": baseline_dose,
        "dose_min": float(dose_values.min()),
        "dose_max": float(dose_values.max()),
        "dose_grid_size": int(len(dose_values)),
        "average_best_dose": best_summary,
        "observed_by_dose_bin": _observed_by_dose_bin(working, "dose", "outcome"),
        "dose_response_curve": curve_rows,
        "policy_optimizer": policy_optimizer,
    }

    run_id, run_dir = create_run_dir(config.artifacts_dir, config.run_name or f"dose-{config.model_name.lower()}")
    config_payload = config.to_dict()
    config_payload.update(
        {
            "task": task,
            "encoded_feature_cols": preprocessor.output_feature_cols,
            "strategy": "dose_response_s_learner",
            "dose_grid": dose_values.tolist(),
            "baseline_dose": baseline_dose,
            "outcome_mapping": outcome_mapping,
        }
    )
    note = _build_dose_note(config_payload, metrics)
    artifacts = {
        "config": save_json(run_dir / "config.json", config_payload),
        "metrics": save_json(run_dir / "metrics.json", metrics),
        "curves": save_json(run_dir / "curves.json", {"dose_response_curve": curve_rows}),
        "predictions": save_dataframe(run_dir / "predictions.csv", prediction_df),
        "run_note": save_text(run_dir / "run_note.md", note),
        "preprocessor": save_pickle(run_dir / "preprocessor.pkl", preprocessor),
        "model": save_pickle(
            run_dir / "model.pkl",
            {
                "model": model,
                "dose_grid": dose_values,
                "baseline_dose": baseline_dose,
                "task": task,
                "strategy": "dose_response_s_learner",
            },
        ),
    }
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "config": config_payload,
        "metrics": metrics,
        "artifacts": artifacts,
        "preview": prediction_df.head(50).to_dict(orient="records"),
    }


def score_dose_response_run(run_dir: str | Path, data_frame: pd.DataFrame) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    config = load_json(run_dir / "config.json")
    preprocessor = load_pickle(run_dir / "preprocessor.pkl")
    payload = load_pickle(run_dir / "model.pkl")
    feature_cols = list(config.get("feature_cols") or preprocessor.feature_cols)
    missing = [col for col in feature_cols if col not in data_frame.columns]
    if missing:
        raise ValueError(f"Scoring data is missing feature columns: {missing}")

    x = preprocessor.transform(data_frame)
    predictions = _score_dose_response(
        data_frame,
        x,
        payload["model"],
        dose_values=np.asarray(payload["dose_grid"], dtype="float64"),
        task=payload.get("task", config.get("task", "classification")),
        baseline_dose=float(payload.get("baseline_dose", config.get("baseline_dose", 0.0))),
    )
    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "model_name": config.get("model_name", "DoseResponseGBM"),
        "task": payload.get("task", config.get("task", "classification")),
        "strategy": payload.get("strategy", "dose_response_s_learner"),
        "predictions": predictions,
    }
