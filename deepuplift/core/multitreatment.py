from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .artifacts import create_run_dir, load_json, load_pickle, save_dataframe, save_json, save_pickle, save_text
from .preprocess import TabularPreprocessor
from .schema import UpliftConfig
from .sklearn_models import _default_effect_estimator, _default_outcome_estimator, _fit_estimator, _predict_outcome
from .trainer import _infer_task, _plain_value, _prepare_target, _set_seed


def _safe_label(value: Any) -> str:
    return str(value).replace(" ", "_").replace("/", "_")


def _aligned_propensity_matrix(propensity_model: Any, x: Any, treatment_values: list[Any]) -> np.ndarray:
    propensities = propensity_model.predict_proba(x)
    logistic = getattr(propensity_model, "named_steps", {}).get("logisticregression")
    propensity_classes = list(logistic.classes_ if logistic is not None else propensity_model.classes_)
    aligned = np.zeros((len(x), len(treatment_values)), dtype="float64")
    for idx, value in enumerate(treatment_values):
        if value in propensity_classes:
            aligned[:, idx] = propensities[:, propensity_classes.index(value)]
        else:
            aligned[:, idx] = 1e-6
    return aligned


def _predict_multi_action_outcomes(
    model_payload: dict,
    x: Any,
    treatment_values: list[Any],
    control_label: Any,
    task: str,
) -> dict[Any, np.ndarray]:
    models = model_payload["models"]
    effect_models = model_payload.get("effect_models") or {}
    predictions = {}

    if effect_models:
        control_pred = _predict_outcome(models[control_label], x, task)
        predictions[control_label] = control_pred
        for value in treatment_values:
            if value == control_label:
                continue
            uplift_pred = np.asarray(effect_models[value].predict(x)).reshape(-1)
            outcome_pred = control_pred + uplift_pred
            if task == "classification":
                outcome_pred = np.clip(outcome_pred, 0.0, 1.0)
            predictions[value] = outcome_pred
        return predictions

    for value in treatment_values:
        predictions[value] = _predict_outcome(models[value], x, task)
    return predictions


def _top_action_rows(prediction_df: pd.DataFrame, action_labels: list[Any], control_label: Any) -> list[dict]:
    rows = []
    for action in action_labels:
        if action == control_label:
            continue
        uplift_col = f"uplift_{_safe_label(action)}_vs_control"
        top = prediction_df.sort_values(uplift_col, ascending=False).head(max(1, int(len(prediction_df) * 0.1)))
        rows.append(
            {
                "treatment": _plain_value(action),
                "top_fraction": 0.1,
                "rows": int(len(top)),
                "mean_predicted_uplift": float(top[uplift_col].mean()),
                "actual_treated_rows": int((top["treatment"] == action).sum()),
            }
        )
    return rows


def _ipw_policy_value(
    prediction_df: pd.DataFrame,
    propensity: np.ndarray,
    treatment_values: list[Any],
    control_label: Any,
    outcome_col: str = "outcome",
) -> dict:
    action_index = {value: idx for idx, value in enumerate(treatment_values)}
    observed_t = prediction_df["treatment"].map(action_index).to_numpy()
    recommended_t = prediction_df["recommended_treatment"].map(action_index).to_numpy()
    control_index = action_index[control_label]
    outcome = prediction_df[outcome_col].to_numpy(dtype="float64")

    clipped = np.clip(propensity, 0.02, 0.98)
    recommended_prob = clipped[np.arange(len(prediction_df)), recommended_t]
    control_prob = clipped[np.arange(len(prediction_df)), control_index]
    recommended_value = np.mean(np.where(observed_t == recommended_t, outcome / recommended_prob, 0.0))
    control_value = np.mean(np.where(observed_t == control_index, outcome / control_prob, 0.0))
    return {
        "recommended_policy_value_ipw": float(recommended_value),
        "control_policy_value_ipw": float(control_value),
        "incremental_policy_value_ipw": float(recommended_value - control_value),
    }


def _multi_propensity_diagnostics(propensity: np.ndarray, treatment_values: list[Any]) -> list[dict]:
    rows = []
    for idx, value in enumerate(treatment_values):
        probs = propensity[:, idx]
        rows.append(
            {
                "treatment": _plain_value(value),
                "min": float(np.min(probs)),
                "p05": float(np.quantile(probs, 0.05)),
                "median": float(np.median(probs)),
                "p95": float(np.quantile(probs, 0.95)),
                "max": float(np.max(probs)),
                "low_propensity_rate": float(np.mean(probs < 0.05)),
                "deterministic_rate": float(np.mean(probs > 0.95)),
            }
        )
    return rows


def _bootstrap_multi_policy_value(
    prediction_df: pd.DataFrame,
    propensity: np.ndarray,
    treatment_values: list[Any],
    control_label: Any,
    n_bootstrap: int,
    random_state: int,
) -> dict:
    if n_bootstrap <= 0 or len(prediction_df) == 0:
        return {}

    rng = np.random.default_rng(random_state)
    rows = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(prediction_df), len(prediction_df))
        sample_df = prediction_df.iloc[indices].reset_index(drop=True)
        sample_propensity = propensity[indices]
        try:
            rows.append(_ipw_policy_value(sample_df, sample_propensity, treatment_values, control_label))
        except Exception:
            continue

    if not rows:
        return {"n_bootstrap": int(n_bootstrap), "n_success": 0, "metrics": {}}

    boot = pd.DataFrame(rows)
    summary = {}
    for key in [
        "recommended_policy_value_ipw",
        "control_policy_value_ipw",
        "incremental_policy_value_ipw",
    ]:
        values = boot[key].dropna()
        if values.empty:
            continue
        summary[key] = {
            "mean": float(values.mean()),
            "low": float(values.quantile(0.025)),
            "high": float(values.quantile(0.975)),
        }
    return {
        "n_bootstrap": int(n_bootstrap),
        "n_success": int(len(boot)),
        "metrics": summary,
    }


def _score_multi_predictions(
    source_df: pd.DataFrame,
    x: Any,
    model_payload: dict,
    treatment_values: list[Any],
    control_label: Any,
    task: str,
) -> pd.DataFrame:
    predictions = source_df.copy().reset_index(drop=True)
    action_predictions = _predict_multi_action_outcomes(model_payload, x, treatment_values, control_label, task)

    for value in treatment_values:
        predictions[f"y_pred_{_safe_label(value)}"] = action_predictions[value]

    control_col = f"y_pred_{_safe_label(control_label)}"
    for value in treatment_values:
        if value == control_label:
            continue
        predictions[f"uplift_{_safe_label(value)}_vs_control"] = predictions[f"y_pred_{_safe_label(value)}"] - predictions[control_col]

    y_cols = [f"y_pred_{_safe_label(value)}" for value in treatment_values]
    best_index = predictions[y_cols].to_numpy().argmax(axis=1)
    predictions["recommended_treatment"] = [treatment_values[idx] for idx in best_index]
    predictions["recommended_outcome_pred"] = predictions[y_cols].max(axis=1)
    predictions["recommended_uplift_vs_control"] = predictions["recommended_outcome_pred"] - predictions[control_col]
    predictions = predictions.sort_values("recommended_uplift_vs_control", ascending=False).reset_index(drop=True)
    predictions["uplift_rank"] = np.arange(1, len(predictions) + 1)
    predictions["uplift_percentile"] = predictions["uplift_rank"] / len(predictions)
    return predictions


def _recommended_action_counts(prediction_df: pd.DataFrame) -> list[dict]:
    counts = prediction_df["recommended_treatment"].value_counts().rename_axis("treatment").reset_index(name="rows")
    counts["fraction"] = counts["rows"] / len(prediction_df)
    counts["treatment"] = counts["treatment"].map(_plain_value)
    return counts.to_dict(orient="records")


def _top_recommended_rows(prediction_df: pd.DataFrame, top_fractions: list[float]) -> list[dict]:
    rows = []
    for fraction in top_fractions:
        n_rows = max(1, int(len(prediction_df) * fraction))
        top = prediction_df.head(n_rows)
        action_counts = _recommended_action_counts(top)
        rows.append(
            {
                "top_fraction": float(fraction),
                "rows": int(n_rows),
                "mean_recommended_uplift": float(top["recommended_uplift_vs_control"].mean()),
                "mean_recommended_outcome_pred": float(top["recommended_outcome_pred"].mean()),
                "recommended_action_counts": action_counts,
            }
        )
    return rows


def _build_multi_note(config_payload: dict, metrics: dict) -> str:
    lines = [
        "# DeepUplift Multi-Treatment Run Note",
        "",
        "## Configuration",
        "",
        f"- Model: `{config_payload.get('model_name')}`",
        f"- Task: `{config_payload.get('task')}`",
        f"- Treatment: `{config_payload.get('treatment_col')}`",
        f"- Outcome: `{config_payload.get('outcome_col')}`",
        f"- Control treatment: `{metrics.get('control_treatment')}`",
        f"- Treatment values: {metrics.get('treatment_values')}",
        "",
        "## Policy",
        "",
        f"- Recommended policy value IPW: {metrics.get('recommended_policy_value_ipw')}",
        f"- Control policy value IPW: {metrics.get('control_policy_value_ipw')}",
        f"- Incremental policy value IPW: {metrics.get('incremental_policy_value_ipw')}",
        "",
        "## Policy Value Confidence",
        "",
    ]
    bootstrap = metrics.get("policy_value_bootstrap") or {}
    bootstrap_metrics = bootstrap.get("metrics") or {}
    if bootstrap_metrics:
        for key, value in bootstrap_metrics.items():
            lines.append(f"- `{key}`: mean {value.get('mean')}, 95% CI [{value.get('low')}, {value.get('high')}]")
    else:
        lines.append("- Not computed. Set `bootstrap_samples` above 0 to enable.")
    lines.extend(
        [
            "",
            "## Action Propensity Overlap",
            "",
        ]
    )
    for row in metrics.get("propensity_by_action", []):
        lines.append(
            f"- `{row.get('treatment')}`: p05 {row.get('p05')}, median {row.get('median')}, "
            f"p95 {row.get('p95')}, low propensity rate {row.get('low_propensity_rate'):.1%}"
        )
    lines.extend(
        [
            "",
            "## Recommended Action Mix",
            "",
        ]
    )
    for row in metrics.get("recommended_action_counts", []):
        lines.append(f"- `{row.get('treatment')}`: {row.get('rows')} rows ({row.get('fraction'):.1%})")
    lines.extend(
        [
            "",
            "## Caution",
            "",
            "Multi-treatment policy value here is an IPW holdout estimate and depends on observed overlap for every action. Use it as a decision aid, not as proof of identification.",
            "",
        ]
    )
    return "\n".join(lines)


def train_multi_treatment_model(
    data_frame: pd.DataFrame,
    config: UpliftConfig | Dict[str, Any],
    model_family: str = "gbm",
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

    task = _infer_task(df[config.outcome_col], config.task)
    working = df[required_cols].dropna(subset=[config.treatment_col, config.outcome_col]).copy()
    treatment_values = sorted(working[config.treatment_col].dropna().unique().tolist(), key=lambda item: str(item))
    if len(treatment_values) <= 2:
        raise ValueError(f"Multi-treatment workflow requires more than two treatment values; got {treatment_values}.")
    control_label = config.control_value if config.control_value in treatment_values else treatment_values[0]
    working["outcome"], outcome_mapping = _prepare_target(working[config.outcome_col], task, config.outcome_col)

    split_df = working[config.feature_cols + [config.treatment_col, "outcome"]].copy()
    train_df, test_df = train_test_split(split_df, test_size=config.test_size, random_state=config.random_state)

    preprocessor = TabularPreprocessor(config.feature_cols, categorical_cols=config.categorical_cols)
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    y_train = train_df["outcome"]
    t_train = train_df[config.treatment_col]

    forest = model_family in {"forest", "dr_forest"}
    doubly_robust = model_family in {"dr_gbm", "dr_forest"}
    outcome_estimator = _default_outcome_estimator(task, config.model_params, forest=forest)
    effect_estimator = _default_effect_estimator(config.model_params, forest=forest)
    models = {}
    for value in treatment_values:
        mask = t_train == value
        models[value] = _fit_estimator(outcome_estimator, X_train[mask], y_train[mask], task)

    propensity_model = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=2000))
    propensity_model.fit(X_train, t_train)
    aligned_train_propensities = _aligned_propensity_matrix(propensity_model, X_train, treatment_values)
    aligned_propensities = _aligned_propensity_matrix(propensity_model, X_test, treatment_values)

    effect_models = {}
    if doubly_robust:
        y_array = y_train.to_numpy(dtype="float64")
        t_array = t_train.to_numpy()
        control_index = treatment_values.index(control_label)
        control_train_pred = _predict_outcome(models[control_label], X_train, task)
        control_e = np.clip(aligned_train_propensities[:, control_index], 0.02, 0.98)
        for value in treatment_values:
            if value == control_label:
                continue
            action_index = treatment_values.index(value)
            action_train_pred = _predict_outcome(models[value], X_train, task)
            action_e = np.clip(aligned_train_propensities[:, action_index], 0.02, 0.98)
            pseudo_effect = (
                action_train_pred
                - control_train_pred
                + (t_array == value) * (y_array - action_train_pred) / action_e
                - (t_array == control_label) * (y_array - control_train_pred) / control_e
            )
            effect_models[value] = _fit_estimator(effect_estimator, X_train, pseudo_effect, "regression")

    model_payload = {
        "models": models,
        "effect_models": effect_models,
        "propensity_model": propensity_model,
        "treatment_values": treatment_values,
        "control_treatment": control_label,
        "task": task,
        "strategy": "doubly_robust" if doubly_robust else "t_learner",
    }

    prediction_df = pd.DataFrame(
        {
            "treatment": test_df[config.treatment_col].reset_index(drop=True),
            "outcome": test_df["outcome"].reset_index(drop=True),
        }
    )
    action_predictions = _predict_multi_action_outcomes(model_payload, X_test, treatment_values, control_label, task)
    for value in treatment_values:
        col = f"y_pred_{_safe_label(value)}"
        prediction_df[col] = action_predictions[value]

    control_col = f"y_pred_{_safe_label(control_label)}"
    for value in treatment_values:
        if value == control_label:
            continue
        prediction_df[f"uplift_{_safe_label(value)}_vs_control"] = prediction_df[f"y_pred_{_safe_label(value)}"] - prediction_df[control_col]

    y_cols = [f"y_pred_{_safe_label(value)}" for value in treatment_values]
    best_index = prediction_df[y_cols].to_numpy().argmax(axis=1)
    prediction_df["recommended_treatment"] = [treatment_values[idx] for idx in best_index]
    prediction_df["recommended_outcome_pred"] = prediction_df[y_cols].max(axis=1)
    prediction_df["recommended_uplift_vs_control"] = prediction_df["recommended_outcome_pred"] - prediction_df[control_col]
    prediction_df = pd.concat([prediction_df, test_df[config.feature_cols].reset_index(drop=True)], axis=1)

    counts = prediction_df["recommended_treatment"].value_counts().rename_axis("treatment").reset_index(name="rows")
    counts["fraction"] = counts["rows"] / len(prediction_df)
    observed_by_treatment = (
        prediction_df.groupby("treatment", observed=True)["outcome"].agg(["count", "mean"]).reset_index().rename(columns={"count": "rows", "mean": "outcome_mean"})
    )
    policy_values = _ipw_policy_value(prediction_df, aligned_propensities, treatment_values, control_label)
    metrics = {
        "model_name": config.model_name,
        "strategy": model_payload["strategy"],
        "task": task,
        "control_treatment": _plain_value(control_label),
        "treatment_values": [_plain_value(value) for value in treatment_values],
        "observed_by_treatment": observed_by_treatment.to_dict(orient="records"),
        "recommended_action_counts": counts.to_dict(orient="records"),
        "top_actions": _top_action_rows(prediction_df, treatment_values, control_label),
        "propensity_by_action": _multi_propensity_diagnostics(aligned_propensities, treatment_values),
        "policy_value_bootstrap": _bootstrap_multi_policy_value(
            prediction_df,
            aligned_propensities,
            treatment_values,
            control_label,
            n_bootstrap=int(config.bootstrap_samples or 0),
            random_state=int(config.random_state),
        ),
        **policy_values,
    }

    run_id, run_dir = create_run_dir(config.artifacts_dir, config.run_name or f"multi-{config.model_name.lower()}")
    config_payload = config.to_dict()
    config_payload.update(
        {
            "task": task,
            "encoded_feature_cols": preprocessor.output_feature_cols,
            "treatment_values": [_plain_value(value) for value in treatment_values],
            "control_treatment": _plain_value(control_label),
            "strategy": model_payload["strategy"],
            "outcome_mapping": outcome_mapping,
        }
    )
    note = _build_multi_note(config_payload, metrics)
    artifacts = {
        "config": save_json(run_dir / "config.json", config_payload),
        "metrics": save_json(run_dir / "metrics.json", metrics),
        "predictions": save_dataframe(run_dir / "predictions.csv", prediction_df),
        "run_note": save_text(run_dir / "run_note.md", note),
        "preprocessor": save_pickle(run_dir / "preprocessor.pkl", preprocessor),
        "model": save_pickle(
            run_dir / "model.pkl",
            model_payload,
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


def score_multi_treatment_run(
    run_dir: str | Path,
    data_frame: pd.DataFrame,
    top_fractions: list[float] | None = None,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    model_path = run_dir / "model.pkl"
    preprocessor_path = run_dir / "preprocessor.pkl"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {run_dir}.")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.pkl in {run_dir}.")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Missing preprocessor.pkl in {run_dir}.")

    config = load_json(config_path)
    model_payload = load_pickle(model_path)
    preprocessor = load_pickle(preprocessor_path)
    if not isinstance(model_payload, dict) or "models" not in model_payload:
        raise ValueError(f"{run_dir} does not look like a multi-treatment run.")

    models = model_payload["models"]
    treatment_values = list(model_payload.get("treatment_values") or models.keys())
    control_label = model_payload.get("control_treatment", treatment_values[0])
    task = model_payload.get("task", config.get("task", "classification"))
    if len(treatment_values) <= 2:
        raise ValueError(f"Multi-treatment scoring requires more than two treatment values; got {treatment_values}.")

    feature_cols = list(config.get("feature_cols") or preprocessor.feature_cols)
    missing = [col for col in feature_cols if col not in data_frame.columns]
    if missing:
        raise ValueError(f"Scoring data is missing feature columns: {missing}")

    x = preprocessor.transform(data_frame)
    predictions = _score_multi_predictions(
        source_df=data_frame,
        x=x,
        model_payload=model_payload,
        treatment_values=treatment_values,
        control_label=control_label,
        task=task,
    )
    top_fractions = top_fractions or [0.05, 0.1, 0.2]
    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "model_name": config.get("model_name", "MultiTLearner"),
        "strategy": model_payload.get("strategy", "t_learner"),
        "task": task,
        "control_treatment": _plain_value(control_label),
        "treatment_values": [_plain_value(value) for value in treatment_values],
        "predictions": predictions,
        "top_counts": _top_recommended_rows(predictions, top_fractions),
        "recommended_action_counts": _recommended_action_counts(predictions),
    }
