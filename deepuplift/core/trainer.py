from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from .artifacts import (
    build_environment_snapshot,
    build_evidence_manifest,
    create_run_dir,
    dataframe_fingerprint,
    save_dataframe,
    save_json,
    save_pickle,
    save_text,
    save_torch_model,
)
from .data_profile import build_feature_profile
from .evaluator import BUSINESS_SCORE_COLS, FRONTIER_EVIDENCE_COLS, ORACLE_UPLIFT_COLS, evaluate_uplift_predictions
from .preprocess import TabularPreprocessor
from .promotion import promotion_gate
from .readiness import decision_readiness
from .registry import build_model
from .schema import UpliftConfig, UpliftRunResult


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_data(data_path: Optional[str], data_frame: Optional[pd.DataFrame]) -> pd.DataFrame:
    if data_frame is not None:
        return data_frame.copy()
    if data_path is None:
        raise ValueError("Either data_path or data_frame must be provided.")
    return pd.read_csv(data_path)


def _infer_task(outcome: pd.Series, configured_task: str) -> str:
    if configured_task != "auto":
        return configured_task
    unique_count = outcome.dropna().nunique()
    return "classification" if unique_count <= 2 else "regression"


def _normalize_binary_column(
    series: pd.Series,
    positive_value: Any = None,
    negative_value: Any = None,
    column_name: str = "column",
) -> tuple[pd.Series, Dict[str, Any]]:
    values = series.dropna().unique().tolist()
    if len(values) != 2:
        raise ValueError(f"{column_name} must have exactly two values for the current binary uplift workflow; got {values}.")

    if positive_value is not None and negative_value is not None:
        mapping = {negative_value: 0, positive_value: 1}
    elif set(values) == {0, 1}:
        mapping = {0: 0, 1: 1}
    elif set(values) == {False, True}:
        mapping = {False: 0, True: 1}
    else:
        ordered = sorted(values, key=lambda item: str(item))
        mapping = {ordered[0]: 0, ordered[1]: 1}

    mapped = series.map(mapping)
    if mapped.isna().any():
        raise ValueError(f"{column_name} contains values outside the binary mapping: {mapping}.")
    return mapped.astype("float32"), {
        "control": _plain_value(next(k for k, v in mapping.items() if v == 0)),
        "treated": _plain_value(next(k for k, v in mapping.items() if v == 1)),
    }


def _plain_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _prepare_target(series: pd.Series, task: str, column_name: str) -> tuple[pd.Series, Dict[str, Any]]:
    if task == "classification":
        mapped, mapping = _normalize_binary_column(series, column_name=column_name)
        return mapped, mapping
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        raise ValueError(f"{column_name} must be numeric for regression.")
    return values.astype("float32"), {}


def _stratify_key(df: pd.DataFrame, treatment_col: str, outcome_col: str, task: str):
    if task != "classification":
        return None
    key = df[[treatment_col, outcome_col]].astype(str).agg("_".join, axis=1)
    if key.value_counts().min() < 2:
        return None
    return key


def _tensor_to_array(value: Any) -> np.ndarray:
    mean_attr = getattr(value, "mean", None)
    if mean_attr is not None and not callable(mean_attr) and not isinstance(value, (torch.Tensor, np.ndarray, pd.Series, pd.DataFrame)):
        value = mean_attr
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().reshape(-1)
    return np.asarray(value).reshape(-1)


def _recommendations(df: pd.DataFrame, result: Dict[str, Any], model_name: str) -> list[str]:
    treatment_rate = df["treatment"].mean()
    rows = [
        f"当前训练完成模型：{model_name}。",
        f"样本 treatment 占比为 {treatment_rate:.2%}，评估时建议同时关注 QINI/AUUC 和 Top-K 人群表现。",
    ]
    qini = result["metrics"].get("qini_score")
    random_qini = result["metrics"].get("random_qini_score")
    if qini is not None and random_qini is not None:
        if qini > random_qini:
            rows.append("模型排序优于随机基线，可以继续比较更多模型并观察 Top 10% 人群。")
        else:
            rows.append("模型暂未明显超过随机基线，建议检查 treatment 随机性、特征质量或增加训练轮数。")
    top_k = result["metrics"].get("top_k") or []
    if top_k:
        best = top_k[1] if len(top_k) > 1 else top_k[0]
        uplift = best.get("observed_uplift")
        if uplift is not None:
            rows.append(f"Top {best['top_fraction']:.0%} 人群的观测 uplift 约为 {uplift:.4f}。")
    return rows


def _format_number(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _build_run_note(config_payload: Dict[str, Any], eval_result: Dict[str, Any], recommendations: list[str]) -> str:
    metrics = eval_result.get("metrics", {})
    top_k = metrics.get("top_k") or []
    oracle_top_k = metrics.get("oracle_top_k") or {}
    business_alignment = metrics.get("business_score_alignment") or {}
    full_funnel = metrics.get("full_funnel") or {}
    delayed_feedback = metrics.get("delayed_feedback") or {}
    industrial_policy = metrics.get("industrial_policy") or {}
    policy_best = metrics.get("policy_best") or {}
    sensitivity = metrics.get("sensitivity") or {}
    overlap_trim = metrics.get("overlap_trim") or {}
    bootstrap = metrics.get("bootstrap") or {}
    readiness = decision_readiness(metrics)
    distilled_segments = (eval_result.get("curves") or {}).get("distilled_segments") or []

    lines = [
        "# DeepUplift Run Note",
        "",
        "## Configuration",
        "",
        f"- Model: `{config_payload.get('model_name')}`",
        f"- Task: `{config_payload.get('task')}`",
        f"- Treatment: `{config_payload.get('treatment_col')}`",
        f"- Outcome: `{config_payload.get('outcome_col')}`",
        f"- Features: {len(config_payload.get('feature_cols') or [])}",
        f"- Test size: {_format_number(config_payload.get('test_size'))}",
        f"- Max rows: {config_payload.get('max_rows') or 'all'}",
        "",
        "## Evaluation",
        "",
        f"- QINI: {_format_number(metrics.get('qini_score'))}",
        f"- AUUC: {_format_number(metrics.get('auuc_score'))}",
        f"- Calibration MAE: {_format_number(metrics.get('calibration_mae'))}",
    ]

    lines.extend(
        [
            "",
            "## Evidence Files",
            "",
            "- `config.json`: training configuration and column schema.",
            "- `metrics.json`: evaluation metrics, policy value, calibration, sensitivity, and overlap checks.",
            "- `readiness.json`: decision-readiness score, level, blockers, and rollout recommendation.",
            "- `promotion.json`: research/shadow/pilot promotion gate and next required evidence.",
            "- `curves.json`: QINI/AUUC, Top-K, calibration, segment, and diagnostic curve data.",
            "- `nuisance_diagnostics.json`: DR/R/IPW nuisance, propensity, pseudo-outcome, residual and cross-fit diagnostics when generated.",
            "- `environment.json`: Python, package, platform, and git dirty-state snapshot.",
            "- `evidence_manifest.json`: artifact contract, data fingerprint, file sizes, and SHA-256 hashes.",
            "- `predictions.csv`: holdout rows with predicted outcomes and uplift score.",
            "- `train_history.csv`: epoch or estimator training history when available.",
            "- `run_note.md`: this human-readable decision report.",
            "",
            "## Decision Readiness",
            "",
            f"- Score: {readiness.get('score')}/100",
            f"- Level: `{readiness.get('level')}`",
            f"- Recommendation: {readiness.get('recommendation')}",
        ]
    )
    blockers = readiness.get("blockers") or []
    if blockers:
        lines.extend(["", "### Blockers"])
        lines.extend(f"- {item}" for item in blockers)
    checks = readiness.get("checks") or []
    if checks:
        lines.extend(["", "### Checks"])
        for check in checks:
            lines.append(
                f"- {check.get('check')}: `{check.get('status')}` "
                f"({check.get('points')}/{check.get('max_points')}), "
                f"value={check.get('value')}, {check.get('guidance')}"
            )

    if top_k:
        lines.extend(["", "## Top-K"])
        for row in top_k:
            lines.append(
                f"- Top {row.get('top_fraction', 0):.0%}: observed uplift "
                f"{_format_number(row.get('observed_uplift'))}, predicted uplift "
                f"{_format_number(row.get('mean_predicted_uplift'))}, rows {row.get('rows')}"
            )

    if oracle_top_k.get("rows"):
        lines.extend(["", f"## Oracle Top-K Recall (`{oracle_top_k.get('oracle_col')}`)"])
        for row in oracle_top_k["rows"]:
            lines.append(
                f"- Top {row.get('top_fraction', 0):.0%}: recall {_format_number(row.get('topk_recall'))}, "
                f"gain capture {_format_number(row.get('oracle_gain_capture'))}, rows {row.get('rows')}"
            )

    if business_alignment.get("rows"):
        lines.extend(["", f"## Business Score Top-K Alignment (`{business_alignment.get('business_col')}`)"])
        for row in business_alignment["rows"]:
            lines.append(
                f"- Top {row.get('top_fraction', 0):.0%}: overlap recall "
                f"{_format_number(row.get('topk_overlap_recall'))}, score capture "
                f"{_format_number(row.get('business_score_capture'))}, rows {row.get('rows')}"
            )

    if full_funnel.get("top_k"):
        lines.extend(["", "## Full-Funnel ECUP Metrics"])
        stages = ", ".join(full_funnel.get("stage_cols") or [])
        lines.append(f"- Stages: {stages or 'NA'}")
        for row in full_funnel["top_k"]:
            if abs(row.get("top_fraction", 0) - 0.1) < 1e-9:
                lines.append(
                    f"- Top 10% `{row.get('stage')}` uplift: {_format_number(row.get('observed_uplift'))}, "
                    f"treated/control rows {row.get('treated_rows')}/{row.get('control_rows')}"
                )

    if delayed_feedback.get("top_k"):
        lines.extend(["", "## Delayed Feedback Uplift"])
        summary = delayed_feedback.get("summary") or {}
        if summary:
            lines.append(
                f"- Label window median/max: {_format_number(summary.get('label_window_days_median'))}/"
                f"{_format_number(summary.get('label_window_days_max'))}; "
                f"censored rate: {_format_number(summary.get('censored_rate'))}; "
                f"observed delay p50/p90: {_format_number(summary.get('observed_conversion_delay_days_p50'))}/"
                f"{_format_number(summary.get('observed_conversion_delay_days_p90'))}"
            )
        for row in delayed_feedback["top_k"]:
            if abs(row.get("top_fraction", 0) - 0.1) < 1e-9:
                lines.append(
                    f"- Top 10% `{row.get('window')}` uplift: {_format_number(row.get('observed_uplift'))}, "
                    f"treated/control rows {row.get('treated_rows')}/{row.get('control_rows')}"
                )

    if industrial_policy.get("top10"):
        lines.extend(["", "## Industrial Scenario Policy Metrics"])
        for row in industrial_policy["top10"]:
            lines.append(
                f"- {row.get('scenario')} Top {row.get('top_fraction', 0):.0%}: "
                f"policy value sum {_format_number(row.get('policy_value_sum'))}, "
                f"ROI proxy {_format_number(row.get('roi_proxy'))}, "
                f"rows {row.get('rows')}"
            )

    if policy_best:
        lines.extend(
            [
                "",
                "## Policy",
                "",
                f"- Recommended threshold: Top {policy_best.get('top_fraction', 0):.0%}",
                f"- Observed net value: {_format_number(policy_best.get('observed_net_value'))}",
                f"- Predicted net value: {_format_number(policy_best.get('predicted_net_value'))}",
            ]
        )

    if bootstrap:
        qini_ci = ((bootstrap.get("metrics") or {}).get("qini_score") or {})
        auuc_ci = ((bootstrap.get("metrics") or {}).get("auuc_score") or {})
        lines.extend(
            [
                "",
                "## Bootstrap CI",
                "",
                f"- Samples: {bootstrap.get('n_success', 0)}/{bootstrap.get('n_bootstrap', 0)}",
                f"- QINI 95% CI: [{_format_number(qini_ci.get('low'))}, {_format_number(qini_ci.get('high'))}]",
                f"- AUUC 95% CI: [{_format_number(auuc_ci.get('low'))}, {_format_number(auuc_ci.get('high'))}]",
            ]
        )

    if sensitivity:
        lines.extend(
            [
                "",
                "## Sensitivity Checks",
                "",
                f"- Verdict: `{sensitivity.get('verdict')}`",
                f"- Message: {sensitivity.get('message')}",
                f"- Permutations: {sensitivity.get('n_permutations')}",
            ]
        )
        for name, payload in (sensitivity.get("tests") or {}).items():
            qini = payload.get("qini_score") or {}
            lines.append(
                f"- {name}: qini null p95 {_format_number(qini.get('null_p95'))}, "
                f"excess {_format_number(qini.get('excess_over_p95'))}, p-value {_format_number(qini.get('p_value'))}"
            )

    if overlap_trim:
        propensity = overlap_trim.get("propensity") or {}
        lines.extend(
            [
                "",
                "## Overlap Trimming",
                "",
                f"- Propensity p05/median/p95: {_format_number(propensity.get('p05'))} / {_format_number(propensity.get('median'))} / {_format_number(propensity.get('p95'))}",
                f"- Weak overlap rate: {_format_number(propensity.get('weak_overlap_rate'))}",
            ]
        )
        for row in overlap_trim.get("rows", []):
            lines.append(
                f"- Trim {row.get('trim'):.0%}: kept {row.get('kept_fraction', 0):.0%}, "
                f"QINI {_format_number(row.get('qini_score'))}, "
                f"Top10 uplift {_format_number(row.get('top10_observed_uplift'))}"
            )

    if distilled_segments:
        lines.extend(["", "## Causal Distillation Segments", ""])
        for row in distilled_segments[:5]:
            lines.append(
                f"- {row.get('rule')}: rows {row.get('rows')}, "
                f"predicted uplift {_format_number(row.get('mean_predicted_uplift'))}, "
                f"observed uplift {_format_number(row.get('observed_uplift'))}"
            )

    if recommendations:
        lines.extend(["", "## Agent Recommendations", ""])
        lines.extend(f"- {item}" for item in recommendations)

    lines.extend(
        [
            "",
            "## Caution",
            "",
            "This note summarizes statistical workflow checks. It does not by itself prove causal identification; validate experiment design, overlap, leakage and business constraints before deployment.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_saved_artifact_summary(artifacts: Dict[str, str]) -> str:
    rows = []
    for name, artifact_path in artifacts.items():
        path = Path(artifact_path)
        exists = path.exists() and path.is_file()
        rows.append(
            (
                name,
                path.name,
                "yes" if exists else "no",
                f"{path.stat().st_size / 1024:.2f}" if exists else "0.00",
            )
        )
    lines = [
        "",
        "### Saved Artifact Presence",
        "",
        "| Artifact | File | Present | Size KB |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(f"| {name} | `{file_name}` | {present} | {size_kb} |" for name, file_name, present, size_kb in rows)
    return "\n".join(lines)


def train_uplift_model(
    data_path: Optional[str] = None,
    config: UpliftConfig | Dict[str, Any] | None = None,
    data_frame: Optional[pd.DataFrame] = None,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> UpliftRunResult:
    if config is None:
        raise ValueError("config is required.")
    if isinstance(config, dict):
        config = UpliftConfig.from_dict(config)

    _set_seed(config.random_state)
    df = _load_data(data_path, data_frame)
    if config.max_rows:
        df = df.head(config.max_rows).copy()

    oracle_cols = [col for col in ORACLE_UPLIFT_COLS if col in df.columns and col not in config.feature_cols]
    business_score_cols = [
        col
        for col in BUSINESS_SCORE_COLS
        if col in df.columns and col not in config.feature_cols and col not in oracle_cols
    ]
    frontier_cols = [
        col
        for col in FRONTIER_EVIDENCE_COLS
        if col in df.columns and col not in config.feature_cols and col not in oracle_cols and col not in business_score_cols
    ]
    required_cols = list(
        dict.fromkeys(
        [config.treatment_col, config.outcome_col]
        + list(config.feature_cols)
        + oracle_cols
        + business_score_cols
        + frontier_cols
        )
    )
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")

    task = _infer_task(df[config.outcome_col], config.task)
    working = df[required_cols].dropna(subset=[config.treatment_col, config.outcome_col]).copy()
    working["treatment"], treatment_mapping = _normalize_binary_column(
        working[config.treatment_col],
        positive_value=config.treated_value,
        negative_value=config.control_value,
        column_name=config.treatment_col,
    )
    working["outcome"], outcome_mapping = _prepare_target(working[config.outcome_col], task, config.outcome_col)

    split_cols = list(dict.fromkeys(config.feature_cols + ["treatment", "outcome"] + oracle_cols + business_score_cols + frontier_cols))
    split_df = working[split_cols].copy()
    stratify = _stratify_key(split_df, "treatment", "outcome", task)
    train_df, test_df = train_test_split(
        split_df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )

    preprocessor = TabularPreprocessor(config.feature_cols, categorical_cols=config.categorical_cols)
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    y_train = train_df["outcome"]
    t_train = train_df["treatment"]
    y_test = test_df["outcome"].reset_index(drop=True)
    t_test = test_df["treatment"].reset_index(drop=True)

    model, loss_f = build_model(
        model_name=config.model_name,
        input_dim=X_train.shape[1],
        task=task,
        model_params=config.model_params,
    )

    history = model.fit(
        X_train,
        y_train,
        t_train,
        valid_perc=config.valid_perc,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        loss_f=loss_f,
        tensorboard=config.tensorboard,
        callback=callback,
    )

    t_pred, y_preds, *_ = model.predict(X_test, t_test)
    y0_pred = _tensor_to_array(y_preds[0])
    y1_pred = _tensor_to_array(y_preds[1])

    prediction_df = pd.DataFrame(
        {
            "treatment": t_test,
            "outcome": y_test,
            "y0_pred": y0_pred,
            "y1_pred": y1_pred,
            "uplift_score": y1_pred - y0_pred,
        }
    )
    feature_preview = test_df[config.feature_cols].reset_index(drop=True)
    prediction_df = pd.concat([prediction_df, feature_preview], axis=1)
    if oracle_cols:
        prediction_df = pd.concat([prediction_df, test_df[oracle_cols].reset_index(drop=True)], axis=1)
    if business_score_cols:
        prediction_df = pd.concat([prediction_df, test_df[business_score_cols].reset_index(drop=True)], axis=1)
    if frontier_cols:
        prediction_df = pd.concat([prediction_df, test_df[frontier_cols].reset_index(drop=True)], axis=1)

    eval_result = evaluate_uplift_predictions(
        prediction_df,
        outcome_col="outcome",
        treatment_col="treatment",
        uplift_col="uplift_score",
        segment_cols=config.feature_cols,
        bootstrap_samples=config.bootstrap_samples,
        sensitivity_samples=config.sensitivity_samples,
        random_state=config.random_state,
        policy_contact_cost=config.policy_contact_cost,
        policy_conversion_value=config.policy_conversion_value,
    )

    run_id, run_dir = create_run_dir(config.artifacts_dir, config.run_name)
    config_payload = config.to_dict()
    config_payload.update(
        {
            "task": task,
            "encoded_feature_cols": preprocessor.output_feature_cols,
            "treatment_mapping": treatment_mapping,
            "outcome_mapping": outcome_mapping,
        }
    )
    model_metadata = {
        "model_name": config.model_name,
        "task": task,
        "input_dim": X_train.shape[1],
        "feature_cols": config.feature_cols,
        "encoded_feature_cols": preprocessor.output_feature_cols,
        "model_params": config.model_params,
    }
    if hasattr(model, "state_dict"):
        model_path = save_torch_model(run_dir / "model.pt", model, model_metadata)
    else:
        model_path = save_pickle(run_dir / "model.pkl", {"model": model, "metadata": model_metadata})

    recommendations = _recommendations(prediction_df, eval_result, config.model_name)
    readiness = decision_readiness(eval_result["metrics"])
    promotion = promotion_gate(eval_result["metrics"], readiness)
    run_note = _build_run_note(config_payload, eval_result, recommendations)
    data_fingerprint = dataframe_fingerprint(working, columns=required_cols)
    environment_path = save_json(run_dir / "environment.json", build_environment_snapshot(Path.cwd()))
    artifacts = {
        "config": save_json(run_dir / "config.json", config_payload),
        "metrics": save_json(run_dir / "metrics.json", eval_result["metrics"]),
        "readiness": save_json(run_dir / "readiness.json", readiness),
        "promotion": save_json(run_dir / "promotion.json", promotion),
        "curves": save_json(run_dir / "curves.json", eval_result["curves"]),
        "feature_profile": save_json(run_dir / "feature_profile.json", build_feature_profile(working, config.feature_cols)),
        "environment": environment_path,
        "predictions": save_dataframe(run_dir / "predictions.csv", prediction_df),
        "history": save_dataframe(run_dir / "train_history.csv", pd.DataFrame(history)),
        "preprocessor": save_pickle(run_dir / "preprocessor.pkl", preprocessor),
        "model": model_path,
    }
    nuisance_diagnostics = getattr(model, "nuisance_diagnostics_", None)
    if nuisance_diagnostics:
        artifacts["nuisance_diagnostics"] = save_json(run_dir / "nuisance_diagnostics.json", nuisance_diagnostics)
    artifacts["run_note"] = save_text(run_dir / "run_note.md", run_note)
    artifacts["evidence_manifest"] = save_json(
        run_dir / "evidence_manifest.json",
        build_evidence_manifest(
            run_id=run_id,
            run_dir=run_dir,
            artifacts=artifacts,
            config=config_payload,
            model_metadata=model_metadata,
            data_fingerprint=data_fingerprint,
            metrics=eval_result["metrics"],
            readiness=readiness,
            environment_path=environment_path,
        ),
    )
    run_note = run_note + _build_saved_artifact_summary(artifacts)
    artifacts["run_note"] = save_text(run_dir / "run_note.md", run_note)

    return UpliftRunResult(
        run_id=run_id,
        run_dir=str(run_dir),
        config=config_payload,
        train_history=history,
        eval_metrics=eval_result["metrics"],
        curves=eval_result["curves"],
        artifacts=artifacts,
        preview=prediction_df.head(50).to_dict(orient="records"),
        recommendations=recommendations,
    )
