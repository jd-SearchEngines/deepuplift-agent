from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .preprocess import TabularPreprocessor


def _clean_float(value: Any):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return number


def _binary_map(series: pd.Series) -> pd.Series | None:
    values = series.dropna().unique().tolist()
    if len(values) != 2:
        return None
    if set(values) == {0, 1}:
        mapping = {0: 0, 1: 1}
    elif set(values) == {False, True}:
        mapping = {False: 0, True: 1}
    else:
        ordered = sorted(values, key=lambda item: str(item))
        mapping = {ordered[0]: 0, ordered[1]: 1}
    return series.map(mapping).astype("float32")


def _design_type(df: pd.DataFrame, treatment_col: str, outcome_col: str) -> Dict[str, Any]:
    treatment_unique = df[treatment_col].dropna().nunique()
    outcome_unique = df[outcome_col].dropna().nunique()
    if treatment_unique == 2:
        treatment_type = "binary"
    elif is_numeric_dtype(df[treatment_col]) and treatment_unique > 10:
        treatment_type = "continuous"
    else:
        treatment_type = "multi-valued"

    if outcome_unique <= 2:
        outcome_type = "binary"
    elif is_numeric_dtype(df[outcome_col]):
        outcome_type = "continuous"
    else:
        outcome_type = "categorical"

    return {
        "treatment_type": treatment_type,
        "outcome_type": outcome_type,
        "treatment_unique": int(treatment_unique),
        "outcome_unique": int(outcome_unique),
    }


def _standardized_mean_difference(df: pd.DataFrame, col: str, t_binary: pd.Series):
    values = pd.to_numeric(df[col], errors="coerce")
    treated = values[t_binary == 1].dropna()
    control = values[t_binary == 0].dropna()
    if len(treated) == 0 or len(control) == 0:
        return None
    pooled = np.sqrt((treated.var() + control.var()) / 2)
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return _clean_float((treated.mean() - control.mean()) / pooled)


def _categorical_balance(df: pd.DataFrame, col: str, t_binary: pd.Series):
    values = df[col].astype("object").where(df[col].notna(), "__missing__").astype(str)
    treated = values[t_binary == 1].value_counts(normalize=True)
    control = values[t_binary == 0].value_counts(normalize=True)
    levels = set(treated.index) | set(control.index)
    if not levels:
        return None
    max_diff = max(abs(treated.get(level, 0.0) - control.get(level, 0.0)) for level in levels)
    return _clean_float(max_diff)


def diagnose_uplift_data(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    feature_cols: List[str],
) -> Dict[str, Any]:
    required_cols = [treatment_col, outcome_col] + list(feature_cols)
    working = df[required_cols].dropna(subset=[treatment_col, outcome_col]).copy()
    design = _design_type(working, treatment_col, outcome_col)
    warnings = []

    treatment_distribution = (
        working[treatment_col].value_counts(dropna=False).rename_axis("value").reset_index(name="rows").to_dict(orient="records")
    )
    outcome_by_treatment = (
        working.groupby(treatment_col, observed=True)[outcome_col]
        .agg(["mean", "count"])
        .reset_index()
        .to_dict(orient="records")
    )
    missingness = (
        working[required_cols]
        .isna()
        .mean()
        .sort_values(ascending=False)
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "column"})
        .to_dict(orient="records")
    )

    balance_rows = []
    overlap = {}
    selection_bias = {
        "risk_level": "not_applicable",
        "signals": [],
        "recommended_models": [],
        "interview_note": "Selection-bias diagnostics require binary treatment and usable features.",
    }
    t_binary = _binary_map(working[treatment_col])
    if design["treatment_type"] != "binary" or t_binary is None:
        warnings.append("当前核心训练流程主要支持 binary treatment；multi-valued/continuous treatment 需要使用后续扩展模块。")
    else:
        ratio = float(t_binary.mean())
        bias_signals = []
        if ratio < 0.2 or ratio > 0.8:
            warnings.append("Treatment/control 比例明显不均衡，建议优先比较 X-Learner、DR-Learner、CFRNet，并检查 overlap。")
            bias_signals.append({"signal": "treatment_ratio_imbalance", "value": _clean_float(ratio), "severity": "medium"})

        for col in feature_cols:
            if col not in working.columns:
                continue
            if is_numeric_dtype(working[col]):
                score = _standardized_mean_difference(working, col, t_binary)
                metric = "smd"
            else:
                score = _categorical_balance(working, col, t_binary)
                metric = "max_level_rate_diff"
            if score is not None:
                balance_rows.append({"feature": col, "metric": metric, "imbalance": score, "abs_imbalance": abs(score)})

        if feature_cols:
            try:
                prep = TabularPreprocessor(feature_cols)
                x = prep.fit_transform(working)
                model = LogisticRegression(max_iter=1000)
                model.fit(x, t_binary)
                propensity = model.predict_proba(x)[:, 1]
                auc = roc_auc_score(t_binary, propensity)
                weak_overlap = ((propensity < 0.05) | (propensity > 0.95)).mean()
                overlap = {
                    "treatment_auc": _clean_float(auc),
                    "weak_overlap_rate": _clean_float(weak_overlap),
                    "min": _clean_float(np.min(propensity)),
                    "p05": _clean_float(np.quantile(propensity, 0.05)),
                    "median": _clean_float(np.median(propensity)),
                    "p95": _clean_float(np.quantile(propensity, 0.95)),
                    "max": _clean_float(np.max(propensity)),
                }
                if auc > 0.8:
                    warnings.append("Propensity AUC 较高，说明 treatment assignment 可被特征强预测；观测数据下需要更谨慎。")
                    bias_signals.append({"signal": "high_propensity_auc", "value": _clean_float(auc), "severity": "high" if auc > 0.85 else "medium"})
                if weak_overlap > 0.05:
                    warnings.append("存在 weak overlap 样本，Top-K 策略和 uplift 估计可能不稳定。")
                    bias_signals.append({"signal": "weak_overlap", "value": _clean_float(weak_overlap), "severity": "high"})
            except Exception as exc:
                overlap = {"error": str(exc)}

    balance = sorted(balance_rows, key=lambda row: row["abs_imbalance"], reverse=True)
    if balance and balance[0]["abs_imbalance"] > 0.25:
        warnings.append("部分特征在 treatment/control 间差异较大，建议查看 Feature Balance 并使用平衡/DR 类模型。")
    if design["treatment_type"] == "binary" and t_binary is not None:
        high_balance = [row for row in balance if row.get("abs_imbalance") is not None and row["abs_imbalance"] > 0.25]
        medium_balance = [row for row in balance if row.get("abs_imbalance") is not None and row["abs_imbalance"] > 0.10]
        if high_balance:
            bias_signals.append(
                {
                    "signal": "feature_balance_imbalance",
                    "value": len(high_balance),
                    "severity": "high" if len(high_balance) >= 3 else "medium",
                    "top_features": [row["feature"] for row in high_balance[:5]],
                }
            )
        elif medium_balance:
            bias_signals.append(
                {
                    "signal": "mild_feature_balance_imbalance",
                    "value": len(medium_balance),
                    "severity": "medium",
                    "top_features": [row["feature"] for row in medium_balance[:5]],
                }
            )
        severities = [row.get("severity") for row in bias_signals]
        if "high" in severities:
            risk_level = "high"
        elif "medium" in severities:
            risk_level = "medium"
        else:
            risk_level = "low"
        selection_bias = {
            "risk_level": risk_level,
            "signals": bias_signals,
            "recommended_models": [
                "DRLearnerLightGBM",
                "RLearnerLightGBM",
                "DomainAdaptationLightGBM",
                "CFRNet",
                "EconMLCausalForestDML",
            ]
            if risk_level in {"medium", "high"}
            else ["TLearnerLightGBM", "SLearnerLightGBM", "DragonNet"],
            "interview_note": (
                "This is an SSB/selection-bias proxy built from treatment imbalance, feature balance, "
                "propensity separability, and weak-overlap signals. It does not prove ignorability, "
                "but it tells the Agent when to prefer DR/R/domain-adaptation/balancing models and "
                "when to avoid overconfident Top-K deployment."
            ),
        }
        if risk_level == "high":
            warnings.append("Selection bias / SSB 风险较高：建议优先使用 DR/R/Domain Adaptation/CFR 类模型，并通过 bootstrap、sensitivity 和在线实验验证。")

    return {
        "design": design,
        "treatment_distribution": treatment_distribution,
        "outcome_by_treatment": outcome_by_treatment,
        "missingness": missingness,
        "feature_balance": balance,
        "overlap": overlap,
        "selection_bias": selection_bias,
        "warnings": warnings,
    }
