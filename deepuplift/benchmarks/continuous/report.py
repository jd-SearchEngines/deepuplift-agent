from __future__ import annotations

from typing import Any


def continuous_benchmark_report(result: dict[str, Any]) -> str:
    lines = [
        "# Continuous Treatment Benchmark",
        "",
        f"Status: **{result.get('status', 'UNKNOWN')}**",
        "",
        f"Dataset: `{result.get('dataset', {}).get('name', 'unknown')}`; rows={result.get('dataset', {}).get('sample_size')}; seed={result.get('dataset', {}).get('split_seed')}; treatment range={result.get('dataset', {}).get('treatment_range')}; ground truth={result.get('dataset', {}).get('ground_truth_available')}",
        "",
        "All errors and regrets are computed on the held-out test rows and a shared dose grid. Economic decisions maximize estimated effect times outcome value minus dose cost.",
        "",
        "| Model | Status | Maturity | MISE | Max-effect dose | Max-net-value dose | Optimal dose error | Economic regret | Train s | Curve s | Decision s | Peak MB |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in result.get("models", []):
        metrics = model.get("metrics", {})
        values = [
            model.get("model"),
            model.get("status"),
            model.get("maturity", "NOT_AVAILABLE"),
            metrics.get("mise"),
            metrics.get("predicted_max_effect_dose_mean"),
            metrics.get("predicted_max_net_value_dose_mean"),
            metrics.get("optimal_dose_error"),
            metrics.get("economic_policy_regret"),
            model.get("train_seconds"),
            model.get("predict_curve_seconds"),
            model.get("decision_seconds"),
            model.get("peak_memory_mb"),
        ]
        formatted = ["—" if value is None else (f"{value:.5f}" if isinstance(value, (int, float)) else str(value)) for value in values]
        lines.append("| " + " | ".join(formatted) + " |")
    lines += ["", "## Decision distinction", "", "`predicted_max_effect_dose_mean` and `predicted_max_net_value_dose_mean` are reported separately. The policy uses the latter objective; maximum predicted effect is a model diagnostic.", ""]
    for model in result.get("models", []):
        if model.get("status") == "BLOCKED":
            lines.append(f"- `{model.get('model')}`: BLOCKED — {model.get('error', 'unavailable')}")
        elif model.get("prediction_metadata", {}).get("giks"):
            giks = model["prediction_metadata"]["giks"]
            lines.append(f"- `{model['model']}` GIKS: base={giks.get('base_model')}; GI={giks.get('gradient_interpolation_accepted')}/{giks.get('gradient_interpolation_attempted')}; GP smoothing={giks.get('kernel_smoothing_accepted')}/{giks.get('kernel_smoothing_attempted')}; accepted pseudo-labels={giks.get('accepted_pseudo_label_count')}/{giks.get('pseudo_label_count')}.")
    lines += ["", "## Treatment support", "", "Observed and local-support summaries are empirical coverage diagnostics. Evaluation curves are scored only over the intersection of the truth grid and training dose support; unsupported causal baselines leave causal and economic regret metrics unavailable.", ""]
    for model in result.get("models", []):
        support = model.get("support_metrics", {})
        if support:
            lines.append(
                f"- `{model.get('model')}`: train={support.get('observed_dose_range')}; "
                f"test={support.get('test_dose_range')}; out-of-range test fraction={support.get('extrapolation_fraction')}; "
                f"mean local support score={support.get('mean_local_support_score')}; "
                f"GIKS pseudo extrapolation fraction={support.get('giks_pseudo_extrapolation_fraction')}."
            )
    lines.append("")
    return "\n".join(lines)


__all__ = ["continuous_benchmark_report"]
