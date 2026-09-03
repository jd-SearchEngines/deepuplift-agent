from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core import UpliftConfig, train_uplift_model
from deepuplift.core.artifacts import build_environment_snapshot, file_sha256, save_json
from deepuplift.core.glossary import glossary_rows
from deepuplift.core.registry import MODEL_REGISTRY, missing_dependencies


DEFAULT_DATASETS = [
    "ihdp_npci_1",
    "acic_style_synthetic_known_cate_1k6",
    "synthetic_known_cate_shift_1k8",
]
DEFAULT_MODELS = ["TLearnerGBM", "DRLearnerGBM", "CFRNet", "DragonNet", "EFIN", "DESCN"]
TRUE_EFFECT_COLS = ["true_uplift", "true_effect", "ite", "cate", "tau", "treatment_effect", "mu1_minus_mu0"]
PRESETS = {
    "smoke": {"seeds": [20260520], "rows": 220, "known_effect_bootstrap_samples": 8, "tier": "smoke"},
    "paper-lite": {"seeds": [20260520], "rows": 420, "known_effect_bootstrap_samples": 20, "tier": "paper_level_lite"},
    "benchmark-v3": {
        "seeds": [20260520, 20260521, 20260522],
        "rows": 420,
        "known_effect_bootstrap_samples": 30,
        "tier": "paper_level_multi_seed",
    },
}


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["id"]): item for item in payload.get("datasets", []) if isinstance(item, dict)}


def _clean_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _model_params(model_name: str) -> dict[str, Any]:
    if model_name == "CFRNet":
        return {"share_dim": 8, "share_hidden_dims": [24], "base_hidden_dims": [16], "alpha": 0.2, "ipm_mode": "mmd_rbf"}
    if model_name == "DragonNet":
        return {"share_dim": 8, "share_hidden_dims": [24], "base_hidden_dims": [16], "alpha": 0.15, "beta": 0.05, "tarreg": True}
    if model_name == "EFIN":
        return {"hc_dim": 16, "hu_dim": 8, "is_self": False}
    if model_name in {"DESCN", "ESX"}:
        return {"share_dim": 8, "base_dim": 8, "do_rate": 0.0}
    return {}


def _unsupported_task_reason(model_name: str, task: str) -> str | None:
    spec = MODEL_REGISTRY.get(model_name)
    if spec is None:
        return f"unknown model: {model_name}"
    if task not in spec.supported_tasks:
        return f"{model_name} supports {spec.supported_tasks}; dataset task is {task}"
    return None


def _true_effect_col(df: pd.DataFrame) -> str | None:
    return next((col for col in TRUE_EFFECT_COLS if col in df.columns), None)


def _known_effect_metrics(predictions: pd.DataFrame, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    effect_col = _true_effect_col(predictions)
    if effect_col is None or "uplift_score" not in predictions.columns:
        return {"available": False, "reason": "missing true effect or uplift_score"}
    work = predictions[["uplift_score", effect_col] + [col for col in ["oracle_policy_value"] if col in predictions.columns]].replace(
        [np.inf, -np.inf], np.nan
    )
    work = work.dropna(subset=["uplift_score", effect_col]).copy()
    if work.empty:
        return {"available": False, "reason": "no valid rows after dropping missing effects"}
    pred = work["uplift_score"].astype(float).to_numpy()
    true = work[effect_col].astype(float).to_numpy()
    diff = pred - true
    pehe = float(np.sqrt(np.mean(np.square(diff))))
    ate_error = float(abs(np.mean(pred) - np.mean(true)))
    corr = float(np.corrcoef(pred, true)[0, 1]) if len(work) > 2 and np.std(pred) > 1e-12 and np.std(true) > 1e-12 else None
    policy_top10_value = None
    if "oracle_policy_value" in work.columns:
        top_n = max(1, int(math.ceil(len(work) * 0.1)))
        policy_top10_value = float(work.sort_values("uplift_score", ascending=False).head(top_n)["oracle_policy_value"].sum())

    boot = {"samples": int(bootstrap_samples), "metrics": {}}
    if bootstrap_samples > 0:
        rng = np.random.default_rng(seed)
        buckets = {"pehe": [], "ate_error": [], "effect_corr": [], "policy_top10_oracle_value": []}
        values = work.reset_index(drop=True)
        for _ in range(bootstrap_samples):
            idx = rng.integers(0, len(values), size=len(values))
            sample = values.iloc[idx]
            sample_pred = sample["uplift_score"].astype(float).to_numpy()
            sample_true = sample[effect_col].astype(float).to_numpy()
            buckets["pehe"].append(float(np.sqrt(np.mean(np.square(sample_pred - sample_true)))))
            buckets["ate_error"].append(float(abs(np.mean(sample_pred) - np.mean(sample_true))))
            if np.std(sample_pred) > 1e-12 and np.std(sample_true) > 1e-12:
                buckets["effect_corr"].append(float(np.corrcoef(sample_pred, sample_true)[0, 1]))
            if "oracle_policy_value" in sample.columns:
                top_n = max(1, int(math.ceil(len(sample) * 0.1)))
                buckets["policy_top10_oracle_value"].append(
                    float(sample.sort_values("uplift_score", ascending=False).head(top_n)["oracle_policy_value"].sum())
                )
        for metric, values_list in buckets.items():
            clean = np.asarray([value for value in values_list if value is not None and np.isfinite(value)], dtype=float)
            if clean.size:
                boot["metrics"][metric] = {
                    "mean": float(np.mean(clean)),
                    "low": float(np.quantile(clean, 0.025)),
                    "high": float(np.quantile(clean, 0.975)),
                }
    return {
        "available": True,
        "effect_col": effect_col,
        "rows": int(len(work)),
        "pehe": pehe,
        "ate_error": ate_error,
        "effect_corr": corr,
        "policy_top10_oracle_value": policy_top10_value,
        "bootstrap": boot,
    }


def _metric_at_fraction(rows: list[dict[str, Any]] | dict[str, Any], fraction: float) -> dict[str, Any]:
    if isinstance(rows, dict):
        rows = rows.get("rows") or []
    for row in rows or []:
        try:
            if abs(float(row.get("top_fraction") or 0.0) - fraction) < 1e-9:
                return row
        except Exception:
            continue
    return {}


def _run_one(
    dataset: dict[str, Any],
    model_name: str,
    *,
    rows: int,
    seed: int,
    artifacts_dir: Path,
    known_effect_bootstrap: int,
) -> dict[str, Any]:
    task = dataset.get("task", "classification")
    unsupported = _unsupported_task_reason(model_name, task)
    if unsupported:
        return {
            "dataset_id": dataset["id"],
            "dataset_name": dataset.get("name"),
            "model": model_name,
            "seed": seed,
            "status": "skipped_task",
            "skip_reason": unsupported,
        }
    missing = missing_dependencies(model_name)
    if missing:
        return {
            "dataset_id": dataset["id"],
            "dataset_name": dataset.get("name"),
            "model": model_name,
            "seed": seed,
            "status": "skipped_dependency",
            "missing_dependencies": "; ".join(missing),
        }

    data_path = ROOT / dataset["path"]
    frame = pd.read_csv(data_path).head(rows).copy()
    quick = dataset.get("quick_train") or {}
    result = train_uplift_model(
        data_frame=frame,
        config=UpliftConfig(
            treatment_col=dataset["treatment_col"],
            outcome_col=dataset["outcome_col"],
            feature_cols=dataset["feature_cols"],
            model_name=model_name,
            task=task,
            test_size=0.30,
            valid_perc=0.2,
            epochs=int(quick.get("epochs", 2)),
            batch_size=int(quick.get("batch_size", 128)),
            learning_rate=float(quick.get("learning_rate", 0.001)),
            random_state=seed,
            artifacts_dir=str(artifacts_dir),
            run_name=f"paper-benchmark-{dataset['id']}-{model_name.lower()}-seed{seed}",
            bootstrap_samples=0,
            sensitivity_samples=0,
            policy_contact_cost=0.05,
            policy_conversion_value=10.0,
            model_params=_model_params(model_name),
        ),
    )
    predictions = pd.read_csv(result.artifacts["predictions"])
    known_metrics = _known_effect_metrics(predictions, known_effect_bootstrap, seed=seed + 1009)
    metrics = result.eval_metrics
    top10 = _metric_at_fraction(metrics.get("top_k") or [], 0.1)
    oracle_top10 = _metric_at_fraction(metrics.get("oracle_top_k") or {}, 0.1)
    policy_best = metrics.get("policy_best") or {}
    qini_boot = (((metrics.get("bootstrap") or {}).get("metrics") or {}).get("qini_score") or {})
    return {
        "dataset_id": dataset["id"],
        "dataset_name": dataset.get("name"),
        "dataset_kind": dataset.get("kind"),
        "model": model_name,
        "seed": seed,
        "status": "ok",
        "run_id": result.run_id,
        "run_dir": result.run_dir,
        "evidence_manifest": result.artifacts.get("evidence_manifest"),
        "predictions": result.artifacts.get("predictions"),
        "metrics_path": result.artifacts.get("metrics"),
        "history": result.artifacts.get("history"),
        "qini": _clean_float(metrics.get("qini_score")),
        "qini_ci_low": _clean_float(qini_boot.get("low")),
        "qini_ci_high": _clean_float(qini_boot.get("high")),
        "auuc": _clean_float(metrics.get("auuc_score")),
        "top10_observed_uplift": _clean_float(top10.get("observed_uplift")),
        "oracle_top10_recall": _clean_float(oracle_top10.get("topk_recall")),
        "oracle_top10_gain_capture": _clean_float(oracle_top10.get("oracle_gain_capture")),
        "policy_top_fraction": _clean_float(policy_best.get("top_fraction")),
        "policy_observed_net_value": _clean_float(policy_best.get("observed_net_value")),
        "policy_predicted_net_value": _clean_float(policy_best.get("predicted_net_value")),
        "pehe": _clean_float(known_metrics.get("pehe")),
        "ate_error": _clean_float(known_metrics.get("ate_error")),
        "effect_corr": _clean_float(known_metrics.get("effect_corr")),
        "policy_top10_oracle_value": _clean_float(known_metrics.get("policy_top10_oracle_value")),
        "known_effect": known_metrics,
        "interview_line": (
            f"{model_name} on {dataset['id']}: PEHE={_fmt(known_metrics.get('pehe'))}, "
            f"ATE error={_fmt(known_metrics.get('ate_error'))}, QINI={_fmt(metrics.get('qini_score'))}; "
            "this row links source code, benchmark metric and evidence manifest."
        ),
    }


def _summary(values: pd.Series, lower_is_better: bool = False) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"mean": None, "std": None, "low": None, "high": None, "best": None}
    mean = float(numeric.mean())
    std = float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0
    half_width = 1.96 * std / (len(numeric) ** 0.5) if len(numeric) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "low": mean - half_width,
        "high": mean + half_width,
        "best": float(numeric.min() if lower_is_better else numeric.max()),
    }


def _leaderboard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok = [row for row in rows if row.get("status") == "ok"]
    if not ok:
        return []
    frame = pd.DataFrame(ok)
    metrics = {
        "pehe": True,
        "ate_error": True,
        "effect_corr": False,
        "qini": False,
        "auuc": False,
        "policy_observed_net_value": False,
        "policy_top10_oracle_value": False,
        "oracle_top10_recall": False,
    }
    output = []
    for (dataset_id, model), part in frame.groupby(["dataset_id", "model"]):
        row: dict[str, Any] = {
            "dataset_id": dataset_id,
            "dataset_name": part["dataset_name"].dropna().iloc[0],
            "dataset_kind": part["dataset_kind"].dropna().iloc[0],
            "model": model,
            "ok_runs": int(len(part)),
            "seeds": sorted(int(seed) for seed in pd.to_numeric(part["seed"], errors="coerce").dropna().unique()),
            "evidence_manifest_count": int(part["evidence_manifest"].dropna().nunique()),
        }
        for metric, lower in metrics.items():
            stats = _summary(part[metric], lower_is_better=lower) if metric in part else _summary(pd.Series(dtype=float))
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
            row[f"{metric}_ci_low"] = stats["low"]
            row[f"{metric}_ci_high"] = stats["high"]
            row[f"{metric}_best"] = stats["best"]
        output.append(row)
    return sorted(
        output,
        key=lambda row: (
            row.get("pehe_mean") is not None,
            -(row.get("pehe_mean") or 1e18),
            -(row.get("ate_error_mean") or 1e18),
            row.get("qini_mean") if row.get("qini_mean") is not None else -1e18,
        ),
        reverse=True,
    )


def _best_by_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    leaderboard = _leaderboard(rows)
    output = []
    for dataset_id in sorted({row["dataset_id"] for row in leaderboard}):
        part = [row for row in leaderboard if row["dataset_id"] == dataset_id and row.get("pehe_mean") is not None]
        if part:
            output.append(sorted(part, key=lambda row: (row.get("pehe_mean") or 1e18, row.get("ate_error_mean") or 1e18))[0])
    return output


def _rankings_by_metric(leaderboard: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    ranking_specs = {
        "pehe": ("pehe_mean", True),
        "ate_error": ("ate_error_mean", True),
        "qini": ("qini_mean", False),
        "auuc": ("auuc_mean", False),
        "policy_top10_oracle_value": ("policy_top10_oracle_value_mean", False),
        "oracle_top10_recall": ("oracle_top10_recall_mean", False),
    }
    rankings: dict[str, list[dict[str, Any]]] = {}
    for name, (field, lower_is_better) in ranking_specs.items():
        candidates = [row for row in leaderboard if row.get(field) is not None]
        ordered = sorted(candidates, key=lambda row: float(row[field]), reverse=not lower_is_better)
        rankings[name] = [
            {
                "rank": index + 1,
                "dataset_id": row.get("dataset_id"),
                "model": row.get("model"),
                "metric": field,
                "value": row.get(field),
                "ok_runs": row.get("ok_runs"),
                "ci_low": row.get(field.replace("_mean", "_ci_low")),
                "ci_high": row.get(field.replace("_mean", "_ci_high")),
            }
            for index, row in enumerate(ordered)
        ]
    return rankings


def _stability_summary(leaderboard: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    rows = []
    for row in leaderboard:
        pehe_mean = _clean_float(row.get("pehe_mean"))
        pehe_low = _clean_float(row.get("pehe_ci_low"))
        pehe_high = _clean_float(row.get("pehe_ci_high"))
        qini_mean = _clean_float(row.get("qini_mean"))
        qini_low = _clean_float(row.get("qini_ci_low"))
        qini_high = _clean_float(row.get("qini_ci_high"))
        if len(seeds) <= 1 or int(row.get("ok_runs") or 0) <= 1:
            verdict = "single_seed_smoke"
            reason = "Only one seed is available; use this row for regression smoke, not final paper claims."
        else:
            pehe_width = None if pehe_low is None or pehe_high is None else pehe_high - pehe_low
            qini_crosses_zero = qini_low is not None and qini_high is not None and qini_low <= 0 <= qini_high
            if pehe_mean is not None and pehe_width is not None and pehe_width <= max(abs(pehe_mean) * 0.5, 1e-6) and not qini_crosses_zero:
                verdict = "stable_candidate"
                reason = "PEHE confidence interval is reasonably tight and QINI interval does not cross zero."
            else:
                verdict = "needs_review"
                reason = "Metric confidence interval is wide or QINI crosses zero; inspect seeds and failure attribution."
        rows.append(
            {
                "dataset_id": row.get("dataset_id"),
                "model": row.get("model"),
                "ok_runs": row.get("ok_runs"),
                "pehe_mean": pehe_mean,
                "pehe_ci_width": None if pehe_low is None or pehe_high is None else pehe_high - pehe_low,
                "qini_mean": qini_mean,
                "qini_ci_crosses_zero": qini_low is not None and qini_high is not None and qini_low <= 0 <= qini_high,
                "verdict": verdict,
                "reason": reason,
            }
        )
    return rows


def _failure_attribution(rows: list[dict[str, Any]], failures: list[str]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    skipped: dict[str, int] = {}
    errors: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        reason = row.get("skip_reason") or row.get("missing_dependencies")
        if reason:
            key = str(reason).splitlines()[0][:180]
            skipped[key] = skipped.get(key, 0) + 1
        if row.get("error"):
            key = str(row["error"]).splitlines()[0][:180]
            errors[key] = errors.get(key, 0) + 1
    return {"status_counts": status_counts, "skipped": skipped, "errors": errors, "failures": failures}


def _previous_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_diff(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    if not previous:
        return {"status": "no_previous", "rows": [], "summary": "No previous paper benchmark suite found."}
    prev = {(row.get("dataset_id"), row.get("model")): row for row in previous.get("leaderboard") or []}
    curr = {(row.get("dataset_id"), row.get("model")): row for row in current.get("leaderboard") or []}
    rows = []
    for key in sorted(set(prev) | set(curr), key=lambda item: (str(item[0]), str(item[1]))):
        if key not in prev:
            rows.append({"dataset_id": key[0], "model": key[1], "change_type": "added"})
            continue
        if key not in curr:
            rows.append({"dataset_id": key[0], "model": key[1], "change_type": "removed"})
            continue
        diff = {"dataset_id": key[0], "model": key[1], "change_type": "changed"}
        changed = False
        for metric in ["pehe_mean", "ate_error_mean", "qini_mean", "policy_top10_oracle_value_mean"]:
            prev_value = prev[key].get(metric)
            curr_value = curr[key].get(metric)
            delta = None if prev_value is None or curr_value is None else float(curr_value) - float(prev_value)
            diff[f"previous_{metric}"] = prev_value
            diff[f"current_{metric}"] = curr_value
            diff[f"delta_{metric}"] = delta
            changed = changed or (delta is not None and abs(delta) > 1e-9)
        if changed:
            rows.append(diff)
    return {"status": "ok", "rows": rows, "summary": f"{len(rows)} paper benchmark leaderboard rows changed."}


def _suite_manifest(output_paths: list[Path], run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    files = []
    for path in output_paths:
        if path.exists() and path.is_file():
            files.append({"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    run_manifest_paths = [Path(str(row.get("evidence_manifest"))) for row in run_rows if row.get("status") == "ok" and row.get("evidence_manifest")]
    return {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "artifact": "paper_level_benchmark",
        "files": files,
        "run_evidence_manifests": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in run_manifest_paths
            if path.exists()
        ],
        "environment": build_environment_snapshot(ROOT),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return "NA"
        return f"{number:.6g}"
    except (TypeError, ValueError):
        return str(value)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# DeepUplift Paper-Level Benchmark Evidence",
        "",
        f"Generated at: `{payload['generated_at']}`",
        "",
        "## Summary",
        "",
        f"- Status: `{payload['status']}`",
        f"- Benchmark tier: `{payload.get('benchmark_tier', 'NA')}`",
        f"- Datasets: {', '.join(payload['datasets'])}",
        f"- Models: {', '.join(payload['models'])}",
        f"- Seeds: {', '.join(str(seed) for seed in payload['seeds'])}",
        f"- Successful runs: {sum(1 for row in payload['runs'] if row.get('status') == 'ok')}",
        f"- Skipped/failed rows: {sum(1 for row in payload['runs'] if row.get('status') != 'ok')}",
        "",
        "## How To Read The Terms",
        "",
        "| Term | Plain Meaning | Formula | Pitfall |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload.get("metric_glossary") or []:
        lines.append(
            f"| {row.get('term')} | {row.get('short_cn')} | `{row.get('formula')}` | {row.get('common_pitfall')} |"
        )
    lines.extend(
        [
        "",
        "## Best By Dataset",
        "",
        "| Dataset | Model | PEHE | ATE Error | QINI | Oracle Top10 Recall | Evidence Manifest |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload.get("best_by_dataset") or []:
        lines.append(
            f"| {row.get('dataset_id')} | {row.get('model')} | {_fmt(row.get('pehe_mean'))} | "
            f"{_fmt(row.get('ate_error_mean'))} | {_fmt(row.get('qini_mean'))} | "
            f"{_fmt(row.get('oracle_top10_recall_mean'))} | {row.get('evidence_manifest_count')} manifests |"
        )
    lines.extend(
        [
            "",
            "## Leaderboard",
            "",
            "| Dataset | Model | OK Runs | PEHE Mean | PEHE 95% CI | ATE Error | QINI | Policy Oracle Top10 |",
            "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in payload.get("leaderboard") or []:
        lines.append(
            f"| {row.get('dataset_id')} | {row.get('model')} | {row.get('ok_runs')} | "
            f"{_fmt(row.get('pehe_mean'))} | [{_fmt(row.get('pehe_ci_low'))}, {_fmt(row.get('pehe_ci_high'))}] | "
            f"{_fmt(row.get('ate_error_mean'))} | {_fmt(row.get('qini_mean'))} | {_fmt(row.get('policy_top10_oracle_value_mean'))} |"
        )
    lines.extend(
        [
            "",
            "## Stability Summary",
            "",
            "| Dataset | Model | OK Runs | PEHE CI Width | QINI Crosses Zero | Verdict | Reason |",
            "| --- | --- | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in payload.get("stability_summary") or []:
        lines.append(
            f"| {row.get('dataset_id')} | {row.get('model')} | {row.get('ok_runs')} | "
            f"{_fmt(row.get('pehe_ci_width'))} | {row.get('qini_ci_crosses_zero')} | "
            f"{row.get('verdict')} | {row.get('reason')} |"
        )
    failure = payload.get("failure_attribution") or {}
    lines.extend(
        [
            "",
            "## Failure Attribution",
            "",
            f"- Status counts: `{failure.get('status_counts')}`",
            f"- Skipped: `{failure.get('skipped')}`",
            f"- Errors: `{failure.get('errors')}`",
            "",
            "## Interview Packaging",
            "",
            "- 30 seconds: smoke evidence proves the deep model code path runs; paper-level evidence adds PEHE/ATE error against known CATE.",
            "- 5 minutes: open `模型拆解 -> Paper-Level Benchmark Evidence`, compare CFRNet/DragonNet/EFIN/DESCN against T/DR baselines, then show the run manifest.",
            "- 15 minutes: explain why IHDP is regression/semi-synthetic, why ACIC-style is guarded local simulation, and why DESCN is skipped on regression tasks but evaluated on classification known-CATE datasets.",
            "",
            "## Boundaries",
            "",
            "- This suite is paper-level offline evidence, not online incrementality proof.",
            "- The ACIC row is ACIC-style local simulation, not an official ACIC raw-file redistribution.",
            "- Promotion still requires overlap, OPE, holdout and rollout monitoring.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-level CATE/uplift benchmarks with PEHE and ATE error.")
    parser.add_argument("--manifest", default="examples/datasets/manifest.json")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="paper-lite")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--known-effect-bootstrap-samples", type=int, default=None)
    parser.add_argument("--artifacts-dir", default="reports/paper_benchmark_runs")
    parser.add_argument("--output-prefix", default="paper_benchmark")
    parser.add_argument("--no-update-latest", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    preset = PRESETS[args.preset]
    output_prefix = str(args.output_prefix or "paper_benchmark").strip() or "paper_benchmark"
    seeds = args.seeds or list(preset["seeds"])
    rows_per_dataset = int(args.rows or preset["rows"])
    known_effect_bootstrap_samples = int(
        args.known_effect_bootstrap_samples
        if args.known_effect_bootstrap_samples is not None
        else preset["known_effect_bootstrap_samples"]
    )

    manifest_path = ROOT / args.manifest
    datasets = _load_manifest(manifest_path)
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    artifacts_dir = ROOT / args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    previous = _previous_payload(reports_dir / f"{output_prefix}_latest.json")

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for dataset_id in args.datasets:
        dataset = datasets.get(dataset_id)
        if dataset is None:
            row = {"dataset_id": dataset_id, "status": "fail", "error": f"unknown dataset: {dataset_id}"}
            rows.append(row)
            failures.append(row["error"])
            continue
        for model_name in args.models:
            for seed in seeds:
                try:
                    row = _run_one(
                        dataset,
                        model_name,
                        rows=rows_per_dataset,
                        seed=seed,
                        artifacts_dir=artifacts_dir,
                        known_effect_bootstrap=known_effect_bootstrap_samples,
                    )
                    if row.get("status") == "fail":
                        failures.append(f"{dataset_id}/{model_name}/seed{seed}: {row.get('error')}")
                    rows.append(row)
                except Exception as exc:
                    rows.append(
                        {
                            "dataset_id": dataset_id,
                            "dataset_name": dataset.get("name"),
                            "model": model_name,
                            "seed": seed,
                            "status": "fail",
                            "error": str(exc),
                        }
                    )
                    failures.append(f"{dataset_id}/{model_name}/seed{seed}: {exc}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    leaderboard = _leaderboard(rows)
    rankings = _rankings_by_metric(leaderboard)
    stability = _stability_summary(leaderboard, seeds)
    payload = {
        "schema_version": 1,
        "status": "fail" if failures else "ok",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "preset": args.preset,
        "benchmark_tier": preset["tier"],
        "manifest": str(manifest_path.relative_to(ROOT)),
        "datasets": args.datasets,
        "models": args.models,
        "seeds": seeds,
        "rows_per_dataset": rows_per_dataset,
        "known_effect_bootstrap_samples": known_effect_bootstrap_samples,
        "runs": rows,
        "leaderboard": leaderboard,
        "rankings": rankings,
        "stability_summary": stability,
        "metric_glossary": glossary_rows("benchmark"),
        "best_by_dataset": _best_by_dataset(rows),
        "failure_attribution": _failure_attribution(rows, failures),
        "failures": failures,
    }
    payload["run_diff"] = _run_diff(previous, payload)

    json_path = reports_dir / f"{output_prefix}_{timestamp}.json"
    csv_path = reports_dir / f"{output_prefix}_{timestamp}.csv"
    leaderboard_path = reports_dir / f"{output_prefix}_leaderboard_{timestamp}.csv"
    diff_path = reports_dir / f"{output_prefix}_run_diff_{timestamp}.json"
    manifest_out = reports_dir / f"{output_prefix}_manifest_{timestamp}.json"
    md_path = reports_dir / f"{output_prefix}_{timestamp}.md"
    doc_path = ROOT / "docs" / "DEEPUplift_PAPER_LEVEL_BENCHMARK.md"
    latest_json = reports_dir / f"{output_prefix}_latest.json"
    latest_csv = reports_dir / f"{output_prefix}_latest.csv"
    latest_leaderboard = reports_dir / f"{output_prefix}_leaderboard_latest.csv"
    latest_diff = reports_dir / f"{output_prefix}_run_diff_latest.json"
    latest_manifest = reports_dir / f"{output_prefix}_manifest_latest.json"
    latest_md = reports_dir / f"{output_prefix}_latest.md"

    save_json(json_path, payload)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    pd.DataFrame(leaderboard).to_csv(leaderboard_path, index=False)
    save_json(diff_path, payload["run_diff"])
    _write_markdown(md_path, payload)
    manifest_files = [json_path, csv_path, leaderboard_path, diff_path, md_path]
    if not args.no_update_latest:
        save_json(latest_json, payload)
        pd.DataFrame(rows).to_csv(latest_csv, index=False)
        pd.DataFrame(leaderboard).to_csv(latest_leaderboard, index=False)
        save_json(latest_diff, payload["run_diff"])
        _write_markdown(latest_md, payload)
        _write_markdown(doc_path, payload)
        manifest_files.append(doc_path)
    suite_manifest = _suite_manifest(manifest_files, rows)
    save_json(manifest_out, suite_manifest)
    if not args.no_update_latest:
        save_json(latest_manifest, suite_manifest)

    print(
        json.dumps(
            {
                "status": payload["status"],
                "json": str(json_path.relative_to(ROOT)),
                "leaderboard": str(leaderboard_path.relative_to(ROOT)),
                "run_diff": str(diff_path.relative_to(ROOT)),
                "manifest": str(manifest_out.relative_to(ROOT)),
                "doc": str((doc_path if not args.no_update_latest else md_path).relative_to(ROOT)),
                "runs": len(rows),
                "leaderboard_rows": len(leaderboard),
                "failures": failures,
            },
            ensure_ascii=False,
        )
    )
    if args.strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
