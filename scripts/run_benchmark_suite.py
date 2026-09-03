from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core import UpliftConfig, train_uplift_model
from deepuplift.core.artifacts import save_json
from deepuplift.core.diagnostics import diagnose_uplift_data
from deepuplift.core.evaluator import frontier_metric_summary
from deepuplift.core.readiness import decision_readiness
from deepuplift.core.registry import missing_dependencies


DEFAULT_DATASETS = [
    "criteo_visit_10k",
    "doubleml_coupon_uplift",
    "ihdp_npci_1",
    "synthetic_llm_routing_uplift_8k",
]

DEFAULT_MODELS = ["TLearnerGBM", "DRLearnerGBM"]


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["id"]): item for item in payload.get("datasets", [])}


def _top_fraction_row(rows: list[dict[str, Any]] | dict[str, Any], fraction: float) -> dict[str, Any]:
    if isinstance(rows, dict):
        rows = rows.get("rows") or []
    for row in rows or []:
        try:
            if abs(float(row.get("top_fraction", 0.0)) - fraction) < 1e-9:
                return row
        except Exception:
            continue
    return {}


def _overlap_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    overlap = metrics.get("overlap_trim") or {}
    propensity = overlap.get("propensity") or {}
    trim05 = {}
    for row in overlap.get("rows") or []:
        try:
            if abs(float(row.get("trim", 0.0)) - 0.05) < 1e-9:
                trim05 = row
                break
        except Exception:
            continue
    return {
        "weak_overlap_rate": propensity.get("weak_overlap_rate"),
        "propensity_auc": propensity.get("treatment_auc"),
        "trim05_kept_fraction": trim05.get("kept_fraction"),
        "trim05_qini": trim05.get("qini_score"),
    }


def _diagnostic_summary(diagnostics: dict[str, Any]) -> dict[str, Any]:
    selection_bias = diagnostics.get("selection_bias") or {}
    overlap = diagnostics.get("overlap") or {}
    return {
        "diagnostic_warning_count": len(diagnostics.get("warnings") or []),
        "selection_bias_risk": selection_bias.get("risk_level"),
        "diagnostic_propensity_auc": overlap.get("treatment_auc"),
        "diagnostic_weak_overlap_rate": overlap.get("weak_overlap_rate"),
    }


def _metric_row(dataset: dict[str, Any], model_name: str, result: Any, diagnostics: dict[str, Any]) -> dict[str, Any]:
    metrics = result.eval_metrics
    readiness = decision_readiness(metrics)
    top10 = _top_fraction_row(metrics.get("top_k") or [], 0.1)
    oracle_top10 = _top_fraction_row(metrics.get("oracle_top_k") or {}, 0.1)
    policy_best = metrics.get("policy_best") or {}
    overlap = _overlap_summary(metrics)
    frontier = frontier_metric_summary(metrics)
    return {
        "dataset_id": dataset["id"],
        "dataset_name": dataset["name"],
        "dataset_kind": dataset.get("kind"),
        "model": model_name,
        "status": "ok",
        "run_id": result.run_id,
        "run_dir": result.run_dir,
        "evidence_manifest": result.artifacts.get("evidence_manifest"),
        "promotion": result.artifacts.get("promotion"),
        "environment": result.artifacts.get("environment"),
        "readiness_score": readiness.get("score"),
        "readiness_level": readiness.get("level"),
        "readiness_blockers": len(readiness.get("blockers") or []),
        "qini": metrics.get("qini_score"),
        "auuc": metrics.get("auuc_score"),
        "top10_observed_uplift": top10.get("observed_uplift"),
        "top10_rows": top10.get("rows"),
        "policy_top_fraction": policy_best.get("top_fraction"),
        "policy_observed_net_value": policy_best.get("observed_net_value"),
        "policy_predicted_net_value": policy_best.get("predicted_net_value"),
        "calibration_mae": metrics.get("calibration_mae"),
        "sensitivity_verdict": (metrics.get("sensitivity") or {}).get("verdict"),
        "oracle_top10_recall": oracle_top10.get("topk_recall"),
        "oracle_top10_gain_capture": oracle_top10.get("oracle_gain_capture"),
        **overlap,
        **frontier,
        **_diagnostic_summary(diagnostics),
    }


def run_one(
    dataset: dict[str, Any],
    model_name: str,
    *,
    rows: int,
    seed: int,
    artifacts_dir: Path,
    sensitivity_samples: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    missing = missing_dependencies(model_name)
    if missing:
        return {
            "dataset_id": dataset["id"],
            "dataset_name": dataset["name"],
            "dataset_kind": dataset.get("kind"),
            "model": model_name,
            "seed": seed,
            "status": "skipped",
            "missing_dependencies": "; ".join(missing),
        }

    df = pd.read_csv(ROOT / dataset["path"]).head(rows).copy()
    diagnostics = diagnose_uplift_data(df, dataset["treatment_col"], dataset["outcome_col"], dataset["feature_cols"])
    quick = dataset.get("quick_train") or {}
    cfg = UpliftConfig(
        treatment_col=dataset["treatment_col"],
        outcome_col=dataset["outcome_col"],
        feature_cols=dataset["feature_cols"],
        model_name=model_name,
        task=dataset.get("task", "classification"),
        max_rows=rows,
        test_size=0.30,
        random_state=seed,
        epochs=int(quick.get("epochs", 2)),
        batch_size=int(quick.get("batch_size", 128)),
        learning_rate=float(quick.get("learning_rate", 0.001)),
        sensitivity_samples=sensitivity_samples,
        bootstrap_samples=bootstrap_samples,
        artifacts_dir=str(artifacts_dir),
        run_name=f"benchmark-{dataset['id']}-{model_name.lower()}-seed{seed}",
    )
    result = train_uplift_model(config=cfg, data_frame=df)
    row = _metric_row(dataset, model_name, result, diagnostics)
    row["seed"] = seed
    return row


def best_by_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return []
    frame = pd.DataFrame(ok_rows)
    score_cols = ["readiness_score", "policy_observed_net_value", "qini", "top10_observed_uplift"]
    for col in score_cols:
        if col not in frame.columns:
            frame[col] = None
    output = []
    for dataset_id, part in frame.groupby("dataset_id"):
        view = part.sort_values(score_cols, ascending=[False, False, False, False], na_position="last")
        output.append(view.iloc[0].to_dict())
    return output


def _summary(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"mean": None, "std": None, "low": None, "high": None}
    mean = float(numeric.mean())
    std = float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0
    half_width = 1.96 * std / (len(numeric) ** 0.5) if len(numeric) > 1 else 0.0
    return {"mean": mean, "std": std, "low": mean - half_width, "high": mean + half_width}


def benchmark_leaderboard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return []
    frame = pd.DataFrame(ok_rows)
    metrics = [
        "readiness_score",
        "policy_observed_net_value",
        "policy_predicted_net_value",
        "qini",
        "auuc",
        "top10_observed_uplift",
        "calibration_mae",
        "weak_overlap_rate",
        "trim05_kept_fraction",
    ]
    output = []
    for (dataset_id, model), part in frame.groupby(["dataset_id", "model"]):
        seed_values = part["seed"] if "seed" in part else pd.Series(dtype=float)
        row: dict[str, Any] = {
            "dataset_id": dataset_id,
            "dataset_name": part["dataset_name"].dropna().iloc[0] if "dataset_name" in part and not part["dataset_name"].dropna().empty else dataset_id,
            "dataset_kind": part["dataset_kind"].dropna().iloc[0] if "dataset_kind" in part and not part["dataset_kind"].dropna().empty else None,
            "model": model,
            "ok_runs": int(len(part)),
            "seeds": sorted(int(seed) for seed in pd.to_numeric(seed_values, errors="coerce").dropna().unique()),
            "evidence_manifest_count": int(part.get("evidence_manifest", pd.Series(dtype=object)).dropna().nunique()),
        }
        for metric in metrics:
            stats = _summary(part[metric]) if metric in part else {"mean": None, "std": None, "low": None, "high": None}
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
            row[f"{metric}_ci_low"] = stats["low"]
            row[f"{metric}_ci_high"] = stats["high"]
        output.append(row)
    return sorted(
        output,
        key=lambda row: (
            row.get("readiness_score_mean") is not None,
            row.get("readiness_score_mean") or -1,
            row.get("policy_observed_net_value_mean") if row.get("policy_observed_net_value_mean") is not None else -1e18,
            row.get("qini_mean") if row.get("qini_mean") is not None else -1e18,
        ),
        reverse=True,
    )


def failure_attribution(rows: list[dict[str, Any]], failures: list[str]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    missing_dependency_counts: dict[str, int] = {}
    error_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        missing = row.get("missing_dependencies")
        if missing:
            for item in str(missing).split(";"):
                key = item.strip().split(" import failed")[0][:120]
                if key:
                    missing_dependency_counts[key] = missing_dependency_counts.get(key, 0) + 1
        error = row.get("error")
        if error:
            key = str(error).splitlines()[0][:160]
            error_counts[key] = error_counts.get(key, 0) + 1
    return {
        "status_counts": status_counts,
        "missing_dependency_counts": missing_dependency_counts,
        "error_counts": error_counts,
        "failure_count": len(failures),
        "failures": failures,
    }


def _leaderboard_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    leaderboard = payload.get("leaderboard")
    if isinstance(leaderboard, list):
        return [row for row in leaderboard if isinstance(row, dict)]
    return benchmark_leaderboard(payload.get("runs") or [])


def benchmark_run_diff(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    if not previous:
        return {"status": "no_previous", "rows": [], "summary": "No previous benchmark suite found."}
    prev_rows = _leaderboard_from_payload(previous)
    curr_rows = _leaderboard_from_payload(current)
    prev_index = {(row.get("dataset_id"), row.get("model")): row for row in prev_rows}
    curr_index = {(row.get("dataset_id"), row.get("model")): row for row in curr_rows}
    keys = sorted(set(prev_index) | set(curr_index), key=lambda item: (str(item[0]), str(item[1])))
    diff_rows = []
    for key in keys:
        prev = prev_index.get(key)
        curr = curr_index.get(key)
        if prev is None:
            diff_rows.append({"dataset_id": key[0], "model": key[1], "change_type": "added"})
            continue
        if curr is None:
            diff_rows.append({"dataset_id": key[0], "model": key[1], "change_type": "removed"})
            continue
        row = {"dataset_id": key[0], "model": key[1], "change_type": "changed"}
        changed = False
        for metric in ["readiness_score_mean", "policy_observed_net_value_mean", "qini_mean", "top10_observed_uplift_mean"]:
            prev_value = prev.get(metric)
            curr_value = curr.get(metric)
            delta = None if prev_value is None or curr_value is None else float(curr_value) - float(prev_value)
            changed = changed or (delta is not None and abs(delta) > 1e-9)
            row[f"previous_{metric}"] = prev_value
            row[f"current_{metric}"] = curr_value
            row[f"delta_{metric}"] = delta
        row["previous_ok_runs"] = prev.get("ok_runs")
        row["current_ok_runs"] = curr.get("ok_runs")
        if changed or row["previous_ok_runs"] != row["current_ok_runs"]:
            diff_rows.append(row)
    return {
        "status": "ok",
        "previous_generated_at": previous.get("generated_at"),
        "current_generated_at": current.get("generated_at"),
        "rows": diff_rows,
        "summary": f"{len(diff_rows)} leaderboard rows changed, added, or removed.",
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# DeepUplift Benchmark Suite Report",
        "",
        f"Generated at: `{payload['generated_at']}`",
        "",
        "## Summary",
        "",
        f"- Status: `{payload['status']}`",
        f"- Datasets: {len(payload['datasets'])}",
        f"- Candidate models: {', '.join(payload['models'])}",
        f"- Seeds: {', '.join(str(seed) for seed in payload.get('seeds') or [])}",
        f"- Rows per dataset: {payload['rows_per_dataset']}",
        f"- Successful runs: {sum(1 for row in payload['runs'] if row.get('status') == 'ok')}",
        f"- Skipped / failed runs: {sum(1 for row in payload['runs'] if row.get('status') != 'ok')}",
        "",
        "## Best By Dataset",
        "",
        "| Dataset | Model | Readiness | QINI | Top10 Uplift | Promotion | Evidence Manifest |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload.get("best_by_dataset") or []:
        lines.append(
            "| {dataset} | {model} | {readiness} | {qini} | {top10} | `{promotion}` | `{manifest}` |".format(
                dataset=row.get("dataset_id"),
                model=row.get("model"),
                readiness=row.get("readiness_score"),
                qini=_fmt(row.get("qini")),
                top10=_fmt(row.get("top10_observed_uplift")),
                promotion=row.get("promotion"),
                manifest=row.get("evidence_manifest"),
            )
        )
    lines.extend(
        [
            "",
            "## Leaderboard",
            "",
            "| Dataset | Model | OK Runs | Readiness Mean | QINI Mean | QINI 95% CI | Policy Net Mean | Top10 Mean |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for row in payload.get("leaderboard") or []:
        lines.append(
            "| {dataset} | {model} | {ok_runs} | {readiness} | {qini} | [{qini_low}, {qini_high}] | {policy} | {top10} |".format(
                dataset=row.get("dataset_id"),
                model=row.get("model"),
                ok_runs=row.get("ok_runs"),
                readiness=_fmt(row.get("readiness_score_mean")),
                qini=_fmt(row.get("qini_mean")),
                qini_low=_fmt(row.get("qini_ci_low")),
                qini_high=_fmt(row.get("qini_ci_high")),
                policy=_fmt(row.get("policy_observed_net_value_mean")),
                top10=_fmt(row.get("top10_observed_uplift_mean")),
            )
        )
    failure_payload = payload.get("failure_attribution") or {}
    lines.extend(
        [
            "",
            "## Failure Attribution",
            "",
            f"- Status counts: `{failure_payload.get('status_counts')}`",
            f"- Missing dependency counts: `{failure_payload.get('missing_dependency_counts')}`",
            f"- Error counts: `{failure_payload.get('error_counts')}`",
        ]
    )
    run_diff = payload.get("run_diff") or {}
    lines.extend(
        [
            "",
            "## Run Diff",
            "",
            f"- Status: `{run_diff.get('status')}`",
            f"- Summary: {run_diff.get('summary')}",
        ]
    )
    lines.extend(
        [
            "",
            "## Review Notes",
            "",
            "- This suite is a benchmark protocol, not an online causal proof.",
            "- Promotion still requires overlap, calibration, bootstrap, policy value, and online holdout validation.",
            "- Each successful row links to an `evidence_manifest.json` with environment snapshot and artifact hashes.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
        if pd.isna(number):
            return "NA"
        return f"{number:.6g}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a source-backed DeepUplift benchmark suite.")
    parser.add_argument("--manifest", default="examples/datasets/manifest.json")
    parser.add_argument("--dataset-id", action="append", dest="dataset_ids")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--rows", type=int, default=400)
    parser.add_argument("--sensitivity-samples", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--artifacts-dir", default="reports/benchmark_suite_runs")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any requested run fails or is skipped.")
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    datasets = load_manifest(manifest_path)
    selected = args.dataset_ids or DEFAULT_DATASETS
    artifacts_dir = ROOT / args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    previous_payload = None
    previous_latest = reports_dir / "benchmark_suite_latest.json"
    if previous_latest.exists():
        try:
            previous_payload = json.loads(previous_latest.read_text(encoding="utf-8"))
        except Exception:
            previous_payload = None

    rows = []
    failures = []
    for dataset_id in selected:
        if dataset_id not in datasets:
            message = f"unknown dataset: {dataset_id}"
            rows.append({"dataset_id": dataset_id, "status": "fail", "error": message})
            failures.append(message)
            continue
        dataset = datasets[dataset_id]
        for model_name in args.models:
            for seed in args.seeds:
                try:
                    row = run_one(
                        dataset,
                        model_name,
                        rows=args.rows,
                        seed=seed,
                        artifacts_dir=artifacts_dir,
                        sensitivity_samples=args.sensitivity_samples,
                        bootstrap_samples=args.bootstrap_samples,
                    )
                    if row.get("status") != "ok":
                        failures.append(f"{dataset_id}/{model_name}/seed{seed}: {row.get('missing_dependencies') or row.get('error')}")
                    rows.append(row)
                except Exception as exc:
                    failures.append(f"{dataset_id}/{model_name}/seed{seed}: {exc}")
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

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = reports_dir / f"benchmark_suite_{timestamp}.json"
    csv_path = reports_dir / f"benchmark_suite_{timestamp}.csv"
    latest_json = reports_dir / "benchmark_suite_latest.json"
    latest_csv = reports_dir / "benchmark_suite_latest.csv"
    leaderboard_csv = reports_dir / f"benchmark_leaderboard_{timestamp}.csv"
    latest_leaderboard_csv = reports_dir / "benchmark_leaderboard_latest.csv"
    run_diff_json = reports_dir / f"benchmark_run_diff_{timestamp}.json"
    latest_run_diff_json = reports_dir / "benchmark_run_diff_latest.json"
    md_path = reports_dir / f"benchmark_suite_{timestamp}.md"
    latest_md = reports_dir / "benchmark_suite_latest.md"

    leaderboard = benchmark_leaderboard(rows)
    payload = {
        "status": "fail" if failures else "ok",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "datasets": selected,
        "models": args.models,
        "seeds": args.seeds,
        "rows_per_dataset": args.rows,
        "runs": rows,
        "best_by_dataset": best_by_dataset(rows),
        "leaderboard": leaderboard,
        "failure_attribution": failure_attribution(rows, failures),
        "failures": failures,
        "csv": str(csv_path.relative_to(ROOT)),
        "leaderboard_csv": str(leaderboard_csv.relative_to(ROOT)),
        "run_diff_json": str(run_diff_json.relative_to(ROOT)),
        "markdown": str(md_path.relative_to(ROOT)),
    }
    payload["run_diff"] = benchmark_run_diff(previous_payload, payload)
    save_json(json_path, payload)
    save_json(latest_json, payload)
    save_json(run_diff_json, payload["run_diff"])
    save_json(latest_run_diff_json, payload["run_diff"])
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    pd.DataFrame(rows).to_csv(latest_csv, index=False)
    pd.DataFrame(leaderboard).to_csv(leaderboard_csv, index=False)
    pd.DataFrame(leaderboard).to_csv(latest_leaderboard_csv, index=False)
    write_markdown(md_path, payload)
    write_markdown(latest_md, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "json": str(json_path.relative_to(ROOT)),
                "csv": str(csv_path.relative_to(ROOT)),
                "markdown": str(md_path.relative_to(ROOT)),
                "runs": len(rows),
                "leaderboard_rows": len(leaderboard),
                "run_diff": str(run_diff_json.relative_to(ROOT)),
                "failures": failures,
            },
            ensure_ascii=False,
        )
    )
    if args.strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
