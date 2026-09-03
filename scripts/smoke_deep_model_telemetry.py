from __future__ import annotations

import json
import math
import csv
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEEP_MODELS = ("CFRNet", "DragonNet", "EFIN", "DESCN")


ADVANCED_HOOKS = {
    "CFRNet": "phi_x representation export for per-epoch IPM/MMD and treated-control distance histograms",
    "DragonNet": "e_hat propensity export plus separate targeted-regularization loss curve",
    "EFIN": "interaction attention export plus treatment-feature ablation",
    "DESCN": "mu0/mu1/tau/estr/escr head export plus cross-head consistency residuals",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def clean_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def metric(value: Any) -> str:
    number = clean_float(value)
    return "NA" if number is None else f"{number:.4g}"


def latest_training_rows(root: Path) -> list[dict[str, Any]]:
    payload = read_json(root / "reports" / "deep_model_training_evidence_latest.json")
    rows = payload.get("leaderboard") or []
    return [row for row in rows if isinstance(row, dict) and row.get("model") in DEEP_MODELS]


def load_history(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "train_history.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_predictions(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "predictions.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def manifest_file_count(manifest: dict[str, Any]) -> int:
    files = manifest.get("files")
    if isinstance(files, list):
        return len(files)
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, dict):
        return len(artifacts)
    return 0


def delta_from_history(history: pd.DataFrame, col: str) -> float | None:
    if history.empty or col not in history.columns or len(history[col].dropna()) < 2:
        return None
    values = pd.to_numeric(history[col], errors="coerce").dropna()
    if len(values) < 2:
        return None
    return clean_float(values.iloc[0] - values.iloc[-1])


def prediction_telemetry(predictions: pd.DataFrame) -> dict[str, Any]:
    if predictions.empty or "uplift_score" not in predictions.columns:
        return {
            "prediction_rows": 0,
            "uplift_score_std": None,
            "positive_score_rate": None,
            "top10_true_uplift_mean": None,
            "all_true_uplift_mean": None,
            "top10_true_uplift_gain": None,
        }
    uplift = pd.to_numeric(predictions["uplift_score"], errors="coerce")
    rows = len(predictions)
    positive_rate = clean_float((uplift > 0).mean())
    uplift_std = clean_float(uplift.std(ddof=0))
    top_n = max(1, int(math.ceil(rows * 0.1)))
    top = predictions.assign(_uplift=uplift).sort_values("_uplift", ascending=False).head(top_n)
    if "true_uplift" in predictions.columns:
        true_uplift = pd.to_numeric(predictions["true_uplift"], errors="coerce")
        top_true = pd.to_numeric(top["true_uplift"], errors="coerce")
        all_mean = clean_float(true_uplift.mean())
        top_mean = clean_float(top_true.mean())
        gain = clean_float((top_mean or 0.0) - (all_mean or 0.0))
    else:
        all_mean = None
        top_mean = None
        gain = None
    return {
        "prediction_rows": rows,
        "uplift_score_std": uplift_std,
        "positive_score_rate": positive_rate,
        "top10_true_uplift_mean": top_mean,
        "all_true_uplift_mean": all_mean,
        "top10_true_uplift_gain": gain,
    }


def cfr_balance_telemetry(root: Path) -> dict[str, Any]:
    diff = read_json(root / "reports" / "cfrnet_differentiable_balance_smoke_latest.json")
    train = read_json(root / "reports" / "cfrnet_training_balance_smoke_latest.json")
    results = diff.get("results") or {}
    grad_norms = [clean_float(row.get("grad_norm")) for row in results.values() if isinstance(row, dict)]
    grad_norms = [value for value in grad_norms if value is not None]
    terms = [clean_float(value) for value in train.get("balance_terms") or []]
    terms = [value for value in terms if value is not None]
    balance_delta = None
    if len(terms) >= 2:
        balance_delta = clean_float(terms[0] - terms[-1])
    return {
        "balance_modes": len(results),
        "max_balance_grad_norm": max(grad_norms) if grad_norms else None,
        "balance_term_points": len(terms),
        "balance_term_delta": balance_delta,
    }


def model_row(root: Path, training_row: dict[str, Any]) -> dict[str, Any]:
    model = str(training_row.get("model"))
    run_dir = Path(str(training_row.get("run_dir") or ""))
    history = load_history(run_dir)
    predictions = load_predictions(run_dir)
    metrics = read_json(run_dir / "metrics.json")
    manifest = read_json(run_dir / "evidence_manifest.json")
    pred = prediction_telemetry(predictions)

    train_loss_delta = delta_from_history(history, "train_loss")
    valid_loss_delta = delta_from_history(history, "valid_loss")
    outcome_loss_delta = delta_from_history(history, "train_outcome_loss")
    aux_loss_delta = delta_from_history(history, "train_treatment_loss")
    has_history = not history.empty
    has_predictions = pred.get("prediction_rows", 0) > 0
    has_manifest = bool(manifest)
    basic_pass = has_history and has_predictions and has_manifest and clean_float(train_loss_delta) is not None
    advanced_hook = ADVANCED_HOOKS.get(model, "model-specific forward export")
    advanced_status = "missing_guarded_hook"
    if model == "CFRNet":
        cfr = cfr_balance_telemetry(root)
    else:
        cfr = {}
    pass_gate = "basic telemetry pass; advanced hook still required" if basic_pass else "basic telemetry incomplete"
    fail_action = "wire the model-specific forward hook before stronger paper-level claims" if basic_pass else "rerun training smoke and inspect missing run artifacts"
    return {
        "model": model,
        "status": "ok" if basic_pass else "review",
        "run_dir": str(run_dir),
        "epochs": int(len(history)) if has_history else 0,
        "train_loss_delta": train_loss_delta,
        "valid_loss_delta": valid_loss_delta,
        "outcome_loss_delta": outcome_loss_delta,
        "aux_loss_delta": aux_loss_delta,
        "qini_score": clean_float(metrics.get("qini_score") or training_row.get("qini_score")),
        "auuc_score": clean_float(metrics.get("auuc_score") or training_row.get("auuc_score")),
        "calibration_mae": clean_float(metrics.get("calibration_mae") or training_row.get("calibration_mae")),
        "prediction_rows": pred.get("prediction_rows"),
        "uplift_score_std": pred.get("uplift_score_std"),
        "positive_score_rate": pred.get("positive_score_rate"),
        "top10_true_uplift_mean": pred.get("top10_true_uplift_mean"),
        "all_true_uplift_mean": pred.get("all_true_uplift_mean"),
        "top10_true_uplift_gain": pred.get("top10_true_uplift_gain"),
        "manifest_files": manifest_file_count(manifest),
        "basic_telemetry_pass": basic_pass,
        "advanced_hook_status": advanced_status,
        "advanced_hook_needed": advanced_hook,
        "cfr_balance_modes": cfr.get("balance_modes"),
        "cfr_max_balance_grad_norm": cfr.get("max_balance_grad_norm"),
        "cfr_balance_term_points": cfr.get("balance_term_points"),
        "cfr_balance_term_delta": cfr.get("balance_term_delta"),
        "pass_gate": pass_gate,
        "fail_action": fail_action,
        "ui_route": "Model Deconstruction -> Deep Model Telemetry Smoke -> Telemetry Gap Matrix",
        "interview_line": (
            f"{model} basic telemetry: train_loss_delta={metric(train_loss_delta)}, "
            f"uplift_score_std={metric(pred.get('uplift_score_std'))}, "
            f"top10_true_uplift_gain={metric(pred.get('top10_true_uplift_gain'))}; "
            f"advanced hook={advanced_status}."
        ),
    }


def build_payload(root: Path) -> dict[str, Any]:
    rows = [model_row(root, row) for row in latest_training_rows(root)]
    status_counts: dict[str, int] = {}
    hook_counts: dict[str, int] = {}
    for row in rows:
        status_counts[str(row.get("status"))] = status_counts.get(str(row.get("status")), 0) + 1
        hook_counts[str(row.get("advanced_hook_status"))] = hook_counts.get(str(row.get("advanced_hook_status")), 0) + 1
    return {
        "schema_version": 1,
        "status": "ok" if len(rows) >= 4 and all(row.get("basic_telemetry_pass") for row in rows) else "review",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "summary": {
            "models": len({row["model"] for row in rows}),
            "rows": len(rows),
            "basic_telemetry_pass": sum(1 for row in rows if row.get("basic_telemetry_pass")),
            "advanced_hook_gaps": sum(1 for row in rows if row.get("advanced_hook_status") == "missing_guarded_hook"),
            "status_counts": status_counts,
            "advanced_hook_status_counts": hook_counts,
        },
        "rows": rows,
        "interview_tracks": {
            "30s": "Telemetry smoke proves the deep models have inspectable run artifacts: loss history, predictions, metrics and manifest.",
            "5min": "It separates basic telemetry that already exists from advanced hooks still needed for representation, propensity, attention or head-level claims.",
            "15min": "Use telemetry smoke before rewriting models: first prove the evidence pipe can read loss/prediction/manifest signals, then implement guarded forward hooks model by model.",
        },
    }


def flatten(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model",
        "status",
        "epochs",
        "train_loss_delta",
        "valid_loss_delta",
        "outcome_loss_delta",
        "aux_loss_delta",
        "qini_score",
        "auuc_score",
        "calibration_mae",
        "prediction_rows",
        "uplift_score_std",
        "positive_score_rate",
        "top10_true_uplift_gain",
        "manifest_files",
        "basic_telemetry_pass",
        "advanced_hook_status",
        "advanced_hook_needed",
        "cfr_balance_modes",
        "cfr_max_balance_grad_norm",
        "cfr_balance_term_points",
        "cfr_balance_term_delta",
        "pass_gate",
        "fail_action",
        "ui_route",
        "interview_line",
        "run_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: flatten(row.get(field)) for field in fields})


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    tracks = payload.get("interview_tracks") or {}
    table = "\n".join(
        "| {model} | {status} | {loss} | {spread} | {gain} | {hook} |".format(
            model=row.get("model", ""),
            status=row.get("status", ""),
            loss=metric(row.get("train_loss_delta")),
            spread=metric(row.get("uplift_score_std")),
            gain=metric(row.get("top10_true_uplift_gain")),
            hook=str(row.get("advanced_hook_status", "")).replace("|", "/"),
        )
        for row in rows
    )
    details = []
    for row in rows:
        details.append(
            f"""## {row.get("model")}

Run dir: `{row.get("run_dir")}`
Status: `{row.get("status")}`
Epochs: `{row.get("epochs")}`
Train loss delta: `{metric(row.get("train_loss_delta"))}`
Uplift score std: `{metric(row.get("uplift_score_std"))}`
Top10 true uplift gain: `{metric(row.get("top10_true_uplift_gain"))}`

Advanced hook needed: {row.get("advanced_hook_needed")}

Pass gate: {row.get("pass_gate")}

Fail action: {row.get("fail_action")}

Interview line: {row.get("interview_line")}
"""
        )
    return f"""# DeepUplift Deep Model Telemetry Smoke

Generated at: `{payload.get("generated_at")}`

This smoke reads the latest deep-model training artifacts and checks that each model has basic observable telemetry: training history, prediction spread, known-CATE top segment signal, metrics and evidence manifest. It intentionally does not pretend that advanced hooks already exist.

## Summary

| Metric | Value |
| --- | ---: |
| Models | {summary.get("models", 0)} |
| Rows | {summary.get("rows", 0)} |
| Basic telemetry pass | {summary.get("basic_telemetry_pass", 0)} |
| Advanced hook gaps | {summary.get("advanced_hook_gaps", 0)} |

## Interview Tracks

- 30s: {tracks.get("30s", "")}
- 5min: {tracks.get("5min", "")}
- 15min: {tracks.get("15min", "")}

## Telemetry Overview

| Model | Status | Train loss delta | Uplift score std | Top10 true uplift gain | Advanced hook |
| --- | --- | ---: | ---: | ---: | --- |
{table}

{chr(10).join(details)}
"""


def main() -> None:
    root = Path(".")
    payload = build_payload(root)
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    json_path = reports / "deep_model_telemetry_smoke_latest.json"
    csv_path = reports / "deep_model_telemetry_smoke_latest.csv"
    doc_path = root / "docs" / "DEEPUplift_DEEP_MODEL_TELEMETRY_SMOKE.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, payload.get("rows") or [])
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(build_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "json": str(json_path),
                "csv": str(csv_path),
                "markdown": str(doc_path),
                "rows": (payload.get("summary") or {}).get("rows"),
                "basic_telemetry_pass": (payload.get("summary") or {}).get("basic_telemetry_pass"),
            },
            ensure_ascii=False,
        )
    )
    if payload.get("status") not in {"ok", "review"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
