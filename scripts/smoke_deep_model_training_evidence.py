from __future__ import annotations

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

from deepuplift.core.trainer import train_uplift_model


def _synthetic_interaction_uplift(n: int = 520, seed: int = 20260520) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 7))
    campaign_affinity = 0.55 * x[:, 0] - 0.35 * x[:, 1] + 0.25 * x[:, 2] * x[:, 3]
    propensity = 1.0 / (1.0 + np.exp(-(campaign_affinity + 0.25 * rng.normal(size=n))))
    propensity = np.clip(propensity, 0.08, 0.92)
    treatment = rng.binomial(1, propensity)

    base_logit = -1.25 + 0.5 * x[:, 0] - 0.25 * x[:, 3] + 0.18 * x[:, 5]
    interaction_effect = 0.52 + 0.42 * (x[:, 1] > 0).astype(float) + 0.25 * x[:, 2] * x[:, 4]
    sleeping_dog = 0.55 * (x[:, 6] > 0.9).astype(float)
    cate = interaction_effect - sleeping_dog
    outcome_prob = 1.0 / (1.0 + np.exp(-(base_logit + treatment * cate)))
    outcome = rng.binomial(1, outcome_prob)

    data = {f"f{i}": x[:, i] for i in range(x.shape[1])}
    data.update(
        {
            "treatment": treatment,
            "outcome": outcome,
            "true_uplift": cate,
            "expected_incremental_profit": cate * 18.0 - 1.2,
            "negative_uplift_risk": np.clip(-cate, 0.0, None),
        }
    )
    return pd.DataFrame(data)


def _model_config(model_name: str, feature_cols: list[str], seed: int) -> dict[str, Any]:
    small_tower = {
        "share_dim": 8,
        "share_hidden_dims": [24],
        "base_hidden_dims": [16],
    }
    model_params: dict[str, Any] = dict(small_tower)
    if model_name == "CFRNet":
        model_params.update({"alpha": 0.2, "ipm_mode": "mmd_rbf"})
    elif model_name == "DragonNet":
        model_params.update({"alpha": 0.15, "beta": 0.05, "tarreg": True})
    elif model_name == "EFIN":
        model_params = {"hc_dim": 16, "hu_dim": 8, "is_self": False}
    elif model_name == "DESCN":
        model_params = {"share_dim": 8, "base_dim": 8, "do_rate": 0.0}

    return {
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "model_name": model_name,
        "task": "classification",
        "test_size": 0.25,
        "valid_perc": 0.2,
        "epochs": 2,
        "batch_size": 96,
        "learning_rate": 0.001,
        "random_state": seed,
        "artifacts_dir": "reports/deep_model_training_runs",
        "run_name": f"deep-model-evidence-{model_name.lower()}",
        "model_params": model_params,
        "bootstrap_samples": 0,
        "sensitivity_samples": 0,
        "policy_contact_cost": 0.08,
        "policy_conversion_value": 18.0,
    }


def _clean_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _loss_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {}
    first = history[0]
    last = history[-1]
    return {
        "epochs": len(history),
        "train_loss_first": _clean_float(first.get("train_loss")),
        "train_loss_last": _clean_float(last.get("train_loss")),
        "train_loss_delta": _clean_float((first.get("train_loss") or 0.0) - (last.get("train_loss") or 0.0)),
        "valid_loss_first": _clean_float(first.get("valid_loss")),
        "valid_loss_last": _clean_float(last.get("valid_loss")),
        "outcome_loss_last": _clean_float(last.get("train_outcome_loss")),
        "aux_loss_last": _clean_float(last.get("train_treatment_loss")),
    }


def _top_oracle_recall(metrics: dict[str, Any], fraction: float = 0.1) -> float | None:
    for row in ((metrics.get("oracle_top_k") or {}).get("rows") or []):
        if abs(float(row.get("top_fraction") or 0.0) - fraction) < 1e-9:
            return _clean_float(row.get("topk_recall"))
    return None


def _policy_top(metrics: dict[str, Any]) -> dict[str, Any]:
    policy_best = metrics.get("policy_best") or {}
    return {
        "top_fraction": _clean_float(policy_best.get("top_fraction")),
        "observed_net_value": _clean_float(policy_best.get("observed_net_value")),
        "predicted_net_value": _clean_float(policy_best.get("predicted_net_value")),
    }


def _interview_line(model_name: str, row: dict[str, Any]) -> str:
    qini = row.get("qini_score")
    aux = row.get("aux_loss_last")
    if model_name == "CFRNet":
        return f"CFRNet shows representation balance is not just a doc claim: aux/IPM loss={aux}, QINI={qini}, and the run has a manifest."
    if model_name == "DragonNet":
        return f"DragonNet turns propensity into a trained head plus targeted regularization; this smoke exposes propensity loss={aux} and QINI={qini}."
    if model_name == "EFIN":
        return f"EFIN is the interaction-heavy marketing model; this smoke proves its treatment-aware forward/loss can run through the shared evaluator."
    if model_name == "DESCN":
        return f"DESCN/ESX is the entire-space example: propensity, mu0/mu1, tau and cross-head losses all produce a benchmark row."
    return f"{model_name} has a training artifact and evaluator row."


def _write_markdown(payload: dict[str, Any], leaderboard: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in leaderboard.to_dict(orient="records"):
        rows.append(
            "| {model} | {status} | {qini_score} | {auuc_score} | {oracle_top10_recall} | {train_loss_last} | {aux_loss_last} | `{run_dir}` |".format(
                **{
                    key: row.get(key, "")
                    for key in [
                        "model",
                        "status",
                        "qini_score",
                        "auuc_score",
                        "oracle_top10_recall",
                        "train_loss_last",
                        "aux_loss_last",
                        "run_dir",
                    ]
                }
            )
        )

    output.write_text(
        "\n".join(
            [
                "# Deep Model Training Evidence",
                "",
                f"Generated at: `{payload['generated_at']}`",
                "",
                "This smoke turns model deconstruction into runnable evidence. The same synthetic interaction-uplift data is used for `CFRNet`, `DragonNet`, `EFIN` and `DESCN`, then every model goes through the shared trainer, evaluator, artifact manifest and readiness pipeline.",
                "",
                "## Leaderboard",
                "",
                "| Model | Status | QINI | AUUC | Oracle Top10 Recall | Train Loss Last | Aux/IPM Loss Last | Run Dir |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
                *rows,
                "",
                "## Interview Packaging",
                "",
                "- 30 seconds: this is no longer a model-zoo page; each deep uplift architecture has a runnable smoke, evaluator row and evidence manifest.",
                "- 5 minutes: explain one representative model, then open the UI loss curve and the run manifest to prove the source-level story is executable.",
                "- 15 minutes: compare CFR balance, DragonNet propensity regularization, EFIN treatment interaction and DESCN entire-space losses under the same QINI/AUUC/policy gate.",
                "",
                "## Failure Modes To Call Out",
                "",
                "- These are smoke-level runs, not paper-level convergence claims.",
                "- Neural uplift can overfit small synthetic data; always compare against DR/R/LightGBM baselines.",
                "- Good offline QINI still needs overlap diagnostics, OPE and online incrementality before production rollout.",
                "",
                "## Artifact Contract",
                "",
                "- JSON: `reports/deep_model_training_evidence_latest.json`",
                "- Leaderboard CSV: `reports/deep_model_training_leaderboard_latest.csv`",
                "- Run directories: `reports/deep_model_training_runs/*`",
                "- UI: `模型拆解 -> Deep Model Training Evidence` and `Evidence -> Deep Model Training Evidence`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    seed = 20260520
    df = _synthetic_interaction_uplift(seed=seed)
    feature_cols = [f"f{i}" for i in range(7)]
    models = ["CFRNet", "DragonNet", "EFIN", "DESCN"]
    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for index, model_name in enumerate(models):
        try:
            result = train_uplift_model(
                data_frame=df,
                config=_model_config(model_name, feature_cols, seed + index),
            )
            metrics = result.eval_metrics
            summary = _loss_summary(result.train_history)
            row = {
                "model": model_name,
                "status": "ok",
                "run_id": result.run_id,
                "run_dir": result.run_dir,
                "qini_score": _clean_float(metrics.get("qini_score")),
                "auuc_score": _clean_float(metrics.get("auuc_score")),
                "calibration_mae": _clean_float(metrics.get("calibration_mae")),
                "oracle_top10_recall": _top_oracle_recall(metrics),
                "policy": _policy_top(metrics),
                "readiness_level": (metrics.get("readiness") or {}).get("level"),
                "evidence_manifest": result.artifacts.get("evidence_manifest"),
                "history": result.train_history,
                "loss_summary": summary,
                "train_loss_last": summary.get("train_loss_last"),
                "aux_loss_last": summary.get("aux_loss_last"),
                "interview_line": "",
            }
            row["interview_line"] = _interview_line(model_name, row)
        except Exception as exc:
            row = {
                "model": model_name,
                "status": "error",
                "error": str(exc),
                "interview_line": f"{model_name} is guarded by the smoke runner; failure is recorded instead of breaking demo state silently.",
            }
            failures.append(f"{model_name}: {exc}")
        rows.append(row)

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    leaderboard_cols = [
        "model",
        "status",
        "qini_score",
        "auuc_score",
        "calibration_mae",
        "oracle_top10_recall",
        "train_loss_last",
        "aux_loss_last",
        "run_id",
        "run_dir",
        "evidence_manifest",
        "interview_line",
    ]
    leaderboard = pd.DataFrame([{col: row.get(col) for col in leaderboard_cols} for row in rows])
    leaderboard = leaderboard.sort_values(["status", "qini_score"], ascending=[True, False], na_position="last")

    payload = {
        "status": "ok" if len(ok_rows) >= 3 and not failures else "fail",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "dataset": {
            "rows": len(df),
            "features": len(feature_cols),
            "treatment_rate": _clean_float(df["treatment"].mean()),
            "outcome_rate": _clean_float(df["outcome"].mean()),
            "true_uplift_mean": _clean_float(df["true_uplift"].mean()),
        },
        "models": rows,
        "leaderboard": leaderboard.to_dict(orient="records"),
        "ok_models": len(ok_rows),
        "failures": failures,
        "interview_line": "I can open one deep model, show its source/loss, then show a fresh training evidence row and manifest in the UI.",
    }

    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    json_path = reports / "deep_model_training_evidence_latest.json"
    csv_path = reports / "deep_model_training_leaderboard_latest.csv"
    doc_path = Path("docs/UPLIFT_DEEP_MODEL_TRAINING_EVIDENCE.md")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    leaderboard.to_csv(csv_path, index=False)
    _write_markdown(payload, leaderboard, doc_path)

    print(
        json.dumps(
            {
                "status": payload["status"],
                "json": str(json_path),
                "csv": str(csv_path),
                "doc": str(doc_path),
                "ok_models": len(ok_rows),
                "failures": failures,
            },
            ensure_ascii=False,
        )
    )
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
