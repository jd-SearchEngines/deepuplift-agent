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
from scripts.run_paper_benchmark_suite import (
    DEFAULT_DATASETS,
    TRUE_EFFECT_COLS,
    _clean_float,
    _fmt,
    _known_effect_metrics,
    _load_manifest,
    _metric_at_fraction,
)


PRESETS = {
    "tuned-smoke": {
        "datasets": ["acic_style_synthetic_known_cate_1k6", "synthetic_known_cate_shift_1k8"],
        "seeds": [20260520],
        "rows": 180,
        "known_effect_bootstrap_samples": 4,
        "tier": "deep_tuned_smoke",
    },
    "tuned-v1": {
        "datasets": DEFAULT_DATASETS,
        "seeds": [20260520, 20260521],
        "rows": 320,
        "known_effect_bootstrap_samples": 12,
        "tier": "deep_tuned_multi_seed",
    },
}


VARIANTS: list[dict[str, Any]] = [
    {
        "variant_id": "TLearnerGBM_baseline",
        "model": "TLearnerGBM",
        "family": "baseline_meta_learner",
        "role": "Strong tabular baseline for effect accuracy and ranking.",
        "params": {},
        "train": {},
    },
    {
        "variant_id": "DRLearnerGBM_baseline",
        "model": "DRLearnerGBM",
        "family": "baseline_orthogonal_learner",
        "role": "Doubly robust baseline; often wins when nuisance models handle confounding well.",
        "params": {},
        "train": {},
    },
    {
        "variant_id": "CFRNet_default",
        "model": "CFRNet",
        "family": "representation_balancing",
        "role": "Default balance regularization for representation-level treatment/control matching.",
        "params": {"share_dim": 8, "share_hidden_dims": [24], "base_hidden_dims": [16], "alpha": 0.2, "ipm_mode": "mmd_rbf"},
        "train": {"epochs": 2, "learning_rate": 0.001},
    },
    {
        "variant_id": "CFRNet_low_balance",
        "model": "CFRNet",
        "family": "representation_balancing",
        "role": "Ablation with weaker IPM pressure; checks whether balance is over-regularizing small data.",
        "params": {"share_dim": 8, "share_hidden_dims": [24], "base_hidden_dims": [16], "alpha": 0.05, "ipm_mode": "mmd_rbf"},
        "train": {"epochs": 2, "learning_rate": 0.001},
    },
    {
        "variant_id": "CFRNet_high_balance",
        "model": "CFRNet",
        "family": "representation_balancing",
        "role": "Ablation with stronger IPM pressure; checks whether imbalance drives failure.",
        "params": {"share_dim": 8, "share_hidden_dims": [24], "base_hidden_dims": [16], "alpha": 0.5, "ipm_mode": "mmd_rbf"},
        "train": {"epochs": 2, "learning_rate": 0.001},
    },
    {
        "variant_id": "DragonNet_default",
        "model": "DragonNet",
        "family": "targeted_regularization",
        "role": "Default DragonNet with propensity head and targeted regularization.",
        "params": {"share_dim": 8, "share_hidden_dims": [24], "base_hidden_dims": [16], "alpha": 0.15, "beta": 0.05, "tarreg": True},
        "train": {"epochs": 2, "learning_rate": 0.001},
    },
    {
        "variant_id": "DragonNet_no_tarreg",
        "model": "DragonNet",
        "family": "targeted_regularization",
        "role": "Ablation disabling targeted regularization; isolates the propensity-head effect.",
        "params": {"share_dim": 8, "share_hidden_dims": [24], "base_hidden_dims": [16], "alpha": 0.15, "beta": 0.0, "tarreg": False},
        "train": {"epochs": 2, "learning_rate": 0.001},
    },
    {
        "variant_id": "DragonNet_stronger_tarreg",
        "model": "DragonNet",
        "family": "targeted_regularization",
        "role": "Ablation with stronger targeted regularization for confounded treatment assignment.",
        "params": {"share_dim": 8, "share_hidden_dims": [24], "base_hidden_dims": [16], "alpha": 0.2, "beta": 0.1, "tarreg": True},
        "train": {"epochs": 2, "learning_rate": 0.001},
    },
    {
        "variant_id": "EFIN_default",
        "model": "EFIN",
        "family": "feature_interaction_deep_uplift",
        "role": "Default EFIN interaction architecture for marketing-style treatment effects.",
        "params": {"hc_dim": 16, "hu_dim": 8, "is_self": False},
        "train": {"epochs": 2, "learning_rate": 0.001},
    },
    {
        "variant_id": "EFIN_wide",
        "model": "EFIN",
        "family": "feature_interaction_deep_uplift",
        "role": "Wider hidden dimensions; checks whether EFIN is capacity-limited.",
        "params": {"hc_dim": 32, "hu_dim": 16, "is_self": False},
        "train": {"epochs": 2, "learning_rate": 0.001},
    },
    {
        "variant_id": "EFIN_no_attention",
        "model": "EFIN",
        "family": "feature_interaction_deep_uplift",
        "role": "Ablation replacing learned treatment-feature attention with uniform pooling; isolates whether interaction attention adds ranking or policy value.",
        "params": {"hc_dim": 16, "hu_dim": 8, "is_self": False, "use_interaction_attention": False},
        "train": {"epochs": 2, "learning_rate": 0.001},
    },
    {
        "variant_id": "EFIN_path_regularized",
        "model": "EFIN",
        "family": "feature_interaction_deep_uplift",
        "role": "Conservative uplift-path scaling ablation; checks whether the uplift path is too dominant versus the base response path.",
        "params": {"hc_dim": 16, "hu_dim": 8, "is_self": False, "uplift_tau_scale": 0.5},
        "train": {"epochs": 2, "learning_rate": 0.001},
    },
    {
        "variant_id": "DESCN_default",
        "model": "DESCN",
        "family": "entire_space_deep_uplift",
        "role": "Default DESCN/ESX entire-space architecture; classification-only by design.",
        "params": {"share_dim": 8, "base_dim": 8, "do_rate": 0.0},
        "train": {"epochs": 2, "learning_rate": 0.001},
    },
    {
        "variant_id": "DESCN_wide_dropout",
        "model": "DESCN",
        "family": "entire_space_deep_uplift",
        "role": "Wider towers plus dropout; checks whether DESCN needs capacity and regularization.",
        "params": {"share_dim": 16, "base_dim": 16, "do_rate": 0.1},
        "train": {"epochs": 2, "learning_rate": 0.001},
    },
    {
        "variant_id": "DESCN_no_constraint",
        "model": "DESCN",
        "family": "entire_space_deep_uplift",
        "role": "Ablation disabling cross-head consistency losses; isolates whether DESCN constraints help PEHE/QINI or only add optimization pressure.",
        "params": {"share_dim": 8, "base_dim": 8, "do_rate": 0.0, "mu1hat_w": 0.0, "mu0hat_w": 0.0},
        "train": {"epochs": 2, "learning_rate": 0.001},
    },
]


LOWER_IS_BETTER = {"pehe", "ate_error"}
LEADERBOARD_METRICS = [
    "pehe",
    "ate_error",
    "effect_corr",
    "qini",
    "auuc",
    "policy_observed_net_value",
    "policy_top10_oracle_value",
    "oracle_top10_recall",
    "train_loss_delta",
    "valid_loss_delta",
]


def _unsupported_task_reason(model_name: str, task: str) -> str | None:
    spec = MODEL_REGISTRY.get(model_name)
    if spec is None:
        return f"unknown model: {model_name}"
    if task not in spec.supported_tasks:
        return f"{model_name} supports {spec.supported_tasks}; dataset task is {task}"
    return None


def _summary(values: pd.Series, *, lower_is_better: bool = False) -> dict[str, Any]:
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


def _loss_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {
            "epochs": 0,
            "train_loss_first": None,
            "train_loss_last": None,
            "train_loss_delta": None,
            "valid_loss_first": None,
            "valid_loss_last": None,
            "valid_loss_delta": None,
            "aux_loss_last": None,
        }
    first = history[0]
    last = history[-1]
    train_first = _clean_float(first.get("train_loss"))
    train_last = _clean_float(last.get("train_loss"))
    valid_first = _clean_float(first.get("valid_loss"))
    valid_last = _clean_float(last.get("valid_loss"))
    return {
        "epochs": len(history),
        "train_loss_first": train_first,
        "train_loss_last": train_last,
        "train_loss_delta": None if train_first is None or train_last is None else train_first - train_last,
        "valid_loss_first": valid_first,
        "valid_loss_last": valid_last,
        "valid_loss_delta": None if valid_first is None or valid_last is None else valid_first - valid_last,
        "outcome_loss_last": _clean_float(last.get("train_outcome_loss")),
        "aux_loss_last": _clean_float(last.get("train_treatment_loss")),
    }


def _leaderboard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok = [row for row in rows if row.get("status") == "ok"]
    if not ok:
        return []
    frame = pd.DataFrame(ok)
    output: list[dict[str, Any]] = []
    for (dataset_id, variant_id), part in frame.groupby(["dataset_id", "variant_id"]):
        first = part.iloc[0].to_dict()
        row: dict[str, Any] = {
            "dataset_id": dataset_id,
            "dataset_name": first.get("dataset_name"),
            "dataset_kind": first.get("dataset_kind"),
            "model": first.get("model"),
            "variant_id": variant_id,
            "family": first.get("family"),
            "role": first.get("role"),
            "ok_runs": int(len(part)),
            "seeds": sorted(int(seed) for seed in pd.to_numeric(part["seed"], errors="coerce").dropna().unique()),
            "evidence_manifest_count": int(part["evidence_manifest"].dropna().nunique()) if "evidence_manifest" in part else 0,
        }
        for metric in LEADERBOARD_METRICS:
            stats = _summary(part[metric], lower_is_better=metric in LOWER_IS_BETTER) if metric in part else _summary(pd.Series(dtype=float))
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
            row[f"{metric}_ci_low"] = stats["low"]
            row[f"{metric}_ci_high"] = stats["high"]
            row[f"{metric}_best"] = stats["best"]
        output.append(row)
    return sorted(
        output,
        key=lambda row: (
            row.get("dataset_id") or "",
            row.get("pehe_mean") is None,
            row.get("pehe_mean") if row.get("pehe_mean") is not None else 1e18,
            -(row.get("qini_mean") if row.get("qini_mean") is not None else -1e18),
        ),
    )


def _rankings(leaderboard: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    fields = {
        "pehe": ("pehe_mean", True),
        "ate_error": ("ate_error_mean", True),
        "qini": ("qini_mean", False),
        "auuc": ("auuc_mean", False),
        "policy_top10_oracle_value": ("policy_top10_oracle_value_mean", False),
        "oracle_top10_recall": ("oracle_top10_recall_mean", False),
    }
    output: dict[str, list[dict[str, Any]]] = {}
    for metric, (field, lower) in fields.items():
        rows = [row for row in leaderboard if row.get(field) is not None]
        ordered = sorted(rows, key=lambda row: float(row[field]), reverse=not lower)
        output[metric] = [
            {
                "rank": index + 1,
                "dataset_id": row.get("dataset_id"),
                "model": row.get("model"),
                "variant_id": row.get("variant_id"),
                "metric": field,
                "value": row.get(field),
                "ok_runs": row.get("ok_runs"),
                "ci_low": row.get(field.replace("_mean", "_ci_low")),
                "ci_high": row.get(field.replace("_mean", "_ci_high")),
            }
            for index, row in enumerate(ordered)
        ]
    return output


def _best_baselines(leaderboard: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for dataset_id in sorted({row.get("dataset_id") for row in leaderboard}):
        candidates = [
            row
            for row in leaderboard
            if row.get("dataset_id") == dataset_id
            and str(row.get("family") or "").startswith("baseline")
            and row.get("pehe_mean") is not None
        ]
        if candidates:
            output[str(dataset_id)] = sorted(
                candidates,
                key=lambda row: (float(row.get("pehe_mean") or 1e18), float(row.get("ate_error_mean") or 1e18)),
            )[0]
    return output


def _baseline_comparison(leaderboard: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = _best_baselines(leaderboard)
    rows: list[dict[str, Any]] = []
    for row in leaderboard:
        baseline = baselines.get(str(row.get("dataset_id")))
        if not baseline or str(row.get("family") or "").startswith("baseline"):
            continue
        pehe_delta = _delta(row.get("pehe_mean"), baseline.get("pehe_mean"))
        qini_delta = _delta(row.get("qini_mean"), baseline.get("qini_mean"))
        auuc_delta = _delta(row.get("auuc_mean"), baseline.get("auuc_mean"))
        policy_delta = _delta(row.get("policy_top10_oracle_value_mean"), baseline.get("policy_top10_oracle_value_mean"))
        if pehe_delta is not None and pehe_delta <= 0 and qini_delta is not None and qini_delta >= 0:
            verdict = "beats_baseline_on_accuracy_and_ranking"
        elif pehe_delta is not None and pehe_delta <= 0:
            verdict = "accuracy_gain_ranking_tradeoff"
        elif qini_delta is not None and qini_delta >= 0:
            verdict = "ranking_gain_accuracy_tradeoff"
        else:
            verdict = "baseline_still_stronger"
        rows.append(
            {
                "dataset_id": row.get("dataset_id"),
                "model": row.get("model"),
                "variant_id": row.get("variant_id"),
                "best_baseline": baseline.get("variant_id"),
                "baseline_pehe": baseline.get("pehe_mean"),
                "variant_pehe": row.get("pehe_mean"),
                "delta_pehe_vs_baseline": pehe_delta,
                "delta_qini_vs_baseline": qini_delta,
                "delta_auuc_vs_baseline": auuc_delta,
                "delta_policy_top10_oracle_value_vs_baseline": policy_delta,
                "verdict": verdict,
                "explanation": _baseline_explanation(row, baseline, verdict),
            }
        )
    return rows


def _ablation_comparison(leaderboard: list[dict[str, Any]]) -> list[dict[str, Any]]:
    default_rows = {
        (row.get("dataset_id"), row.get("model")): row
        for row in leaderboard
        if str(row.get("variant_id") or "").endswith("_default")
    }
    rows: list[dict[str, Any]] = []
    for row in leaderboard:
        variant_id = str(row.get("variant_id") or "")
        if variant_id.endswith("_default") or str(row.get("family") or "").startswith("baseline"):
            continue
        baseline = default_rows.get((row.get("dataset_id"), row.get("model")))
        if not baseline:
            continue
        pehe_delta = _delta(row.get("pehe_mean"), baseline.get("pehe_mean"))
        qini_delta = _delta(row.get("qini_mean"), baseline.get("qini_mean"))
        rows.append(
            {
                "dataset_id": row.get("dataset_id"),
                "model": row.get("model"),
                "variant_id": row.get("variant_id"),
                "default_variant": baseline.get("variant_id"),
                "delta_pehe_vs_default": pehe_delta,
                "delta_qini_vs_default": qini_delta,
                "delta_train_loss_delta_vs_default": _delta(row.get("train_loss_delta_mean"), baseline.get("train_loss_delta_mean")),
                "verdict": _ablation_verdict(pehe_delta, qini_delta),
            }
        )
    return rows


def _delta(current: Any, reference: Any) -> float | None:
    left = _clean_float(current)
    right = _clean_float(reference)
    if left is None or right is None:
        return None
    return left - right


def _ablation_verdict(pehe_delta: float | None, qini_delta: float | None) -> str:
    if pehe_delta is not None and pehe_delta <= 0 and qini_delta is not None and qini_delta >= 0:
        return "ablation_improves_accuracy_and_ranking"
    if pehe_delta is not None and pehe_delta <= 0:
        return "ablation_improves_accuracy"
    if qini_delta is not None and qini_delta >= 0:
        return "ablation_improves_ranking"
    return "ablation_not_better_than_default"


def _baseline_explanation(row: dict[str, Any], baseline: dict[str, Any], verdict: str) -> str:
    model = row.get("model")
    if verdict == "baseline_still_stronger":
        return (
            f"{baseline.get('variant_id')} remains stronger on this small tabular known-CATE slice. "
            f"{model} may need more samples, longer training, stronger regularization tuning, or better representation diagnostics before paper-level claims."
        )
    if verdict == "ranking_gain_accuracy_tradeoff":
        return (
            f"{model} improves ranking/policy signals but has worse effect accuracy than {baseline.get('variant_id')}; "
            "this is a useful business-targeting candidate but not a clean CATE-estimation win."
        )
    if verdict == "accuracy_gain_ranking_tradeoff":
        return (
            f"{model} improves PEHE/ATE against the baseline but ranking value still lags; "
            "inspect monotonicity, calibration and Top-K policy curves before promotion."
        )
    return f"{model} beats the best tabular baseline on both effect accuracy and ranking for this dataset slice."


def _battle_cards(
    leaderboard: list[dict[str, Any]],
    baseline_comparison: list[dict[str, Any]],
    ablation_comparison: list[dict[str, Any]],
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    comparison_by_variant = {
        (row.get("dataset_id"), row.get("variant_id")): row for row in baseline_comparison
    }
    ablation_by_variant = {
        (row.get("dataset_id"), row.get("variant_id")): row for row in ablation_comparison
    }
    skipped_by_variant: dict[tuple[Any, Any, Any], list[str]] = {}
    for row in runs:
        if row.get("status") == "ok":
            continue
        key = (row.get("dataset_id"), row.get("variant_id"), row.get("model"))
        reason = row.get("skip_reason") or row.get("missing_dependencies") or row.get("error") or row.get("status")
        skipped_by_variant.setdefault(key, []).append(str(reason))

    cards: list[dict[str, Any]] = []
    for row in leaderboard:
        key = (row.get("dataset_id"), row.get("variant_id"))
        comparison = comparison_by_variant.get(key)
        ablation = ablation_by_variant.get(key)
        loss_delta = _clean_float(row.get("train_loss_delta_mean"))
        if str(row.get("family") or "").startswith("baseline"):
            verdict = "baseline_reference"
            reason = "Tree/meta learner baseline is used as the tabular reference for deep-model claims."
        elif comparison:
            verdict = comparison.get("verdict") or "needs_review"
            reason = comparison.get("explanation") or ""
        else:
            verdict = "needs_review"
            reason = "No comparable baseline row found."
        if loss_delta is not None and loss_delta < -1e-6:
            verdict = f"{verdict}+training_loss_review"
            reason = reason + " Training loss increased; inspect learning rate, batch size and seed stability."
        cards.append(
            {
                "dataset_id": row.get("dataset_id"),
                "model": row.get("model"),
                "variant_id": row.get("variant_id"),
                "family": row.get("family"),
                "verdict": verdict,
                "reason": reason,
                "pehe_mean": row.get("pehe_mean"),
                "qini_mean": row.get("qini_mean"),
                "auuc_mean": row.get("auuc_mean"),
                "policy_top10_oracle_value_mean": row.get("policy_top10_oracle_value_mean"),
                "train_loss_delta_mean": row.get("train_loss_delta_mean"),
                "baseline_delta": comparison,
                "ablation_delta": ablation,
                "interview_line": _interview_line(row, comparison, ablation),
            }
        )
    for (dataset_id, variant_id, model_name), reasons in skipped_by_variant.items():
        cards.append(
            {
                "dataset_id": dataset_id,
                "variant_id": variant_id,
                "model": model_name,
                "verdict": "guarded_skip",
                "reason": "; ".join(sorted(set(reasons)))[:300],
                "interview_line": "This row proves the benchmark is guarded: unsupported tasks or missing optional dependencies are recorded rather than crashing the demo.",
            }
        )
    return sorted(cards, key=lambda row: (str(row.get("dataset_id")), str(row.get("variant_id"))))


def _interview_line(row: dict[str, Any], comparison: dict[str, Any] | None, ablation: dict[str, Any] | None) -> str:
    if str(row.get("family") or "").startswith("baseline"):
        return (
            f"{row.get('variant_id')} is the reference point: PEHE={_fmt(row.get('pehe_mean'))}, "
            f"QINI={_fmt(row.get('qini_mean'))}. I use it to avoid over-claiming deep models on small tabular data."
        )
    if comparison:
        return (
            f"{row.get('variant_id')} vs {comparison.get('best_baseline')}: "
            f"delta PEHE={_fmt(comparison.get('delta_pehe_vs_baseline'))}, "
            f"delta QINI={_fmt(comparison.get('delta_qini_vs_baseline'))}; verdict={comparison.get('verdict')}."
        )
    if ablation:
        return f"{row.get('variant_id')} ablation verdict: {ablation.get('verdict')}."
    return f"{row.get('variant_id')} has runnable tuned benchmark evidence and a linked manifest."


def _failure_attribution(runs: list[dict[str, Any]], battle_cards: list[dict[str, Any]], failures: list[str]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    verdict_counts: dict[str, int] = {}
    for row in runs:
        status = str(row.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        reason = row.get("skip_reason") or row.get("missing_dependencies") or row.get("error")
        if reason:
            key = str(reason).splitlines()[0][:180]
            reason_counts[key] = reason_counts.get(key, 0) + 1
    for card in battle_cards:
        verdict = str(card.get("verdict") or "unknown")
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    return {
        "status_counts": status_counts,
        "reason_counts": reason_counts,
        "verdict_counts": verdict_counts,
        "failures": failures,
        "default_failure_reading": [
            "If T/DR wins PEHE on IHDP or ACIC-style, the likely cause is small tabular sample size plus strong tree nuisance fits.",
            "If deep models win QINI/AUUC but lose PEHE, treat them as ranking candidates, not pure CATE estimators.",
            "If loss does not decrease, tune learning rate/epochs/batch size before making architecture claims.",
            "DESCN is classification-only in the current registry, so regression dataset skips are expected and guarded.",
        ],
    }


def _model_scorecards(
    leaderboard: list[dict[str, Any]],
    battle_cards: list[dict[str, Any]],
    baseline_comparison: list[dict[str, Any]],
    ablation_comparison: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_model: dict[str, list[dict[str, Any]]] = {}
    for row in baseline_comparison:
        baseline_by_model.setdefault(str(row.get("model")), []).append(row)
    ablation_by_model: dict[str, list[dict[str, Any]]] = {}
    for row in ablation_comparison:
        ablation_by_model.setdefault(str(row.get("model")), []).append(row)
    cards_by_model: dict[str, list[dict[str, Any]]] = {}
    for card in battle_cards:
        model = card.get("model")
        if model:
            cards_by_model.setdefault(str(model), []).append(card)

    rows: list[dict[str, Any]] = []
    for model in sorted({str(row.get("model")) for row in leaderboard if row.get("model")} | set(cards_by_model)):
        model_lb = [row for row in leaderboard if row.get("model") == model]
        cards = cards_by_model.get(model, [])
        verdict_counts: dict[str, int] = {}
        for card in cards:
            verdict = str(card.get("verdict") or "unknown")
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        best_pehe_row = min(
            [row for row in model_lb if row.get("pehe_mean") is not None],
            key=lambda row: float(row.get("pehe_mean") or 1e18),
            default={},
        )
        best_qini_row = max(
            [row for row in model_lb if row.get("qini_mean") is not None],
            key=lambda row: float(row.get("qini_mean") or -1e18),
            default={},
        )
        comparison_rows = baseline_by_model.get(model, [])
        avg_delta_pehe = _mean_delta(comparison_rows, "delta_pehe_vs_baseline")
        avg_delta_qini = _mean_delta(comparison_rows, "delta_qini_vs_baseline")
        verdict = _model_verdict(model, verdict_counts, avg_delta_pehe, avg_delta_qini)
        rows.append(
            {
                "model": model,
                "verdict": verdict,
                "datasets": sorted({str(row.get("dataset_id")) for row in model_lb if row.get("dataset_id")}),
                "variants": sorted({str(row.get("variant_id")) for row in model_lb if row.get("variant_id")}),
                "ok_runs": int(sum(int(row.get("ok_runs") or 0) for row in model_lb)),
                "battle_cards": len(cards),
                "verdict_counts": verdict_counts,
                "best_pehe_variant": best_pehe_row.get("variant_id"),
                "best_pehe": best_pehe_row.get("pehe_mean"),
                "best_qini_variant": best_qini_row.get("variant_id"),
                "best_qini": best_qini_row.get("qini_mean"),
                "avg_delta_pehe_vs_baseline": avg_delta_pehe,
                "avg_delta_qini_vs_baseline": avg_delta_qini,
                "ablation_rows": len(ablation_by_model.get(model, [])),
                "strength": _model_strength(model, verdict, avg_delta_pehe, avg_delta_qini),
                "watchout": _model_watchout(model, verdict, verdict_counts),
                "interview_line": _model_interview_line(model, verdict, best_pehe_row, best_qini_row),
            }
        )
    return rows


def _mean_delta(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [_clean_float(row.get(field)) for row in rows]
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return float(np.mean(clean))


def _model_verdict(model: str, verdict_counts: dict[str, int], avg_delta_pehe: float | None, avg_delta_qini: float | None) -> str:
    if model in {"TLearnerGBM", "DRLearnerGBM"}:
        return "strong_baseline_reference"
    if verdict_counts.get("guarded_skip") and not any(key.startswith("beats_baseline") for key in verdict_counts):
        return "guarded_or_task_limited"
    if any(key.startswith("beats_baseline_on_accuracy_and_ranking") for key in verdict_counts):
        return "tuned_candidate"
    if any(key.startswith("ranking_gain_accuracy_tradeoff") for key in verdict_counts):
        return "ranking_candidate"
    if avg_delta_pehe is not None and avg_delta_pehe <= 0:
        return "accuracy_candidate"
    if avg_delta_qini is not None and avg_delta_qini > 0:
        return "ranking_candidate"
    return "needs_more_tuning"


def _model_strength(model: str, verdict: str, avg_delta_pehe: float | None, avg_delta_qini: float | None) -> str:
    if verdict == "strong_baseline_reference":
        return "Strong small-tabular reference that prevents over-claiming neural uplift."
    if verdict == "tuned_candidate":
        return "At least one tuned variant beats the best T/DR baseline on both effect accuracy and ranking."
    if verdict == "ranking_candidate":
        return "Useful when ranking or policy utility improves even if CATE accuracy lags."
    if verdict == "accuracy_candidate":
        return "Shows effect-accuracy improvement but still needs ranking/policy checks."
    if verdict == "guarded_or_task_limited":
        return "Correctly records task/dependency boundaries instead of silently breaking the demo."
    return f"Needs more tuning; avg delta PEHE={_fmt(avg_delta_pehe)}, avg delta QINI={_fmt(avg_delta_qini)}."


def _model_watchout(model: str, verdict: str, verdict_counts: dict[str, int]) -> str:
    if model == "DESCN":
        return "DESCN is classification-only in the current registry; regression skips are guarded task boundaries."
    if model in {"CFRNet", "DragonNet", "EFIN"} and verdict == "needs_more_tuning":
        return "Do not sell this as a paper-level win; position it as architecture depth plus failure attribution evidence."
    if any("training_loss_review" in key for key in verdict_counts):
        return "Some variants show training-loss review flags; inspect learning rate, epochs and seed stability."
    if model in {"TLearnerGBM", "DRLearnerGBM"}:
        return "Baseline wins do not prove production readiness; still need overlap/OPE/online incrementality."
    return "Offline known-CATE evidence is not online incrementality proof."


def _model_interview_line(model: str, verdict: str, best_pehe_row: dict[str, Any], best_qini_row: dict[str, Any]) -> str:
    return (
        f"{model}: verdict={verdict}; best PEHE variant={best_pehe_row.get('variant_id', 'NA')} "
        f"({_fmt(best_pehe_row.get('pehe_mean'))}), best QINI variant={best_qini_row.get('variant_id', 'NA')} "
        f"({_fmt(best_qini_row.get('qini_mean'))})."
    )


def _interview_tracks(payload: dict[str, Any]) -> dict[str, str]:
    failure = payload.get("failure_attribution") or {}
    verdicts = failure.get("verdict_counts") or {}
    scorecards = payload.get("model_scorecards") or []
    candidate_models = [row.get("model") for row in scorecards if row.get("verdict") in {"tuned_candidate", "ranking_candidate", "accuracy_candidate"}]
    baseline_models = [row.get("model") for row in scorecards if row.get("verdict") == "strong_baseline_reference"]
    return {
        "30s": (
            "我把深度模型从 smoke evidence 升级成 tuned battle evidence：同一 known-CATE benchmark 下，"
            "T/DR baseline、CFRNet、DragonNet、EFIN、DESCN 都有 PEHE/QINI/policy/loss 和 failure attribution。"
        ),
        "5min": (
            f"先讲 baseline reference={baseline_models or ['TLearnerGBM','DRLearnerGBM']}，再讲 candidate={candidate_models or 'none yet'}。"
            f"当前 verdict 分布是 {verdicts}，所以安全说法是：哪些模型赢了 PEHE，哪些只赢 ranking/policy，哪些需要调参。"
        ),
        "15min": (
            "打开 Evidence 或 Model Deconstruction：先看 leaderboard，再看 baseline delta，再看 ablation。"
            "如果 T/DR 赢，解释小样本 tabular + tree nuisance；如果深度模型只赢 QINI，解释 ranking candidate；"
            "如果 DESCN skip，解释 registry task boundary。"
        ),
        "30min": (
            "逐个模型讲源码/loss：CFRNet 的 IPM，DragonNet 的 propensity+targeted regularization，EFIN 的 treatment-aware interaction，"
            "DESCN 的 entire-space heads。最后回到 manifest、run diff、bootstrap/seed 和上线前的 OPE/holdout。"
        ),
    }


def _recommendations(scorecards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for card in scorecards:
        verdict = card.get("verdict")
        if verdict == "strong_baseline_reference":
            action = "Keep as P0 benchmark baseline and compare every neural claim against it."
        elif verdict == "tuned_candidate":
            action = "Promote to paper-lite follow-up with more seeds, larger rows, and representation/propensity diagnostics."
        elif verdict in {"ranking_candidate", "accuracy_candidate"}:
            action = "Keep as candidate; add metric-conflict analysis and policy/OPE validation before claiming superiority."
        elif verdict == "guarded_or_task_limited":
            action = "Keep guarded; document task boundary and add classification-only demo route."
        else:
            action = "Run wider hyperparameter sweep before interview-level claims."
        rows.append(
            {
                "model": card.get("model"),
                "verdict": verdict,
                "priority": "P0" if verdict in {"strong_baseline_reference", "tuned_candidate"} else "P1",
                "action": action,
            }
        )
    return rows


def _previous_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _run_diff(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    if not previous:
        return {"status": "no_previous", "summary": "No previous deep tuned benchmark found.", "rows": []}
    prev = {(row.get("dataset_id"), row.get("variant_id")): row for row in previous.get("leaderboard") or []}
    curr = {(row.get("dataset_id"), row.get("variant_id")): row for row in current.get("leaderboard") or []}
    rows: list[dict[str, Any]] = []
    for key in sorted(set(prev) | set(curr), key=lambda item: (str(item[0]), str(item[1]))):
        if key not in prev:
            rows.append({"dataset_id": key[0], "variant_id": key[1], "change_type": "added"})
            continue
        if key not in curr:
            rows.append({"dataset_id": key[0], "variant_id": key[1], "change_type": "removed"})
            continue
        diff: dict[str, Any] = {"dataset_id": key[0], "variant_id": key[1], "change_type": "changed"}
        changed = False
        for metric in ["pehe_mean", "ate_error_mean", "qini_mean", "auuc_mean", "policy_top10_oracle_value_mean"]:
            delta = _delta(curr[key].get(metric), prev[key].get(metric))
            diff[f"previous_{metric}"] = prev[key].get(metric)
            diff[f"current_{metric}"] = curr[key].get(metric)
            diff[f"delta_{metric}"] = delta
            changed = changed or (delta is not None and abs(delta) > 1e-9)
        if changed:
            rows.append(diff)
    return {"status": "ok", "summary": f"{len(rows)} deep tuned leaderboard rows changed.", "rows": rows}


def _suite_manifest(output_paths: list[Path], run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    files = []
    for path in output_paths:
        if path.exists() and path.is_file():
            files.append({"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    run_manifest_paths = [Path(str(row.get("evidence_manifest"))) for row in run_rows if row.get("status") == "ok" and row.get("evidence_manifest")]
    return {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "artifact": "deep_model_tuned_benchmark",
        "files": files,
        "run_evidence_manifests": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in run_manifest_paths
            if path.exists()
        ],
        "environment": build_environment_snapshot(ROOT),
    }


def _run_one(
    dataset: dict[str, Any],
    variant: dict[str, Any],
    *,
    rows: int,
    seed: int,
    artifacts_dir: Path,
    known_effect_bootstrap: int,
) -> dict[str, Any]:
    model_name = variant["model"]
    variant_id = variant["variant_id"]
    task = dataset.get("task", "classification")
    unsupported = _unsupported_task_reason(model_name, task)
    if unsupported:
        return {
            "dataset_id": dataset["id"],
            "dataset_name": dataset.get("name"),
            "dataset_kind": dataset.get("kind"),
            "model": model_name,
            "variant_id": variant_id,
            "family": variant.get("family"),
            "role": variant.get("role"),
            "seed": seed,
            "status": "skipped_task",
            "skip_reason": unsupported,
        }
    missing = missing_dependencies(model_name)
    if missing:
        return {
            "dataset_id": dataset["id"],
            "dataset_name": dataset.get("name"),
            "dataset_kind": dataset.get("kind"),
            "model": model_name,
            "variant_id": variant_id,
            "family": variant.get("family"),
            "role": variant.get("role"),
            "seed": seed,
            "status": "skipped_dependency",
            "missing_dependencies": "; ".join(missing),
        }

    data_path = ROOT / dataset["path"]
    frame = pd.read_csv(data_path).head(rows).copy()
    quick = dataset.get("quick_train") or {}
    train = variant.get("train") or {}
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
            epochs=int(train.get("epochs", quick.get("epochs", 2))),
            batch_size=int(train.get("batch_size", quick.get("batch_size", 128))),
            learning_rate=float(train.get("learning_rate", quick.get("learning_rate", 0.001))),
            random_state=seed,
            artifacts_dir=str(artifacts_dir),
            run_name=f"deep-tuned-{dataset['id']}-{variant_id.lower()}-seed{seed}",
            bootstrap_samples=0,
            sensitivity_samples=0,
            policy_contact_cost=0.05,
            policy_conversion_value=10.0,
            model_params=variant.get("params") or {},
        ),
    )
    predictions = pd.read_csv(result.artifacts["predictions"])
    known_metrics = _known_effect_metrics(predictions, known_effect_bootstrap, seed=seed + 7919)
    metrics = result.eval_metrics
    top10 = _metric_at_fraction(metrics.get("top_k") or [], 0.1)
    oracle_top10 = _metric_at_fraction(metrics.get("oracle_top_k") or {}, 0.1)
    policy_best = metrics.get("policy_best") or {}
    qini_boot = (((metrics.get("bootstrap") or {}).get("metrics") or {}).get("qini_score") or {})
    loss = _loss_summary(result.train_history)
    return {
        "dataset_id": dataset["id"],
        "dataset_name": dataset.get("name"),
        "dataset_kind": dataset.get("kind"),
        "model": model_name,
        "variant_id": variant_id,
        "family": variant.get("family"),
        "role": variant.get("role"),
        "model_params": variant.get("params") or {},
        "seed": seed,
        "status": "ok",
        "run_id": result.run_id,
        "run_dir": result.run_dir,
        "evidence_manifest": result.artifacts.get("evidence_manifest"),
        "predictions": result.artifacts.get("predictions"),
        "metrics_path": result.artifacts.get("metrics"),
        "history": result.artifacts.get("history"),
        "history_rows": result.train_history,
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
        **loss,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Deep Model Tuned Benchmark And Failure Attribution",
        "",
        f"Generated at: `{payload['generated_at']}`",
        "",
        "## Summary",
        "",
        f"- Status: `{payload['status']}`",
        f"- Tier: `{payload.get('benchmark_tier')}`",
        f"- Datasets: {', '.join(payload.get('datasets') or [])}",
        f"- Seeds: {', '.join(str(seed) for seed in payload.get('seeds') or [])}",
        f"- Variants: {len(payload.get('variants') or [])}",
        f"- Successful runs: {sum(1 for row in payload.get('runs') or [] if row.get('status') == 'ok')}",
        f"- Guarded skips/failures: {sum(1 for row in payload.get('runs') or [] if row.get('status') != 'ok')}",
        "",
        "## How To Read",
        "",
        "- This suite explains why a deep uplift model wins or loses against strong tabular T/DR baselines.",
        "- PEHE/ATE error evaluate known-CATE effect accuracy; QINI/AUUC/policy value evaluate ranking and targeting utility.",
        "- Loss deltas are smoke-level training diagnostics; they are not a substitute for multi-seed convergence evidence.",
        "",
        "## Battle Interpretation / 面试讲法",
        "",
        "| Duration | Talk Track |",
        "| --- | --- |",
    ]
    for duration, text in (payload.get("interview_tracks") or {}).items():
        lines.append(f"| {duration} | {str(text).replace(chr(10), ' ')} |")
    lines.extend(
        [
            "",
            "## Model Scorecards",
            "",
            "| Model | Verdict | OK Runs | Best PEHE Variant | Best PEHE | Best QINI Variant | Best QINI | Watchout |",
            "| --- | --- | ---: | --- | ---: | --- | ---: | --- |",
        ]
    )
    for row in payload.get("model_scorecards") or []:
        watchout = str(row.get("watchout") or "").replace("\n", " ")
        lines.append(
            f"| {row.get('model')} | {row.get('verdict')} | {row.get('ok_runs')} | {row.get('best_pehe_variant')} | "
            f"{_fmt(row.get('best_pehe'))} | {row.get('best_qini_variant')} | {_fmt(row.get('best_qini'))} | {watchout} |"
        )
    lines.extend(
        [
            "",
            "## Recommended Actions",
            "",
            "| Model | Verdict | Priority | Action |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in payload.get("recommendations") or []:
        lines.append(f"| {row.get('model')} | {row.get('verdict')} | {row.get('priority')} | {row.get('action')} |")
    lines.extend(
        [
            "",
        "## Leaderboard",
        "",
        "| Dataset | Variant | Model | OK Runs | PEHE | QINI | AUUC | Policy Oracle Top10 | Train Loss Delta |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload.get("leaderboard") or []:
        lines.append(
            f"| {row.get('dataset_id')} | {row.get('variant_id')} | {row.get('model')} | {row.get('ok_runs')} | "
            f"{_fmt(row.get('pehe_mean'))} | {_fmt(row.get('qini_mean'))} | {_fmt(row.get('auuc_mean'))} | "
            f"{_fmt(row.get('policy_top10_oracle_value_mean'))} | {_fmt(row.get('train_loss_delta_mean'))} |"
        )
    lines.extend(
        [
            "",
            "## Baseline Battle",
            "",
            "| Dataset | Variant | Best Baseline | Delta PEHE | Delta QINI | Delta AUUC | Verdict |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload.get("baseline_comparison") or []:
        lines.append(
            f"| {row.get('dataset_id')} | {row.get('variant_id')} | {row.get('best_baseline')} | "
            f"{_fmt(row.get('delta_pehe_vs_baseline'))} | {_fmt(row.get('delta_qini_vs_baseline'))} | "
            f"{_fmt(row.get('delta_auuc_vs_baseline'))} | {row.get('verdict')} |"
        )
    lines.extend(
        [
            "",
            "## Failure Attribution Cards",
            "",
            "| Dataset | Variant | Verdict | Reason | Interview Line |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload.get("battle_cards") or []:
        reason = str(row.get("reason") or "").replace("\n", " ")
        interview_line = str(row.get("interview_line") or "").replace("\n", " ")
        lines.append(f"| {row.get('dataset_id')} | {row.get('variant_id')} | {row.get('verdict')} | {reason} | {interview_line} |")
    lines.extend(
        [
            "",
            "## Why T/DR Often Wins",
            "",
            "- On small tabular benchmarks, tree-based T/DR learners can fit nonlinear nuisance functions with less optimization risk.",
            "- Deep representation learners need enough samples, balanced treatment assignment and tuned regularization to convert representation capacity into CATE accuracy.",
            "- A deep model can still be useful when PEHE loses but QINI/AUUC or policy value improves; then the safe claim is ranking or targeting, not unbiased CATE estimation.",
            "",
            "## Evidence Contract",
            "",
            "- JSON: `reports/deep_model_tuned_benchmark_latest.json`",
            "- Leaderboard: `reports/deep_model_tuned_benchmark_leaderboard_latest.csv`",
            "- Manifest: `reports/deep_model_tuned_benchmark_manifest_latest.json`",
            "- Docs: `docs/DEEPUplift_DEEP_MODEL_TUNING_BENCHMARK.md`",
            "- UI: Evidence Dashboard and Model Deconstruction Workbench.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _variant_by_id(variant_ids: list[str] | None) -> list[dict[str, Any]]:
    if not variant_ids:
        return VARIANTS
    wanted = set(variant_ids)
    return [variant for variant in VARIANTS if variant["variant_id"] in wanted or variant["model"] in wanted]


def _selected_variant_ids(*groups: list[str] | None) -> list[str] | None:
    selected: list[str] = []
    for group in groups:
        selected.extend(group or [])
    return list(dict.fromkeys(selected)) or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tuned deep uplift benchmarks with baseline battle and failure attribution.")
    parser.add_argument("--manifest", default="examples/datasets/manifest.json")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="tuned-smoke")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None, help="Alias selector for model names; merged with --variants.")
    parser.add_argument("--include-variants", nargs="+", default=None, help="Alias selector for explicit variant ids; merged with --variants.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--known-effect-bootstrap-samples", type=int, default=None)
    parser.add_argument("--artifacts-dir", default="reports/deep_model_tuned_runs")
    parser.add_argument("--output-prefix", default="deep_model_tuned_benchmark")
    parser.add_argument("--no-update-latest", action="store_true")
    parser.add_argument("--emit-battle-cards", action="store_true", help="Compatibility flag; battle cards are always emitted.")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    output_prefix = str(args.output_prefix or "deep_model_tuned_benchmark").strip() or "deep_model_tuned_benchmark"
    seeds = args.seeds or list(preset["seeds"])
    rows_per_dataset = int(args.rows or preset["rows"])
    datasets_arg = args.datasets or list(preset["datasets"])
    variant_selectors = _selected_variant_ids(args.variants, args.models, args.include_variants)
    variants = _variant_by_id(variant_selectors)
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

    run_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for dataset_id in datasets_arg:
        dataset = datasets.get(dataset_id)
        if dataset is None:
            row = {"dataset_id": dataset_id, "status": "fail", "error": f"unknown dataset: {dataset_id}"}
            run_rows.append(row)
            failures.append(row["error"])
            continue
        for variant in variants:
            for seed in seeds:
                try:
                    row = _run_one(
                        dataset,
                        variant,
                        rows=rows_per_dataset,
                        seed=seed,
                        artifacts_dir=artifacts_dir,
                        known_effect_bootstrap=known_effect_bootstrap_samples,
                    )
                    if row.get("status") == "fail":
                        failures.append(f"{dataset_id}/{variant['variant_id']}/seed{seed}: {row.get('error')}")
                    run_rows.append(row)
                except Exception as exc:  # noqa: BLE001
                    run_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "dataset_name": dataset.get("name"),
                            "dataset_kind": dataset.get("kind"),
                            "model": variant.get("model"),
                            "variant_id": variant.get("variant_id"),
                            "family": variant.get("family"),
                            "role": variant.get("role"),
                            "seed": seed,
                            "status": "fail",
                            "error": str(exc),
                        }
                    )
                    failures.append(f"{dataset_id}/{variant['variant_id']}/seed{seed}: {exc}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    leaderboard = _leaderboard(run_rows)
    baseline = _baseline_comparison(leaderboard)
    ablation = _ablation_comparison(leaderboard)
    battle_cards = _battle_cards(leaderboard, baseline, ablation, run_rows)
    model_scorecards = _model_scorecards(leaderboard, battle_cards, baseline, ablation)
    payload = {
        "schema_version": 1,
        "status": "fail" if failures else "ok",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "preset": args.preset,
        "benchmark_tier": preset["tier"],
        "manifest": str(manifest_path.relative_to(ROOT)),
        "datasets": datasets_arg,
        "seeds": seeds,
        "rows_per_dataset": rows_per_dataset,
        "known_effect_bootstrap_samples": known_effect_bootstrap_samples,
        "variants": [{k: v for k, v in variant.items() if k != "train"} for variant in variants],
        "runs": run_rows,
        "leaderboard": leaderboard,
        "rankings": _rankings(leaderboard),
        "baseline_comparison": baseline,
        "ablation_comparison": ablation,
        "battle_cards": battle_cards,
        "model_scorecards": model_scorecards,
        "failure_attribution": _failure_attribution(run_rows, battle_cards, failures),
        "metric_glossary": glossary_rows("benchmark"),
        "true_effect_columns": TRUE_EFFECT_COLS,
        "failures": failures,
    }
    payload["interview_tracks"] = _interview_tracks(payload)
    payload["recommendations"] = _recommendations(model_scorecards)
    payload["run_diff"] = _run_diff(previous, payload)

    json_path = reports_dir / f"{output_prefix}_{timestamp}.json"
    csv_path = reports_dir / f"{output_prefix}_{timestamp}.csv"
    leaderboard_path = reports_dir / f"{output_prefix}_leaderboard_{timestamp}.csv"
    battle_path = reports_dir / f"{output_prefix}_battle_cards_{timestamp}.csv"
    diff_path = reports_dir / f"{output_prefix}_run_diff_{timestamp}.json"
    manifest_out = reports_dir / f"{output_prefix}_manifest_{timestamp}.json"
    md_path = reports_dir / f"{output_prefix}_{timestamp}.md"
    doc_path = ROOT / "docs" / "DEEPUplift_DEEP_MODEL_TUNING_BENCHMARK.md"
    latest_json = reports_dir / f"{output_prefix}_latest.json"
    latest_csv = reports_dir / f"{output_prefix}_latest.csv"
    latest_leaderboard = reports_dir / f"{output_prefix}_leaderboard_latest.csv"
    latest_battle = reports_dir / f"{output_prefix}_battle_cards_latest.csv"
    latest_diff = reports_dir / f"{output_prefix}_run_diff_latest.json"
    latest_manifest = reports_dir / f"{output_prefix}_manifest_latest.json"
    latest_md = reports_dir / f"{output_prefix}_latest.md"

    save_json(json_path, payload)
    pd.DataFrame(run_rows).to_csv(csv_path, index=False)
    pd.DataFrame(leaderboard).to_csv(leaderboard_path, index=False)
    pd.DataFrame(battle_cards).to_csv(battle_path, index=False)
    save_json(diff_path, payload["run_diff"])
    _write_markdown(md_path, payload)
    manifest_files = [json_path, csv_path, leaderboard_path, battle_path, diff_path, md_path]
    if not args.no_update_latest:
        save_json(latest_json, payload)
        pd.DataFrame(run_rows).to_csv(latest_csv, index=False)
        pd.DataFrame(leaderboard).to_csv(latest_leaderboard, index=False)
        pd.DataFrame(battle_cards).to_csv(latest_battle, index=False)
        save_json(latest_diff, payload["run_diff"])
        _write_markdown(latest_md, payload)
        _write_markdown(doc_path, payload)
        manifest_files.append(doc_path)
    suite_manifest = _suite_manifest(manifest_files, run_rows)
    save_json(manifest_out, suite_manifest)
    if not args.no_update_latest:
        save_json(latest_manifest, suite_manifest)

    ok_runs = sum(1 for row in run_rows if row.get("status") == "ok")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "json": str(json_path.relative_to(ROOT)),
                "leaderboard": str(leaderboard_path.relative_to(ROOT)),
                "battle_cards": str(battle_path.relative_to(ROOT)),
                "manifest": str(manifest_out.relative_to(ROOT)),
                "doc": str((doc_path if not args.no_update_latest else md_path).relative_to(ROOT)),
                "runs": len(run_rows),
                "ok_runs": ok_runs,
                "leaderboard_rows": len(leaderboard),
                "battle_cards_rows": len(battle_cards),
                "failures": failures,
            },
            ensure_ascii=False,
        )
    )
    if args.strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
