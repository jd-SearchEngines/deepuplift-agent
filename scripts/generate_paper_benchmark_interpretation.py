from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


METRIC_SPECS = {
    "pehe_mean": {"label": "PEHE", "lower_is_better": True},
    "ate_error_mean": {"label": "ATE error", "lower_is_better": True},
    "qini_mean": {"label": "QINI", "lower_is_better": False},
    "auuc_mean": {"label": "AUUC", "lower_is_better": False},
    "policy_top10_oracle_value_mean": {"label": "Policy oracle Top10", "lower_is_better": False},
    "oracle_top10_recall_mean": {"label": "Oracle Top10 recall", "lower_is_better": False},
}

MODEL_NOTES = {
    "TLearnerGBM": {
        "family": "Meta learner baseline",
        "strength": "两个 outcome model 分别拟合 treatment/control，工程简单、依赖少，适合作为强 baseline。",
        "watchout": "在 treatment/control 分布差异大、样本少或 uplift 信号弱时，两个模型误差会直接传导到 tau_hat。",
        "interview": "我把 T-Learner 当成 sanity baseline：如果复杂深度模型打不过它，需要先解释数据、loss 和训练稳定性。",
        "code": "deepuplift/core/sklearn_models.py; deepuplift/core/trainer.py",
    },
    "DRLearnerGBM": {
        "family": "Doubly robust / orthogonal learner",
        "strength": "用 outcome nuisance 和 propensity nuisance 构造更稳健的 pseudo outcome，适合有混杂和 propensity 风险的场景。",
        "watchout": "propensity overlap 差或 nuisance 欠拟合时会放大方差，需要看 ESS、clipping 和 cross-fitting 证据。",
        "interview": "DRLearner 的亮点不是分数一定最高，而是把因果估计的抗偏设计落到了训练证据和诊断里。",
        "code": "deepuplift/core/sklearn_models.py; deepuplift/core/diagnostics.py",
    },
    "CFRNet": {
        "family": "Representation balancing deep learner",
        "strength": "共享 representation 后用 IPM/MMD 约束 treatment/control 表征距离，适合 selection bias 明显的场景。",
        "watchout": "balance loss 权重过大可能牺牲 outcome fit；权重过小又退化成普通 TARNet。",
        "interview": "我会重点讲 CFRNet 如何把 causal assumption 转化成可训练的 representation balance loss。",
        "code": "deepuplift/models/CFRNet.py; deepuplift/models/deep_losses.py",
    },
    "DragonNet": {
        "family": "Targeted regularization deep learner",
        "strength": "同时学习 outcome、treatment propensity，并通过 targeted regularization 稳定 treatment effect。",
        "watchout": "tarreg 和 propensity head 需要稳定训练；小样本 smoke 只能证明路径，不能证明论文级收敛。",
        "interview": "DragonNet 是我展示从论文公式到工程 loss 组合的模型：outcome loss、propensity loss、targeted regularization 分层可解释。",
        "code": "deepuplift/models/DragonNet.py; deepuplift/models/deep_losses.py",
    },
    "EFIN": {
        "family": "Industrial uplift deep model",
        "strength": "面向工业 uplift 场景建模 explicit/implicit feature interaction，适合营销、广告、推荐排序语境。",
        "watchout": "对数据规模和特征工程更敏感；在小样本 known-CATE benchmark 上不一定优于树模型 baseline。",
        "interview": "EFIN 的价值在于工业化建模视角：不是只估 CATE，还要把 feature interaction 和线上排序可用性讲清楚。",
        "code": "deepuplift/models/EFIN.py",
    },
    "DESCN": {
        "family": "Entire-space uplift deep model",
        "strength": "适合 classification/full-funnel/entire-space uplift，把 treatment/control/effect 相关空间联合建模。",
        "watchout": "默认不跑 regression IHDP；task-aware skip 本身是平台成熟度的一部分。",
        "interview": "DESCN 我会作为工业 full-funnel uplift 架构讲，强调它不是所有数据集通吃，而是有明确 task boundary。",
        "code": "deepuplift/models/DESCN.py",
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _fmt(value: Any, digits: int = 5) -> str:
    number = _num(value)
    if number is None:
        return "NA"
    return f"{number:.{digits}g}"


def _mean(values: list[float]) -> float | None:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    return float(statistics.mean(clean)) if clean else None


def _best(rows: list[dict[str, Any]], metric: str, *, lower_is_better: bool) -> dict[str, Any] | None:
    candidates = [row for row in rows if _num(row.get(metric)) is not None]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: _num(row.get(metric)) or 0.0, reverse=not lower_is_better)[0]


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        groups.setdefault(value, []).append(row)
    return groups


def _dataset_winners(leaderboard: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for dataset_id, rows in sorted(_group_by(leaderboard, "dataset_id").items()):
        row: dict[str, Any] = {"dataset_id": dataset_id}
        for metric, spec in METRIC_SPECS.items():
            winner = _best(rows, metric, lower_is_better=bool(spec["lower_is_better"]))
            row[f"{metric}_winner"] = winner.get("model") if winner else None
            row[f"{metric}_value"] = winner.get(metric) if winner else None
        output.append(row)
    return output


def _metric_conflicts(winners: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts = []
    for row in winners:
        pehe = row.get("pehe_mean_winner")
        if not pehe:
            continue
        qini = row.get("qini_mean_winner")
        policy = row.get("policy_top10_oracle_value_mean_winner")
        auuc = row.get("auuc_mean_winner")
        notes = []
        if qini and qini != pehe:
            notes.append(f"lowest PEHE is {pehe}, highest QINI is {qini}")
        if auuc and auuc != pehe:
            notes.append(f"lowest PEHE is {pehe}, highest AUUC is {auuc}")
        if policy and policy != pehe:
            notes.append(f"lowest PEHE is {pehe}, highest policy value is {policy}")
        if notes:
            conflicts.append(
                {
                    "dataset_id": row.get("dataset_id"),
                    "conflict": "; ".join(notes),
                    "interview_reading": "Effect accuracy, ranking quality and business value are not interchangeable; show all three before claiming a model is better.",
                }
            )
    return conflicts


def _model_scorecards(
    leaderboard: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    winners: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    stability_by_key = {
        (row.get("dataset_id"), row.get("model")): row
        for row in stability_rows
    }
    winner_counts: dict[str, dict[str, int]] = {}
    for row in winners:
        for metric in METRIC_SPECS:
            model = row.get(f"{metric}_winner")
            if model:
                winner_counts.setdefault(str(model), {}).setdefault(metric, 0)
                winner_counts[str(model)][metric] += 1

    output = []
    for model, rows in sorted(_group_by(leaderboard, "model").items()):
        stability_counts: dict[str, int] = {}
        for row in rows:
            stability = stability_by_key.get((row.get("dataset_id"), row.get("model"))) or {}
            verdict = str(stability.get("verdict") or "unknown")
            stability_counts[verdict] = stability_counts.get(verdict, 0) + 1
        note = MODEL_NOTES.get(model, {})
        scorecard = {
            "model": model,
            "family": note.get("family", "Uplift model"),
            "datasets": len({row.get("dataset_id") for row in rows}),
            "ok_runs": int(sum(int(row.get("ok_runs") or 0) for row in rows)),
            "pehe_mean_over_datasets": _mean([_num(row.get("pehe_mean")) for row in rows if _num(row.get("pehe_mean")) is not None]),
            "ate_error_mean_over_datasets": _mean([_num(row.get("ate_error_mean")) for row in rows if _num(row.get("ate_error_mean")) is not None]),
            "qini_mean_over_datasets": _mean([_num(row.get("qini_mean")) for row in rows if _num(row.get("qini_mean")) is not None]),
            "auuc_mean_over_datasets": _mean([_num(row.get("auuc_mean")) for row in rows if _num(row.get("auuc_mean")) is not None]),
            "policy_top10_oracle_value_mean_over_datasets": _mean(
                [_num(row.get("policy_top10_oracle_value_mean")) for row in rows if _num(row.get("policy_top10_oracle_value_mean")) is not None]
            ),
            "winner_counts": winner_counts.get(model, {}),
            "stability_counts": stability_counts,
            "strength": note.get("strength", ""),
            "watchout": note.get("watchout", ""),
            "interview_line": note.get("interview", ""),
            "code_reference": note.get("code", ""),
        }
        pehe_wins = scorecard["winner_counts"].get("pehe_mean", 0)
        qini_wins = scorecard["winner_counts"].get("qini_mean", 0)
        if pehe_wins and qini_wins:
            scorecard["verdict"] = "accuracy_and_ranking_candidate"
        elif pehe_wins:
            scorecard["verdict"] = "effect_accuracy_candidate"
        elif qini_wins:
            scorecard["verdict"] = "ranking_candidate"
        elif stability_counts.get("needs_review"):
            scorecard["verdict"] = "needs_training_or_data_review"
        else:
            scorecard["verdict"] = "baseline_or_context_specific"
        output.append(scorecard)
    return sorted(
        output,
        key=lambda row: (
            -(row.get("winner_counts") or {}).get("pehe_mean", 0),
            _num(row.get("pehe_mean_over_datasets")) if _num(row.get("pehe_mean_over_datasets")) is not None else 1e18,
            -(_num(row.get("qini_mean_over_datasets")) or -1e18),
        ),
    )


def _stability_reading(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("stability_summary") or []
    counts: dict[str, int] = {}
    for row in rows:
        verdict = str(row.get("verdict") or "unknown")
        counts[verdict] = counts.get(verdict, 0) + 1
    seeds = payload.get("seeds") or []
    if len(seeds) >= 3:
        claim_level = "multi_seed_paper_level"
        message = "This run can support stronger paper-level comparison claims, assuming data generation and task boundaries are accepted."
    else:
        claim_level = "paper_lite_single_seed"
        message = "Use this as paper-level smoke evidence. For interview claims about stability, rerun preset benchmark-v3."
    return {"counts": counts, "claim_level": claim_level, "message": message}


def _interview_tracks(payload: dict[str, Any], scorecards: list[dict[str, Any]], conflicts: list[dict[str, Any]]) -> dict[str, str]:
    top_model = scorecards[0]["model"] if scorecards else "NA"
    tier = payload.get("benchmark_tier", "NA")
    seeds = payload.get("seeds") or []
    conflict_line = (
        f"I also show metric conflicts on {len(conflicts)} dataset(s), so I do not pretend PEHE, QINI and policy value are the same."
        if conflicts
        else "In this run, the main metrics are aligned enough for a cleaner model comparison."
    )
    return {
        "30s": (
            f"DeepUplift now has {tier} benchmark evidence: IHDP, ACIC-style and known-CATE synthetic datasets, "
            f"six model families, PEHE/ATE/QINI/AUUC/policy metrics, and evidence manifests. Current top effect-accuracy candidate: {top_model}."
        ),
        "5min": (
            "Open Evidence Dashboard -> Benchmark Dashboard. First explain PEHE/ATE error as known-CATE accuracy, "
            "then QINI/AUUC as ranking quality, then policy value as business decision value. "
            f"{conflict_line}"
        ),
        "15min": (
            "Walk through the model families: T-Learner as baseline, DRLearner for orthogonal robustness, CFRNet/DragonNet for representation and targeted regularization, "
            "EFIN/DESCN for industrial deep uplift. Then open run diff and failure attribution to show this is governed evidence, not a one-off notebook."
        ),
        "30min": (
            f"Start from the causal estimand, show the dataset/task boundary, review multi-seed design ({len(seeds)} seed(s) in latest run), "
            "compare metric conflicts, inspect evidence manifests, and close with offline-to-online rollout: OPE -> shadow -> small traffic A/B -> ramp-up."
        ),
    }


def _recommendations(payload: dict[str, Any], stability: dict[str, Any], conflicts: list[dict[str, Any]]) -> list[dict[str, str]]:
    recommendations = []
    if len(payload.get("seeds") or []) < 3:
        recommendations.append(
            {
                "priority": "P0",
                "action": "Run benchmark-v3",
                "why": "Current latest benchmark has fewer than 3 seeds, so stability claims should remain conservative.",
                "command": "python scripts/run_paper_benchmark_suite.py --preset benchmark-v3",
            }
        )
    if conflicts:
        recommendations.append(
            {
                "priority": "P0",
                "action": "Explain metric conflicts in review",
                "why": "A model can win PEHE but lose QINI/policy value; this is exactly where decision-platform thinking beats model-zoo thinking.",
                "command": "Open docs/DEEPUplift_BENCHMARK_V3_INTERPRETATION.md and the UI Benchmark Dashboard.",
            }
        )
    if (stability.get("counts") or {}).get("needs_review"):
        recommendations.append(
            {
                "priority": "P1",
                "action": "Inspect unstable rows",
                "why": "Wide intervals or QINI crossing zero indicate data split, loss weight, or sample-size sensitivity.",
                "command": "Filter Stability tab by needs_review and inspect per-run evidence manifests.",
            }
        )
    recommendations.append(
        {
            "priority": "P1",
            "action": "Attach model deconstruction evidence",
            "why": "Interviewers need to see not just who won, but why each loss/architecture is expected to win or fail.",
            "command": "Open docs/model_deconstruction/*.md next to the benchmark scorecards.",
        }
    )
    return recommendations


def build_interpretation(payload: dict[str, Any]) -> dict[str, Any]:
    leaderboard = payload.get("leaderboard") or []
    stability_rows = payload.get("stability_summary") or []
    winners = _dataset_winners(leaderboard)
    conflicts = _metric_conflicts(winners)
    scorecards = _model_scorecards(leaderboard, stability_rows, winners)
    stability = _stability_reading(payload)
    tracks = _interview_tracks(payload, scorecards, conflicts)
    return {
        "schema_version": 1,
        "status": "ok" if leaderboard else "empty",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "benchmark_status": payload.get("status"),
        "benchmark_tier": payload.get("benchmark_tier"),
        "preset": payload.get("preset"),
        "datasets": payload.get("datasets") or [],
        "models": payload.get("models") or [],
        "seeds": payload.get("seeds") or [],
        "rows_per_dataset": payload.get("rows_per_dataset"),
        "known_effect_bootstrap_samples": payload.get("known_effect_bootstrap_samples"),
        "run_counts": {
            "attempted": len(payload.get("runs") or []),
            "ok": sum(1 for row in payload.get("runs") or [] if row.get("status") == "ok"),
            "leaderboard": len(leaderboard),
        },
        "dataset_winners": winners,
        "metric_conflicts": conflicts,
        "model_scorecards": scorecards,
        "stability_reading": stability,
        "failure_attribution": payload.get("failure_attribution") or {},
        "interview_tracks": tracks,
        "recommendations": _recommendations(payload, stability, conflicts),
        "source_artifacts": {
            "benchmark_json": "reports/paper_benchmark_latest.json",
            "leaderboard_csv": "reports/paper_benchmark_leaderboard_latest.csv",
            "run_diff": "reports/paper_benchmark_run_diff_latest.json",
            "manifest": "reports/paper_benchmark_manifest_latest.json",
            "benchmark_doc": "docs/DEEPUplift_PAPER_LEVEL_BENCHMARK.md",
        },
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                value = _fmt(value)
            elif isinstance(value, dict):
                value = ", ".join(f"{k}:{v}" for k, v in sorted(value.items())) or "NA"
            values.append(str(value if value not in (None, "") else "NA"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_markdown(path: Path, interpretation: dict[str, Any]) -> None:
    tracks = interpretation.get("interview_tracks") or {}
    stability = interpretation.get("stability_reading") or {}
    failure = interpretation.get("failure_attribution") or {}
    lines = [
        "# DeepUplift Benchmark v3 Interpretation",
        "",
        f"Generated at: `{interpretation.get('generated_at')}`",
        "",
        "## Executive Summary",
        "",
        f"- Benchmark tier: `{interpretation.get('benchmark_tier')}`",
        f"- Preset: `{interpretation.get('preset')}`",
        f"- Datasets: {', '.join(interpretation.get('datasets') or [])}",
        f"- Models: {', '.join(interpretation.get('models') or [])}",
        f"- Seeds: {', '.join(str(seed) for seed in interpretation.get('seeds') or [])}",
        f"- Run counts: `{interpretation.get('run_counts')}`",
        f"- Claim level: `{stability.get('claim_level')}` - {stability.get('message')}",
        "",
        "## Interview Talk Track",
        "",
        f"- 30 seconds: {tracks.get('30s', 'NA')}",
        f"- 5 minutes: {tracks.get('5min', 'NA')}",
        f"- 15 minutes: {tracks.get('15min', 'NA')}",
        f"- 30 minutes: {tracks.get('30min', 'NA')}",
        "",
        "## Dataset Winners",
        "",
    ]
    winner_rows = []
    for row in interpretation.get("dataset_winners") or []:
        winner_rows.append(
            {
                "dataset_id": row.get("dataset_id"),
                "pehe": f"{row.get('pehe_mean_winner')} ({_fmt(row.get('pehe_mean_value'))})",
                "ate": f"{row.get('ate_error_mean_winner')} ({_fmt(row.get('ate_error_mean_value'))})",
                "qini": f"{row.get('qini_mean_winner')} ({_fmt(row.get('qini_mean_value'))})",
                "auuc": f"{row.get('auuc_mean_winner')} ({_fmt(row.get('auuc_mean_value'))})",
                "policy": f"{row.get('policy_top10_oracle_value_mean_winner')} ({_fmt(row.get('policy_top10_oracle_value_mean_value'))})",
            }
        )
    lines.extend(_markdown_table(winner_rows, [("dataset_id", "Dataset"), ("pehe", "PEHE winner"), ("ate", "ATE winner"), ("qini", "QINI winner"), ("auuc", "AUUC winner"), ("policy", "Policy winner")]))
    lines.extend(["", "## Model Scorecards", ""])
    lines.extend(
        _markdown_table(
            interpretation.get("model_scorecards") or [],
            [
                ("model", "Model"),
                ("family", "Family"),
                ("verdict", "Verdict"),
                ("datasets", "Datasets"),
                ("ok_runs", "OK runs"),
                ("pehe_mean_over_datasets", "Mean PEHE"),
                ("qini_mean_over_datasets", "Mean QINI"),
                ("winner_counts", "Wins"),
            ],
        )
    )
    lines.extend(["", "## Why Models Win Or Lose", ""])
    for row in interpretation.get("model_scorecards") or []:
        lines.extend(
            [
                f"### {row.get('model')}",
                "",
                f"- Family: {row.get('family')}",
                f"- Strength: {row.get('strength')}",
                f"- Watchout: {row.get('watchout')}",
                f"- Interview line: {row.get('interview_line')}",
                f"- Code reference: `{row.get('code_reference')}`",
                "",
            ]
        )
    lines.extend(["## Metric Conflicts", ""])
    conflicts = interpretation.get("metric_conflicts") or []
    if conflicts:
        lines.extend(_markdown_table(conflicts, [("dataset_id", "Dataset"), ("conflict", "Conflict"), ("interview_reading", "How to explain")]))
    else:
        lines.append("No major PEHE/QINI/policy winner conflict in the latest leaderboard.")
    lines.extend(
        [
            "",
            "## Stability And Failure Attribution",
            "",
            f"- Stability counts: `{stability.get('counts')}`",
            f"- Status counts: `{failure.get('status_counts')}`",
            f"- Skipped rows: `{failure.get('skipped')}`",
            f"- Errors: `{failure.get('errors')}`",
            "",
            "## Recommended Next Actions",
            "",
        ]
    )
    for item in interpretation.get("recommendations") or []:
        lines.append(f"- {item.get('priority')}: {item.get('action')} - {item.get('why')} Command/evidence: `{item.get('command')}`")
    lines.extend(
        [
            "",
            "## Evidence Links",
            "",
        ]
    )
    for name, artifact in (interpretation.get("source_artifacts") or {}).items():
        lines.append(f"- {name}: `{artifact}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate human-readable interpretation for paper-level benchmark evidence.")
    parser.add_argument("--benchmark", default="reports/paper_benchmark_latest.json")
    parser.add_argument("--json-output", default="reports/paper_benchmark_interpretation_latest.json")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_BENCHMARK_V3_INTERPRETATION.md")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    benchmark_path = ROOT / args.benchmark
    if not benchmark_path.exists():
        raise SystemExit(f"Benchmark artifact not found: {benchmark_path}")
    payload = _read_json(benchmark_path)
    interpretation = build_interpretation(payload)

    reports_dir = ROOT / args.reports_dir
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    timestamp_json = reports_dir / f"paper_benchmark_interpretation_{timestamp}.json"
    timestamp_md = reports_dir / f"paper_benchmark_interpretation_{timestamp}.md"
    latest_json = ROOT / args.json_output
    doc_output = ROOT / args.doc_output

    _write_json(timestamp_json, interpretation)
    _write_json(latest_json, interpretation)
    write_markdown(timestamp_md, interpretation)
    write_markdown(doc_output, interpretation)
    print(
        json.dumps(
            {
                "status": interpretation["status"],
                "json": str(latest_json.relative_to(ROOT)),
                "doc": str(doc_output.relative_to(ROOT)),
                "timestamped": str(timestamp_json.relative_to(ROOT)),
                "model_scorecards": len(interpretation.get("model_scorecards") or []),
                "metric_conflicts": len(interpretation.get("metric_conflicts") or []),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
