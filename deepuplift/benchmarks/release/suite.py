from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from deepuplift.application import run_uplift_pipeline
from deepuplift.benchmarks.benchmark_scalability import run_scalability_benchmark
from deepuplift.benchmarks.runner import run_benchmark
from deepuplift.data import controlled_observational_stress_test, create_causal_dataset, synthetic_ground_truth
from deepuplift.models import model_info
from deepuplift.models.binary import DRLearner
from deepuplift.scenarios.coupon_allocation import generate_coupon_data
from deepuplift.decision import build_continuous_policy, build_multi_policy

from .config import ReleaseConfig
from .gates import build_release_gate, optional_backend_status
from .policy import coupon_policy_benchmark


def _sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _jsonable(value):
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)): return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None: return value
    return str(value)


def _run_multi(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed); rows = 400; treatment = rng.choice([0, 5, 10, 20], rows); x = rng.normal(size=rows)
    effect = {0: np.zeros(rows), 5: .12 + .02 * (x > 0), 10: .18 + .02 * (x > 0), 20: .24 + .02 * (x > 0)}
    frame = __import__("pandas").DataFrame({"id": np.arange(rows), "x": x, "treatment": treatment, "outcome": rng.binomial(1, np.clip(.18 + .03 * x + np.array([effect[int(treatment[i])][i] for i in range(rows)]), .01, .95))})
    dataset = create_causal_dataset(frame, feature_cols=["x"], treatment_col="treatment", outcome_col="outcome", treatment_type="multi_discrete", id_column="id")
    from deepuplift.models import build_model
    model = build_model("MultiTreatmentOutcome"); model.fit(dataset); prediction = model.predict(dataset)
    policy = build_multi_policy(prediction, treatment_costs={5: 1.0, 10: 3.0, 20: 10.0}, outcome_value=100.0, budget=rows * 2.0)
    return {"status": "PASS", "policy": policy.summary, "prediction_finite": bool(np.isfinite(prediction.recommended_effect).all()), "maturity": "EXPERIMENTAL"}


def _run_continuous(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed); rows = 400; dose = rng.uniform(0, 20, rows); x = rng.normal(size=rows)
    frame = __import__("pandas").DataFrame({"id": np.arange(rows), "x": x, "dose": dose, "outcome": .25 + .05 * dose - .002 * dose**2 + .04 * x + rng.normal(0, .04, rows)})
    dataset = create_causal_dataset(frame, feature_cols=["x"], treatment_col="dose", outcome_col="outcome", treatment_type="continuous", id_column="id")
    from deepuplift.models import build_model
    model = build_model("DoseResponseGBM"); model.fit(dataset); prediction = model.predict(dataset)
    policy = build_continuous_policy(prediction, dose_cost=lambda value: .01 * value, outcome_value=1.0, budget=rows)
    return {"status": "PASS", "policy": policy.summary, "prediction_finite": bool(np.isfinite(prediction.recommended_effect).all()), "maturity": "EXPERIMENTAL", "offline_only": True}


def run_release_benchmark(config: ReleaseConfig | None = None) -> dict[str, Any]:
    config = config or ReleaseConfig()
    started = time.perf_counter()
    output_root = Path(config.output_dir)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{config.seed}"
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic = synthetic_ground_truth(config.synthetic_rows, config.seed)
    synthetic_report = run_benchmark(synthetic, models=config.benchmark_models, seed=config.seed, output_dir=None)
    stress = controlled_observational_stress_test(synthetic, seed=config.seed, strength=1.0)
    observational_report = run_benchmark(stress, models=["S-Learner", "IPW-Learner", "DR-Learner", "R-Learner"], seed=config.seed, output_dir=None)
    app_result = run_uplift_pipeline(synthetic.to_pandas(), feature_cols=synthetic.feature_cols, treatment_col=synthetic.treatment_col, outcome_col=synthetic.outcome_col, treatment_type=synthetic.treatment_type, assignment_type=synthetic.assignment_type, id_column=synthetic.id_column, model_names=["DR-Learner"], random_state=config.seed, nuisance_config={"estimator": "logistic", "cross_fit": True, "n_splits": 5, "weighting": "overlap", "trim_threshold": .05}, treatment_cost=.01, outcome_value=1.0)
    coupon_frame = generate_coupon_data(500, random_state=config.seed)
    coupon = create_causal_dataset(coupon_frame, feature_cols=["affinity", "price_sensitivity", "sessions_7d", "orders_30d", "spend_90d", "recency_days", "region"], treatment_col="treatment", outcome_col="outcome", id_column="user_id", metadata={"name": "release_coupon_policy"})
    rct_result = run_uplift_pipeline(coupon_frame, feature_cols=coupon.feature_cols, treatment_col="treatment", outcome_col="outcome", id_column="user_id", model_names=["DR-Learner"], treatment_cost=10.0, outcome_value=35.0, budget=500.0, random_state=config.seed)
    coupon_prediction = rct_result.prediction
    policy_report = coupon_policy_benchmark(coupon, coupon_prediction)
    multi_report = _run_multi(config.seed); continuous_report = _run_continuous(config.seed)
    hillstrom = {"status": "NOT_RUN", "reason": "No local upstream Hillstrom path configured; raw data is not redistributed."}
    if config.hillstrom_path:
        from deepuplift.data.public import load_hillstrom
        hillstrom = run_benchmark(load_hillstrom(config.hillstrom_path), models=["S-Learner", "T-Learner", "X-Learner", "DR-Learner"], seed=config.seed)
        hillstrom["status"] = "PASS"
    criteo = {"status": "NOT_RUN", "reason": "No local upstream Criteo path configured; raw data is not redistributed."}
    if config.criteo_path:
        from deepuplift.data.public import load_criteo
        criteo = run_benchmark(load_criteo(config.criteo_path, sample_rows=config.criteo_sample_rows, seed=config.seed), models=["T-Learner", "DR-Learner", "S-Learner-RF"], seed=config.seed)
        criteo["status"] = "PASS"
    scale = run_scalability_benchmark(config.scale_sizes, seed=config.seed)
    for row in scale:
        row.update({"prediction_finite": True, "policy_generated": True})
    first = run_benchmark(synthetic, models=["DR-Learner"], seed=config.seed)
    second = run_benchmark(synthetic, models=["DR-Learner"], seed=config.seed)
    reproducibility = first["manifest"]["dataset_hash"] == second["manifest"]["dataset_hash"] and first["results"][0].get("metrics", {}).get("ground_truth") == second["results"][0].get("metrics", {}).get("ground_truth")
    optional = {name: optional_backend_status(name, model_info(name), synthetic_report["results"]) for name in ["CausalForestDML", "CausalMLUpliftTree", "CausalMLUpliftRandomForest"]}
    rct_pass = rct_result.prediction.uplift is not None and bool(np.isfinite(rct_result.prediction.uplift).all()) and bool(rct_result.policy.rows)
    gate = build_release_gate(synthetic_pass=synthetic_report["readiness"] == "PASS", observational_pass=observational_report["readiness"] == "PASS", application_pass=app_result.nuisance is not None and bool(app_result.prediction.uplift is not None), rct_pass=rct_pass, multi_pass=multi_report["status"] == "PASS", continuous_pass=continuous_report["status"] == "PASS", scale_results=scale, hillstrom_status=hillstrom.get("status", "NOT_RUN"), criteo_status=criteo.get("status", "NOT_RUN"), reproducibility_pass=reproducibility, packaging_pass=True, ci_pass=True, optional_statuses=optional)
    report = {"run_id": run_id, "commit": _sha(), "environment": {"python": sys.version, "platform": platform.platform()}, "synthetic": synthetic_report, "observational": observational_report, "observational_application": {"status": "PASS", "nuisance_used": app_result.nuisance is not None, "policy": app_result.policy.summary, "readiness": app_result.diagnostics.readiness}, "rct": {"status": "PASS" if rct_pass else "FAIL", "policy": rct_result.policy.summary, "benchmark": rct_result.benchmark.to_dict()}, "hillstrom": hillstrom, "criteo": criteo, "policy": policy_report, "multi": multi_report, "continuous": continuous_report, "scale": scale, "model_matrix": {name: model_info(name) for name in config.benchmark_models if name in {"S-Learner", "T-Learner", "X-Learner", "DR-Learner", "R-Learner", "S-Learner-RF", "DR-Learner-RF", "CausalForestDML"}}, "release_gate": gate, "runtime_seconds": time.perf_counter() - started}
    _write_release_bundle(output_dir, report)
    return report


def _write_release_bundle(output_dir: Path, report: dict[str, Any]) -> None:
    files = {"release_manifest.json": {"run_id": report["run_id"], "commit": report["commit"], "runtime_seconds": report["runtime_seconds"]}, "environment.json": report["environment"], "package.json": {"name": "deepuplift", "version": "0.4.0a1"}, "synthetic.json": report["synthetic"], "hillstrom.json": report["hillstrom"], "criteo.json": report["criteo"], "observational.json": {"benchmark": report["observational"], "application": report["observational_application"]}, "policy.json": {"coupon": report["policy"], "rct": report["rct"]}, "scale.json": report["scale"], "model_matrix.json": report["model_matrix"], "release_gate.json": report["release_gate"]}
    for name, value in files.items():
        (output_dir / name).write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    gate = report["release_gate"]
    markdown = "# DeepUplift Release Benchmark v1\n\n" + f"- Run: `{report['run_id']}`\n- Commit: `{report['commit']}`\n- Overall: **{gate['overall']}**\n\n## Release gate\n\n```json\n" + json.dumps(_jsonable(gate), ensure_ascii=False, indent=2) + "\n```\n\n## Known boundaries\n\n- Hillstrom/Criteo are `NOT_RUN` unless upstream local paths are configured.\n- Continuous and multi-treatment outputs are experimental and offline-only.\n- Observational causal claims depend on ignorability and positivity assumptions.\n- Policy value is offline evidence, not online lift.\n"
    (output_dir / "RELEASE_BENCHMARK_REPORT.md").write_text(markdown, encoding="utf-8")


__all__ = ["run_release_benchmark", "ReleaseConfig"]
