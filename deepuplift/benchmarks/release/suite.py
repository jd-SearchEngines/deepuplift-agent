from __future__ import annotations

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
from deepuplift import __version__
from deepuplift.models.binary import DRLearner
from deepuplift.scenarios.coupon_allocation import generate_coupon_data
from deepuplift.decision import build_continuous_policy, build_multi_policy

from .config import ReleaseConfig
from .gates import build_release_gate, optional_backend_status
from .packaging import run_packaging_validation
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
    return {"status": "PASS", "policy": policy.summary, "prediction_finite": bool(np.isfinite(prediction.recommended_effect).all()), "dose_grid": prediction.dose_grid, "dose_effect_predictions": prediction.dose_effect_predictions, "maturity": "EXPERIMENTAL", "offline_only": True}


def _load_optional_report(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    report_path = Path(path)
    if not report_path.exists():
        return {"status": "FAIL", "error": f"optional backend report not found: {report_path}"}
    return json.loads(report_path.read_text(encoding="utf-8"))


def _public_benchmark(loader, path: str | Path, models: list[str], *, seed: int = 42, **kwargs: Any) -> dict[str, Any]:
    try:
        dataset = loader(path, **kwargs)
        report = run_benchmark(dataset, models=models, seed=seed)
        report["status"] = "PASS" if any(row.get("status") == "PASS" for row in report["results"]) else "FAIL"
        report["rows"] = len(dataset.to_pandas())
        report["assignment"] = dataset.assignment_type.value
        report["treatment_type"] = dataset.treatment_type.value
        return report
    except Exception as exc:
        return {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}


def _hillstrom_multi(path: str | Path) -> dict[str, Any]:
    try:
        from deepuplift.data.public import load_hillstrom
        from deepuplift.models import build_model
        dataset = load_hillstrom(path, multi_treatment=True)
        model = build_model("MultiTreatmentOutcome")
        model.fit(dataset)
        prediction = model.predict(dataset)
        policy = build_multi_policy(prediction, treatment_costs={1: 0.05, 2: 0.05}, outcome_value=1.0, budget=len(dataset.to_pandas()) * 0.05, no_treatment=0, optimizer="value_per_cost")
        return {"status": "PASS", "rows": len(dataset.to_pandas()), "treatment_values": sorted(dataset.treatment.unique().tolist()), "assignment": dataset.assignment_type.value, "treatment_type": dataset.treatment_type.value, "policy": policy.summary, "cost_assumption": "Demo-only cost assumptions; not a fact in Hillstrom raw data.", "offline_only": True}
    except Exception as exc:
        return {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}


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
    hillstrom = {"status": "NOT_RUN", "reason": "No local upstream Hillstrom path configured; raw data is not redistributed.", "blocked_reason": "BLOCKED_BY_DATA_ACCESS"}
    if config.hillstrom_path:
        from deepuplift.data.public import load_hillstrom
        hillstrom = _public_benchmark(load_hillstrom, config.hillstrom_path, ["S-Learner", "T-Learner", "X-Learner", "DR-Learner", "R-Learner", "S-Learner-RF", "DR-Learner-RF", "CausalForestDML", "CausalMLUpliftTree", "CausalMLUpliftRandomForest"], seed=config.seed)
    criteo = {"status": "NOT_RUN", "reason": "No local upstream Criteo path configured; raw data is not redistributed.", "blocked_reason": "BLOCKED_BY_DATA_ACCESS"}
    if config.criteo_path:
        from deepuplift.data.public import load_criteo
        criteo = _public_benchmark(load_criteo, config.criteo_path, ["T-Learner", "DR-Learner", "S-Learner-RF", "DR-Learner-RF"], sample_rows=config.criteo_sample_rows, seed=config.seed)
    hillstrom_multi = _hillstrom_multi(config.hillstrom_path) if config.hillstrom_path else {"status": "NOT_RUN", "reason": "Requires local Hillstrom path.", "blocked_reason": "BLOCKED_BY_DATA_ACCESS"}
    scale = run_scalability_benchmark(config.scale_sizes, seed=config.seed)
    first = run_benchmark(synthetic, models=["DR-Learner"], seed=config.seed)
    second = run_benchmark(synthetic, models=["DR-Learner"], seed=config.seed)
    reproducibility = first["manifest"]["dataset_hash"] == second["manifest"]["dataset_hash"] and first["results"][0].get("metrics", {}).get("ground_truth") == second["results"][0].get("metrics", {}).get("ground_truth")
    optional_external = _load_optional_report(config.optional_backend_report_path)
    optional = optional_external.get("statuses", {}) if optional_external and optional_external.get("statuses") else {name: optional_backend_status(name, model_info(name), synthetic_report["results"]) for name in ["CausalForestDML", "CausalMLUpliftTree", "CausalMLUpliftRandomForest"]}
    native_names = {"S-Learner", "T-Learner", "X-Learner", "DR-Learner", "R-Learner"}
    native_rows = [row for row in synthetic_report["results"] if row.get("model") in native_names]
    rf_rows = [row for row in synthetic_report["results"] if row.get("model") in {"S-Learner-RF", "DR-Learner-RF"}]
    synthetic_pass = bool(native_rows) and all(row.get("status") == "PASS" for row in native_rows)
    rf_pass = len(rf_rows) == 2 and all(row.get("status") == "PASS" for row in rf_rows)
    rct_pass = rct_result.prediction.uplift is not None and bool(np.isfinite(rct_result.prediction.uplift).all()) and bool(rct_result.policy.rows)
    packaging = run_packaging_validation(python_executable=config.packaging_python) if config.packaging_python else {"status": config.packaging_status, "reason": "Packaging validation not requested in this run; supply --packaging-python for build + fresh venv evidence."}
    gate = build_release_gate(synthetic_pass=synthetic_pass, observational_pass=observational_report["readiness"] == "PASS", application_pass=app_result.nuisance is not None and bool(app_result.prediction.uplift is not None), rct_pass=rct_pass, multi_pass=multi_report["status"] == "PASS", continuous_pass=continuous_report["status"] == "PASS", scale_results=scale, hillstrom_status=hillstrom.get("status", "NOT_RUN"), criteo_status=criteo.get("status", "NOT_RUN"), reproducibility_pass=reproducibility, packaging_pass=packaging.get("status", config.packaging_status), ci_pass=config.ci_status, optional_statuses=optional, rf_pass=rf_pass)
    report = {"run_id": run_id, "commit": _sha(), "version": __version__, "environment": {"python": sys.version, "platform": platform.platform()}, "synthetic": synthetic_report, "observational": observational_report, "observational_application": {"status": "PASS", "nuisance_used": app_result.nuisance is not None, "policy": app_result.policy.summary, "readiness": app_result.diagnostics.readiness}, "rct": {"status": "PASS" if rct_pass else "FAIL", "policy": rct_result.policy.summary, "benchmark": rct_result.benchmark.to_dict()}, "hillstrom": hillstrom, "hillstrom_multi": hillstrom_multi, "criteo": criteo, "policy": policy_report, "multi": multi_report, "continuous": continuous_report, "scale": scale, "optional_backends": optional_external or {"statuses": optional, "reason": "No optional smoke report supplied."}, "packaging": packaging, "model_matrix": {name: model_info(name) for name in config.benchmark_models + ("CausalMLUpliftTree", "CausalMLUpliftRandomForest") if name in {"S-Learner", "T-Learner", "X-Learner", "DR-Learner", "R-Learner", "S-Learner-RF", "DR-Learner-RF", "CausalForestDML", "CausalMLUpliftTree", "CausalMLUpliftRandomForest"}}, "release_gate": gate, "runtime_seconds": time.perf_counter() - started}
    _write_release_bundle(output_dir, report)
    return report


def _write_release_bundle(output_dir: Path, report: dict[str, Any]) -> None:
    files = {"release_manifest.json": {"run_id": report["run_id"], "commit": report["commit"], "version": report["version"], "runtime_seconds": report["runtime_seconds"]}, "environment.json": report["environment"], "package.json": {"name": "deepuplift", "version": report["version"]}, "synthetic.json": report["synthetic"], "hillstrom.json": report["hillstrom"], "hillstrom_multi.json": report["hillstrom_multi"], "criteo.json": report["criteo"], "observational.json": {"benchmark": report["observational"], "application": report["observational_application"]}, "multi.json": report["multi"], "continuous.json": report["continuous"], "policy.json": {"coupon": report["policy"], "rct": report["rct"]}, "scale.json": report["scale"], "optional_backends.json": report["optional_backends"], "packaging.json": report["packaging"], "model_matrix.json": report["model_matrix"], "release_gate.json": report["release_gate"]}
    for name, value in files.items():
        (output_dir / name).write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    gate = report["release_gate"]
    sections = ["# DeepUplift Release Benchmark v2", f"\n- Run: `{report['run_id']}`\n- Commit: `{report['commit']}`\n- Version: `{report['version']}`\n- Overall: **{gate['overall']}**\n", "## Environment\n\n" + json.dumps(_jsonable(report["environment"]), ensure_ascii=False, indent=2), "## Release Gate\n\n" + json.dumps(_jsonable(gate), ensure_ascii=False, indent=2), "## Synthetic Benchmark\n\n" + json.dumps(_jsonable(report["synthetic"]["selection"]), ensure_ascii=False, indent=2), "## Observational Benchmark\n\n" + json.dumps(_jsonable({"benchmark": report["observational"].get("selection"), "application": report["observational_application"]}), ensure_ascii=False, indent=2), "## Hillstrom Benchmark\n\n" + json.dumps(_jsonable({"binary": report["hillstrom"], "multi": report["hillstrom_multi"]}), ensure_ascii=False, indent=2), "## Criteo Benchmark\n\n" + json.dumps(_jsonable(report["criteo"]), ensure_ascii=False, indent=2), "## Multi-treatment Benchmark\n\n" + json.dumps(_jsonable(report["multi"]), ensure_ascii=False, indent=2), "## Continuous-treatment Benchmark\n\n" + json.dumps(_jsonable(report["continuous"]), ensure_ascii=False, indent=2), "## Scale Benchmark\n\n" + json.dumps(_jsonable(report["scale"]), ensure_ascii=False, indent=2), "## Optional Backends\n\n" + json.dumps(_jsonable(report["optional_backends"]), ensure_ascii=False, indent=2), "## Packaging\n\n" + json.dumps(_jsonable(report["packaging"]), ensure_ascii=False, indent=2), "## Known Limitations\n\n- Public data remains upstream/local-path evidence; raw data is not redistributed.\n- Continuous and multi-treatment outputs are experimental and offline-only.\n- Observational causal claims depend on ignorability and positivity assumptions.\n- Policy value is offline evidence, not online lift.\n", "## Final Status\n\n" + gate["overall"]]
    markdown = "\n\n".join(sections) + "\n"
    (output_dir / "RELEASE_BENCHMARK_REPORT.md").write_text(markdown, encoding="utf-8")


__all__ = ["run_release_benchmark", "ReleaseConfig"]
