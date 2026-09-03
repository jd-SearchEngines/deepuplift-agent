from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.registry import MODEL_REGISTRY, missing_dependencies


BACKENDS = [
    {
        "backend": "EconML",
        "module": "econml",
        "model_prefixes": ["EconML"],
        "default_policy": "ready P0 robust causal backend",
        "enable_path": "already used by official CATE adapters when importable",
        "risk": "requires overlap/nuisance quality; inference APIs do not replace experiment design",
    },
    {
        "backend": "scikit-uplift",
        "module": "sklift",
        "model_prefixes": ["SkLift"],
        "default_policy": "ready lightweight uplift baseline",
        "enable_path": "already used by SkLift* registry adapters when importable",
        "risk": "binary-treatment baseline; complex observational bias still needs DR/R/DML checks",
    },
    {
        "backend": "Open Bandit Pipeline",
        "module": "obp",
        "model_prefixes": [],
        "default_policy": "guarded OPE / policy-learning plugin",
        "enable_path": "install obp only in an isolated OPE smoke, then map logs to action/reward/propensity schema",
        "risk": "requires logged bandit schema; not a direct y0/y1 uplift trainer",
    },
    {
        "backend": "RouteLLM",
        "module": "routellm",
        "model_prefixes": [],
        "default_policy": "optional LLM routing benchmark/source plugin",
        "enable_path": "install routellm in a serving/routing helper env, then import route logs into DeepUplift OPE",
        "risk": "server/runtime dependency is not needed for local uplift demo; needs judge calibration and propensity logs",
    },
    {
        "backend": "Vowpal Wabbit",
        "module": "vowpalwabbit",
        "model_prefixes": [],
        "default_policy": "optional online contextual-bandit layer",
        "enable_path": "export action/reward/propensity logs after fixed policy and OPE gates pass",
        "risk": "online learner can amplify biased rewards if exploration and guardrails are missing",
    },
    {
        "backend": "UTBoost",
        "module": "utboost",
        "model_prefixes": ["UTBoost"],
        "default_policy": "guarded optional plugin",
        "enable_path": "pip/conda install utboost, then run catalog audit and no-UI compare before using in live demo",
        "risk": "new native uplift-GBDT dependency; should be compared against stable LightGBM DR/R/DML baselines",
    },
    {
        "backend": "CatBoost",
        "module": "catboost",
        "model_prefixes": ["CatBoost"],
        "default_policy": "optional dependency",
        "enable_path": "install catboost; registry will mark CatBoost models ready when dependency is importable",
        "risk": "heavier categorical boosting backend; not required for default demo",
    },
    {
        "backend": "XGBoost",
        "module": "xgboost",
        "model_prefixes": ["XGBoost"],
        "default_policy": "guarded by DEEPUPLIFT_ENABLE_XGBOOST",
        "enable_path": "set DEEPUPLIFT_ENABLE_XGBOOST=1 only for isolated smoke tests",
        "risk": "local macOS runtime instability was observed; keep disabled in default demo",
    },
    {
        "backend": "CausalML",
        "module": "causalml",
        "model_prefixes": ["CausalML"],
        "default_policy": "Python 3.11 helper environment",
        "enable_path": "scripts/setup_causalml_py311.sh",
        "risk": "Python-version-sensitive package ecosystem; isolate from default Streamlit env",
    },
]


def matching_models(prefixes: list[str]) -> list[str]:
    return sorted(
        model
        for model in MODEL_REGISTRY
        if any(model.startswith(prefix) or prefix in model for prefix in prefixes)
    )


def main() -> None:
    rows = []
    for backend in BACKENDS:
        module_name = backend["module"]
        spec = importlib.util.find_spec(module_name)
        models = matching_models(backend["model_prefixes"])
        missing_models = {model: missing_dependencies(model) for model in models}
        ready_models = [model for model, deps in missing_models.items() if not deps]
        rows.append(
            {
                **backend,
                "importable": spec is not None,
                "python": sys.version.split()[0],
                "explicit_xgboost_flag": os.environ.get("DEEPUPLIFT_ENABLE_XGBOOST", ""),
                "registered_models": len(models),
                "ready_models": len(ready_models),
                "sample_models": models[:8],
                "missing_dependency_examples": {
                    model: deps for model, deps in list(missing_models.items())[:8] if deps
                },
                "recommendation": (
                    "ready for isolated smoke"
                    if ready_models
                    else "keep as source-gated / optional until dependency and smoke are validated"
                ),
            }
        )

    failures = []
    policy_notes = []
    for row in rows:
        if row["backend"] in {"UTBoost", "CatBoost", "CausalML"} and row["ready_models"]:
            policy_notes.append(
                f"{row['backend']} is importable in this environment; keep it behind evidence-gated smoke before live-demo use"
            )
        if row["backend"] == "XGBoost" and row["explicit_xgboost_flag"] and not row["ready_models"]:
            failures.append("XGBoost flag is set but models are still not ready")

    output = {
        "status": "fail" if failures else "ok",
        "rows": rows,
        "policy_notes": policy_notes,
        "failures": failures,
    }
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    path = reports_dir / "optional_backend_probe_latest.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": output["status"],
                "path": str(path.relative_to(ROOT)),
                "rows": len(rows),
                "policy_notes": policy_notes,
                "failures": failures,
            },
            ensure_ascii=False,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
