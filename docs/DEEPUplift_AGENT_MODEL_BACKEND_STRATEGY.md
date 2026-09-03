# DeepUplift Agent Model Backend Strategy

The model strategy is to maximize catalog coverage while keeping the runnable demo stable. The system separates three ideas:

- **Registered**: the model exists in the catalog, has a model card, and can be explained.
- **Ready**: the model is runnable in the current local environment.
- **Guarded / optional**: the model is valuable to expose, but requires an explicit dependency, runtime flag, or Python-version-specific environment.

## Current Verified Coverage

| Backend | Role | Current Status |
| --- | --- | --- |
| DeepUplift | Native neural and benchmark models | Ready |
| LightGBM | Industrial table-data baselines and meta-learners | Ready |
| EconML | Orthogonal / DR / DML / causal forest style learners | Ready |
| scikit-uplift | Official uplift tree/forest/meta-learner adapters | Ready |
| XGBoost | Industrial boosting family | Guarded by runtime flag |
| CatBoost | Categorical-feature industrial boosting family | Optional dependency |
| CausalML | Uplift tree/forest ecosystem | Optional Python 3.11 helper environment |
| UTBoost | Uplift-oriented GBDT plugin | Guarded optional dependency |

The current environment also writes a machine-readable backend probe:

```bash
python3 scripts/probe_optional_uplift_backends.py
```

The probe outputs `reports/optional_backend_probe_latest.json` and records importability, registered model count, ready model count, enable path, risk and recommendation for UTBoost, CatBoost, XGBoost and CausalML. This keeps the model catalog honest: optional coverage is visible, but live-demo readiness is still evidence-gated.

UTBoost has an additional guarded adapter smoke:

```bash
python3 scripts/smoke_utboost_guarded_adapter.py
```

When `utboost` is not installed, the smoke passes with `status=guarded_skip` and records the missing dependency. If `utboost` is installed later, the same script runs a tiny deterministic training smoke against `UTBoostGBM` and writes the run id and metrics to `reports/utboost_guarded_adapter_smoke_latest.json`.

The UTBoost model card also exposes three governance fields in the `Models` page:

- `integration_gate`: backend probe, guarded adapter smoke, catalog audit, and no-UI compare commands.
- `source_evidence`: official repo, local registry path, adapter path, smoke script, and Evidence page location.
- `promotion_rule`: the exact condition for moving from guarded plugin to demo-ready backend.

The same rows are summarized in `Evidence -> Model Card Governance`, so reviewers can audit all guarded/optional promotion rules without opening each model card one by one.

## Why Guard Some Models

Some uplift libraries are excellent but heavy:

- They may pin incompatible Python versions.
- Native extensions can behave differently on macOS/Linux.
- Training smoke can hang or become unstable on one machine.
- Installing all optional stacks can make the demo fragile and slow.

The workbench therefore shows these models in the catalog, explains their assumptions, and marks why they are not default-ready. This is an engineering choice: reliable decision workflow first, optional backend expansion second.

## Guarded Backend Probe Policy

| Backend | Probe Result To Read | Default Action |
| --- | --- | --- |
| UTBoost | `importable=false`, `registered_models=1`, `ready_models=0` unless `utboost` is installed | Keep `UTBoostGBM` visible as P1 guarded plugin; only enable after dependency install, catalog audit and no-UI compare. |
| CatBoost | Optional dependency with CatBoost model family registered | Keep optional; install only for categorical-heavy industrial tables and validate with compare smoke. |
| XGBoost | May be importable, but guarded by `DEEPUPLIFT_ENABLE_XGBOOST` | Keep disabled in default demo; enable only in isolated smoke because local runtime instability was observed. |
| CausalML | Python-version-specific, expected to use Python 3.11 helper environment | Use `scripts/setup_causalml_py311.sh`; do not force into the main Python 3.10 Streamlit env. |

Interview phrasing:

> I do not claim every backend is live-ready by default. I expose UTBoost/CatBoost/XGBoost/CausalML through model cards, source gates and a backend probe, then only promote a backend to ready after dependency import, model audit, no-UI compare and screenshot/evidence validation pass.

## Adapter Contract

Every runnable model must normalize to the same contract:

```text
fit(X_train, y_train, t_train, config)
predict_uplift(X_test) -> y0_pred, y1_pred, uplift_score
metadata -> source, family, assumptions, dependencies, readiness
```

This protects evaluation from backend-specific output shapes. For example, if one model returns treatment-first scores and another returns control-first scores, the adapter absorbs that difference before QINI, AUUC, Top-K, calibration, and policy value are computed.

## Expansion Priority

| Priority | Backend / Family | Why |
| --- | --- | --- |
| 1 | R learner / DR learner / Orthogonal learners | Strong default for observational data and confounding risk |
| 2 | Causal forest / GRF-style learners | Useful for heterogeneous effects and stable segment discovery |
| 3 | Uplift tree / uplift random forest | Easy to explain to business stakeholders |
| 4 | LightGBM / XGBoost / CatBoost meta-learners | Industrial table-data performance and scale |
| 5 | Multi-treatment and dose-response learners | Needed for real campaign portfolios and incentive levels |
| 6 | Sequential / long-term treatment learners | Needed when treatment timing and delayed effects matter |

## Interview Talk Track

I deliberately separated model catalog coverage from local readiness. The model list can reflect industry and open-source uplift methods, while the product only enables models that pass local dependency and smoke-test gates. That is why the current system can discuss 103 models but only marks 73 as ready: it is transparent about capability, risk, and reproducibility.
