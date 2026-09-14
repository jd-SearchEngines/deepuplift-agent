# Continuous dose-response and economic dose selection

This example compares a tree baseline, VCNet's varying-coefficient response, and GIKS with VCNet as its base. The data generator makes dose assignment depend on pre-treatment covariates, then evaluates predictions against held-out potential outcome curves.

Install the optional neural methods and run a small benchmark:

```bash
pip install -e '.[continuous,giks]'
python scripts/run_continuous_benchmark.py \
  --model DoseResponseGBM --model VCNet --model GIKS-VCNet \
  --rows 240 --seed 42 --budget 25 --outcome-value 1.0 \
  --epochs 60 --factual-epochs 50 --giks-epochs 25
```

The report and JSON artifacts are saved below `runs/continuous/`. The default benchmark is offline and does not load CCPFN. Each model result contains a three-user curve preview with outcome/effect values by dose, maximum-effect dose, and the economic policy's selected dose and net value. Metrics include held-out MISE, optimal-effect-dose error, and economic policy regret.

The synthetic setting represents a subsidy from 0 to 20 RMB. Its true dose-response function is concave with covariate-varying slope and curvature. The demonstration cost is `0.015 * dose + 0.0025 * dose**2`; the unit values and cost are assumptions for this example only.

The same flow can be called in Python:

```python
from deepuplift.benchmarks.continuous import run_continuous_suite, synthetic_dose_response

data = synthetic_dose_response(rows=240, seed=42, outcome_value=1.0)
result = run_continuous_suite(
    data=data,
    model_names=["DoseResponseGBM", "VCNet", "GIKS-VCNet"],
    budget=25.0,
    model_options={
        "VCNet": {"epochs": 60},
        "GIKS-VCNet": {"factual_epochs": 50, "giks_epochs": 25},
    },
)
print(result["report"])
for item in result["models"]:
    if item["status"] == "PASS":
        print(item["model"], item["curve_preview"][0])
```

The maximum-effect dose is a property of the estimated effect curve. The policy instead chooses the dose with the highest estimated net value, `effect * outcome_value - cost(dose)`, and applies the optional budget constraint after ranking users. The benchmark reports the two doses separately because they need not match.

## Treatment support and safe decisions

Inspect the range that the model actually observed before interpreting a curve. With training data from 5 to 20 RMB, the default model curve and policy candidates stay between 5 and 20 RMB. A request for 0 to 30 RMB is rejected unless extrapolation is explicitly enabled; enabling it marks every unsupported dose and attaches a warning. Since 0 is not observed in the 5 to 20 example, the model also cannot claim a supported no-treatment baseline, and the default policy returns no treatment with reason `unsupported_no_treatment_baseline`.

```python
from deepuplift.data import diagnose_continuous_treatment, local_treatment_support
from deepuplift.decision import build_continuous_policy
from deepuplift.models.continuous import VCNet

# `train` has observed doses from 5 to 20; `test` contains prediction features.
diagnostics = diagnose_continuous_treatment(train, segment_cols=[train.feature_cols[0]])
model = VCNet(epochs=60).fit(train)
prediction = model.predict(test)
prediction.metadata["local_treatment_support"] = local_treatment_support(
    train, test, dose_grid=prediction.dose_grid
)
support = prediction.metadata["dose_support"]
print(support["observed_min"], support["observed_max"], support["warning"] if support.get("warning") else None)
print("maximum-effect dose:", prediction.recommended_treatment[0])
policy = build_continuous_policy(prediction, dose_cost=lambda dose: 0.015 * dose + 0.0025 * dose**2)
print("maximum-net-value dose:", policy.rows[0]["recommended_treatment"])
print("decision reason:", policy.rows[0]["reason"])
```

“效果最大”不等于“经济收益最大”：the model's `recommended_treatment` is only a maximum-effect reference, while the policy maximizes effect times outcome value minus dose cost. A model can produce a numerical prediction at any dose after explicit extrapolation, but that does not mean the training data supports a causal interpretation there. Local kNN coverage is a practical empirical diagnostic, not an identification guarantee.

Use `load_ihdp()`, `load_news()`, or `load_tcga()` only after obtaining the relevant upstream data and converting it to a local CSV with `dose`, `outcome`, and feature columns. Loaders do not download or redistribute raw files. The upstream continuous benchmark dataset licenses have not been established by this release.

When the official GIKS release files are already available locally, the benchmark runner can use the fixed IHDP or NEWS split and stored response curves directly:

```bash
python scripts/run_continuous_benchmark.py --dataset ihdp \
  --data-path /path/to/GIKS_release/dataset/ihdp/tr_h_1.0_te_l_0.0_h1.0 \
  --split-id 0 --model DoseResponseGBM --model VCNet --model GIKS-VCNet
```

The report records the source path and flags the dataset license as unverified. TCGA is marked `DATA_NOT_AVAILABLE` when no local data path is supplied.
