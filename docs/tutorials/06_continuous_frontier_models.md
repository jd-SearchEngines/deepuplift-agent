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

Use `load_ihdp()`, `load_news()`, or `load_tcga()` only after obtaining the relevant upstream data and converting it to a local CSV with `dose`, `outcome`, and feature columns. Loaders do not download or redistribute raw files. The upstream continuous benchmark dataset licenses have not been established by this release.

When the official GIKS release files are already available locally, the benchmark runner can use the fixed IHDP or NEWS split and stored response curves directly:

```bash
python scripts/run_continuous_benchmark.py --dataset ihdp \
  --data-path /path/to/GIKS_release/dataset/ihdp/tr_h_1.0_te_l_0.0_h1.0 \
  --split-id 0 --model DoseResponseGBM --model VCNet --model GIKS-VCNet
```

The report records the source path and flags the dataset license as unverified. TCGA is marked `DATA_NOT_AVAILABLE` when no local data path is supplied.
