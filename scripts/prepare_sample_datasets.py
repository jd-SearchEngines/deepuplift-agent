from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "examples" / "datasets"
RAW_DIR = DATASET_DIR / "raw"
CRITEO_SOURCE = ROOT / "deepuplift" / "dataset" / "criteo-uplift-v2.1-unbiased-sample50w.csv"

URLS = {
    "hillstrom": "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
    "doubleml_uplift": "https://docs.doubleml.org/tutorial/stable/datasets/data/uplift_data.csv",
    "lalonde": "https://vincentarelbundock.github.io/Rdatasets/csv/MatchIt/lalonde.csv",
    "ecdat_treatment": "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Treatment.csv",
    "nhefs_complete": "https://csvbase.com/rmirror/nhefs-complete.csv",
    "ihdp_npci_1": "https://raw.githubusercontent.com/rguo12/CIKM18-LCVA/master/datasets/IHDP/csv/ihdp_npci_1.csv",
    "smoke_ban": "https://vincentarelbundock.github.io/Rdatasets/csv/AER/SmokeBan.csv",
    "thornton_hiv": "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/thornton_hiv.csv",
}

FAILED_SOURCES: set[str] = set()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def download(url: str, filename: str, force: bool = False, max_time: int = 120) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output = RAW_DIR / filename
    if output.exists() and not force:
        return output
    partial = output.with_suffix(output.suffix + ".part")
    if partial.exists():
        partial.unlink()
    command = [
        "curl",
        "-L",
        "--fail",
        "--connect-timeout",
        "15",
        "--max-time",
        str(max_time),
        "-A",
        "DeepUplift-Agent/1.0",
        "-o",
        str(partial),
        url,
    ]
    subprocess.run(command, check=True)
    partial.replace(output)
    return output


def download_source(source: str, filename: str, max_time: int = 120) -> Path:
    if source in FAILED_SOURCES:
        raise RuntimeError(f"Previous download attempt failed for source: {source}")
    try:
        return download(URLS[source], filename, max_time=max_time)
    except Exception:
        FAILED_SOURCES.add(source)
        raise


def _write_dataset(df: pd.DataFrame, filename: str) -> Path:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    output = DATASET_DIR / filename
    df.to_csv(output, index=False)
    return output


def write_criteo_sample() -> dict:
    if not CRITEO_SOURCE.exists():
        raise FileNotFoundError(f"Criteo source not found: {CRITEO_SOURCE}")

    df = pd.read_csv(CRITEO_SOURCE, nrows=10000)
    output = _write_dataset(df, "criteo_visit_10k.csv")
    feature_cols = [f"f{i}" for i in range(12)]
    return {
        "id": "criteo_visit_10k",
        "name": "Criteo Uplift Visit 10k",
        "kind": "public_sample",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "A 10k-row local sample from the public Criteo Uplift Prediction dataset. Outcome defaults to visit.",
        "source_url": "https://ailab.criteo.com/criteo-uplift-prediction-dataset/",
        "treatment_col": "treatment",
        "outcome_col": "visit",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["TLearnerGBM", "DRLearnerGBM", "RLearnerGBM", "TarNet", "DragonNet", "DESCN"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 10000},
    }


def write_hillstrom_email_visit() -> dict:
    raw = download_source("hillstrom", "hillstrom_email_analytics.csv", max_time=60)
    df = pd.read_csv(raw)
    feature_cols = ["recency", "history_segment", "history", "mens", "womens", "zip_code", "newbie", "channel"]
    output_df = df[feature_cols + ["segment", "visit"]].copy()
    output_df["treatment"] = (output_df["segment"] != "No E-Mail").astype(int)
    output_df["outcome"] = output_df["visit"].astype(int)
    output_df = output_df[feature_cols + ["treatment", "outcome"]]
    output = _write_dataset(output_df, "hillstrom_email_visit_64k.csv")
    return {
        "id": "hillstrom_email_visit_64k",
        "name": "Hillstrom Email Visit 64k",
        "kind": "public_randomized_marketing",
        "path": _relative(output),
        "rows": int(len(output_df)),
        "description": "MineThatData/Hillstrom randomized email campaign. Treatment is any email vs no email; outcome is visit.",
        "source_url": "https://tensorflow.google.cn/datasets/catalog/hillstrom",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["SLearnerGBM", "TLearnerGBM", "XLearnerGBM", "DRLearnerGBM", "RLearnerGBM", "CausalForest", "EFIN"],
        "quick_train": {"epochs": 2, "batch_size": 512, "learning_rate": 0.001, "max_rows": 20000},
    }


def write_hillstrom_mens_visit() -> dict:
    raw = download_source("hillstrom", "hillstrom_email_analytics.csv", max_time=60)
    df = pd.read_csv(raw)
    df = df[df["segment"].isin(["Mens E-Mail", "No E-Mail"])].copy()
    feature_cols = ["recency", "history_segment", "history", "mens", "womens", "zip_code", "newbie", "channel"]
    output_df = df[feature_cols + ["segment", "visit"]].copy()
    output_df["treatment"] = (output_df["segment"] == "Mens E-Mail").astype(int)
    output_df["outcome"] = output_df["visit"].astype(int)
    output_df = output_df[feature_cols + ["treatment", "outcome"]]
    output = _write_dataset(output_df, "hillstrom_mens_visit_42k.csv")
    return {
        "id": "hillstrom_mens_visit_42k",
        "name": "Hillstrom Mens Email Visit 42k",
        "kind": "public_randomized_marketing",
        "path": _relative(output),
        "rows": int(len(output_df)),
        "description": "Hillstrom randomized subset: Mens email vs no email; outcome is visit.",
        "source_url": "https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["TLearnerGBM", "DRLearnerGBM", "RLearnerGBM", "OrthogonalDMLGBM", "CausalForest", "DragonNet"],
        "quick_train": {"epochs": 2, "batch_size": 512, "learning_rate": 0.001, "max_rows": 20000},
    }


def write_hillstrom_womens_conversion() -> dict:
    raw = download_source("hillstrom", "hillstrom_email_analytics.csv", max_time=60)
    df = pd.read_csv(raw)
    df = df[df["segment"].isin(["Womens E-Mail", "No E-Mail"])].copy()
    feature_cols = ["recency", "history_segment", "history", "mens", "womens", "zip_code", "newbie", "channel"]
    output_df = df[feature_cols + ["segment", "conversion"]].copy()
    output_df["treatment"] = (output_df["segment"] == "Womens E-Mail").astype(int)
    output_df["outcome"] = output_df["conversion"].astype(int)
    output_df = output_df[feature_cols + ["treatment", "outcome"]]
    output = _write_dataset(output_df, "hillstrom_womens_conversion_42k.csv")
    return {
        "id": "hillstrom_womens_conversion_42k",
        "name": "Hillstrom Womens Email Conversion 42k",
        "kind": "public_randomized_marketing",
        "path": _relative(output),
        "rows": int(len(output_df)),
        "description": "Hillstrom randomized subset: Womens email vs no email; outcome is conversion.",
        "source_url": "https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["TLearnerGBM", "DRLearnerGBM", "RLearnerGBM", "CausalForest", "TarNet", "DESCN"],
        "quick_train": {"epochs": 2, "batch_size": 512, "learning_rate": 0.001, "max_rows": 20000},
    }


def write_doubleml_uplift() -> dict:
    raw = download_source("doubleml_uplift", "doubleml_uplift_data.csv")
    df = pd.read_csv(raw)
    feature_cols = [col for col in df.columns if col not in {"conversion", "coupon", "ite"}]
    output_df = df.rename(columns={"coupon": "treatment", "conversion": "outcome", "ite": "true_uplift"}).copy()
    output_df["treatment"] = output_df["treatment"].astype(int)
    output_df["outcome"] = output_df["outcome"].astype(int)
    output = _write_dataset(output_df[feature_cols + ["true_uplift", "treatment", "outcome"]], "doubleml_coupon_uplift.csv")
    return {
        "id": "doubleml_coupon_uplift",
        "name": "DoubleML Coupon Uplift",
        "kind": "public_synthetic_uplift",
        "path": _relative(output),
        "rows": int(len(output_df)),
        "description": "DoubleML tutorial uplift dataset with binary coupon treatment, conversion outcome, and known ITE.",
        "source_url": "https://docs.doubleml.org/tutorial/stable/notebooks/Uplift_example_tools.html",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["SLearnerGBM", "TLearnerGBM", "XLearnerGBM", "DRLearnerGBM", "RLearnerGBM", "CausalForest"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 20000},
    }


def write_lalonde() -> dict:
    raw = download_source("lalonde", "lalonde_matchit.csv")
    df = pd.read_csv(raw).drop(columns=["rownames"], errors="ignore")
    feature_cols = ["age", "educ", "race", "married", "nodegree", "re74", "re75"]
    output_df = df[feature_cols + ["treat", "re78"]].rename(columns={"treat": "treatment", "re78": "outcome"})
    output = _write_dataset(output_df, "lalonde_earnings.csv")
    return {
        "id": "lalonde_earnings",
        "name": "LaLonde Earnings",
        "kind": "public_observational_causal",
        "path": _relative(output),
        "rows": int(len(output_df)),
        "description": "Classic National Supported Work/PSID causal dataset. Treatment is job training; outcome is 1978 earnings.",
        "source_url": "https://vincentarelbundock.github.io/Rdatasets/doc/MatchIt/lalonde.html",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "regression",
        "recommended_models": ["TLearnerGBM", "XLearnerGBM", "DRLearnerGBM", "RLearnerGBM", "CausalForest", "CFRNet"],
        "quick_train": {"epochs": 5, "batch_size": 64, "learning_rate": 0.001, "max_rows": 614},
    }


def write_ecdat_treatment() -> dict:
    raw = download_source("ecdat_treatment", "ecdat_treatment.csv")
    df = pd.read_csv(raw).drop(columns=["rownames"], errors="ignore")
    df["treatment"] = df["treat"].astype(str).str.upper().eq("TRUE").astype(int)
    df["outcome"] = pd.to_numeric(df["re78"], errors="coerce")
    feature_cols = ["age", "educ", "ethn", "married", "re74", "re75", "u74", "u75"]
    output_df = df[feature_cols + ["treatment", "outcome"]].dropna().copy()
    output = _write_dataset(output_df, "ecdat_treatment_earnings.csv")
    return {
        "id": "ecdat_treatment_earnings",
        "name": "Ecdat Treatment Earnings",
        "kind": "public_observational_causal",
        "path": _relative(output),
        "rows": int(len(output_df)),
        "description": "Treatment-effect earnings dataset from Ecdat. Treatment is job training; outcome is 1978 earnings.",
        "source_url": "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Treatment.html",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "regression",
        "recommended_models": ["TLearnerGBM", "XLearnerGBM", "DRLearnerGBM", "RLearnerGBM", "OrthogonalDMLGBM", "CausalForest"],
        "quick_train": {"epochs": 5, "batch_size": 128, "learning_rate": 0.001, "max_rows": 2675},
    }


def write_nhefs() -> dict:
    raw = download_source("nhefs_complete", "nhefs_complete.csv")
    df = pd.read_csv(raw)
    feature_cols = ["sex", "race", "age", "education", "smokeintensity", "smokeyrs", "exercise", "active", "wt71"]
    required = feature_cols + ["qsmk", "wt82_71"]
    output_df = df[required].dropna().rename(columns={"qsmk": "treatment", "wt82_71": "outcome"})
    output_df["treatment"] = output_df["treatment"].astype(int)
    output = _write_dataset(output_df, "nhefs_quit_smoking_weight.csv")
    return {
        "id": "nhefs_quit_smoking_weight",
        "name": "NHEFS Quit Smoking Weight",
        "kind": "public_observational_causal",
        "path": _relative(output),
        "rows": int(len(output_df)),
        "description": "NHEFS complete-case data. Treatment is quitting smoking; outcome is weight change from 1971 to 1982.",
        "source_url": "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/nhefs_complete.html",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "regression",
        "recommended_models": ["DRLearnerGBM", "RLearnerGBM", "OrthogonalDMLGBM", "CausalForest", "CFRNet", "DragonNet"],
        "quick_train": {"epochs": 5, "batch_size": 128, "learning_rate": 0.001, "max_rows": 1566},
    }


def write_ihdp() -> dict:
    raw = download_source("ihdp_npci_1", "ihdp_npci_1.csv")
    columns = ["treatment", "yf", "ycf", "mu0", "mu1"] + [f"x{i}" for i in range(1, 26)]
    df = pd.read_csv(raw, header=None, names=columns)
    feature_cols = [f"x{i}" for i in range(1, 26)]
    output_df = df[feature_cols + ["treatment", "yf", "ycf", "mu0", "mu1"]].copy()
    output_df["true_uplift"] = output_df["mu1"] - output_df["mu0"]
    output_df = output_df.rename(columns={"yf": "outcome"})
    output_df["treatment"] = output_df["treatment"].astype(int)
    output = _write_dataset(output_df[feature_cols + ["true_uplift", "ycf", "mu0", "mu1", "treatment", "outcome"]], "ihdp_npci_1.csv")
    return {
        "id": "ihdp_npci_1",
        "name": "IHDP NPCI Trial 1",
        "kind": "public_counterfactual_benchmark",
        "path": _relative(output),
        "rows": int(len(output_df)),
        "description": "IHDP semi-synthetic benchmark with factual/counterfactual outcomes and known treatment effect.",
        "source_url": "https://github.com/rguo12/CIKM18-LCVA/blob/master/datasets/IHDP/csv/ihdp_npci_1.csv",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "regression",
        "recommended_models": ["TLearnerGBM", "DRLearnerGBM", "RLearnerGBM", "CausalForest", "TarNet", "CFRNet", "DragonNet"],
        "quick_train": {"epochs": 10, "batch_size": 64, "learning_rate": 0.001, "max_rows": 747},
    }


def write_smoke_ban() -> dict:
    raw = download_source("smoke_ban", "aer_smoke_ban.csv")
    df = pd.read_csv(raw).drop(columns=["rownames"], errors="ignore")
    feature_cols = ["age", "education", "afam", "hispanic", "gender"]
    output_df = df[feature_cols + ["ban", "smoker"]].dropna().copy()
    output_df["treatment"] = output_df["ban"].astype(str).str.lower().eq("yes").astype(int)
    output_df["outcome"] = output_df["smoker"].astype(str).str.lower().eq("yes").astype(int)
    output_df = output_df[feature_cols + ["treatment", "outcome"]]
    output = _write_dataset(output_df, "aer_smoke_ban.csv")
    return {
        "id": "aer_smoke_ban",
        "name": "AER Workplace Smoke Ban",
        "kind": "public_observational_causal",
        "path": _relative(output),
        "rows": int(len(output_df)),
        "description": "AER workplace smoking-ban dataset. Treatment is work-area smoking ban; outcome is current smoker.",
        "source_url": "https://vincentarelbundock.github.io/Rdatasets/doc/AER/SmokeBan.html",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["SLearnerGBM", "TLearnerGBM", "DRLearnerGBM", "RLearnerGBM", "CausalForest", "CFRNet"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 10000},
    }


def write_thornton_hiv() -> dict:
    raw = download_source("thornton_hiv", "thornton_hiv.csv")
    df = pd.read_csv(raw).drop(columns=["rownames"], errors="ignore")
    feature_cols = ["villnum", "distvct", "age", "hiv2004"]
    output_df = df[feature_cols + ["any", "got"]].dropna().copy()
    output_df["treatment"] = output_df["any"].astype(int)
    output_df["outcome"] = output_df["got"].astype(int)
    output_df = output_df[feature_cols + ["treatment", "outcome"]]
    output = _write_dataset(output_df, "thornton_hiv_incentive.csv")
    return {
        "id": "thornton_hiv_incentive",
        "name": "Thornton HIV Incentive",
        "kind": "public_randomized_experiment",
        "path": _relative(output),
        "rows": int(len(output_df)),
        "description": "Malawi HIV information experiment. Treatment is receiving any incentive; outcome is learning HIV results.",
        "source_url": "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/thornton_hiv.html",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["SLearnerGBM", "TLearnerGBM", "DRLearnerGBM", "RLearnerGBM", "CausalForest", "TarNet"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 4820},
    }


def write_synthetic_sample() -> dict:
    rng = np.random.default_rng(42)
    rows = 5000

    age = rng.normal(38, 11, rows).clip(18, 75)
    recency_days = rng.exponential(18, rows).clip(0, 120)
    prior_spend = rng.gamma(shape=2.2, scale=38, size=rows).clip(0, 600)
    visits_30d = rng.poisson(3, rows).clip(0, 20)
    loyalty_score = rng.beta(2, 4, rows)
    device = rng.choice(["ios", "android", "web"], rows, p=[0.36, 0.34, 0.30])
    region = rng.choice(["east", "south", "west", "north"], rows, p=[0.31, 0.28, 0.22, 0.19])

    high_value = prior_spend > np.quantile(prior_spend, 0.65)
    recent = recency_days < 14
    mobile = np.isin(device, ["ios", "android"])
    true_uplift = 0.02 + 0.10 * high_value + 0.07 * recent + 0.04 * mobile - 0.05 * (visits_30d == 0)
    true_uplift = np.clip(true_uplift, -0.05, 0.28)

    propensity_logit = -0.1 + 0.25 * high_value + 0.15 * mobile - 0.2 * (region == "north")
    propensity = 1 / (1 + np.exp(-propensity_logit))
    treatment = rng.binomial(1, propensity)

    baseline_logit = (
        -2.1
        + 0.018 * (age - 35)
        - 0.016 * recency_days
        + 0.006 * prior_spend
        + 0.11 * visits_30d
        + 0.7 * loyalty_score
        + 0.18 * (region == "east")
    )
    baseline_prob = 1 / (1 + np.exp(-baseline_logit))
    outcome_prob = np.clip(baseline_prob + treatment * true_uplift, 0.01, 0.95)
    outcome = rng.binomial(1, outcome_prob)

    df = pd.DataFrame(
        {
            "age": age.round(2),
            "recency_days": recency_days.round(2),
            "prior_spend": prior_spend.round(2),
            "visits_30d": visits_30d,
            "loyalty_score": loyalty_score.round(4),
            "device": device,
            "region": region,
            "propensity": propensity.round(4),
            "true_uplift": true_uplift.round(4),
            "treatment": treatment,
            "outcome": outcome,
        }
    )
    output = _write_dataset(df, "synthetic_retail_uplift_5k.csv")
    return {
        "id": "synthetic_retail_uplift_5k",
        "name": "Synthetic Retail Uplift 5k",
        "kind": "synthetic",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "A small retail-style uplift dataset with known true_uplift for fast UI testing.",
        "source_url": "",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": ["age", "recency_days", "prior_spend", "visits_30d", "loyalty_score", "device", "region"],
        "task": "classification",
        "recommended_models": ["SLearnerGBM", "TLearnerGBM", "DRLearnerGBM", "RLearnerGBM", "TarNet", "DragonNet", "EFIN"],
        "quick_train": {"epochs": 2, "batch_size": 128, "learning_rate": 0.001, "max_rows": 5000},
    }


def write_synthetic_multi_treatment_sample() -> dict:
    rng = np.random.default_rng(2026)
    rows = 6000
    age = rng.normal(39, 10, rows).clip(18, 75)
    prior_spend = rng.gamma(shape=2.0, scale=42, size=rows).clip(0, 700)
    recency_days = rng.exponential(22, rows).clip(0, 150)
    visits_30d = rng.poisson(3.5, rows).clip(0, 25)
    loyalty_score = rng.beta(2.3, 3.8, rows)
    device = rng.choice(["ios", "android", "web"], rows, p=[0.35, 0.35, 0.30])
    region = rng.choice(["east", "south", "west", "north"], rows, p=[0.30, 0.27, 0.24, 0.19])

    high_value = prior_spend > np.quantile(prior_spend, 0.65)
    recent = recency_days < 14
    mobile = np.isin(device, ["ios", "android"])
    inactive = visits_30d <= 1

    logits = np.vstack(
        [
            np.zeros(rows),
            -0.05 + 0.45 * mobile + 0.20 * recent - 0.10 * high_value,
            -0.10 + 0.55 * recent + 0.20 * inactive + 0.10 * (device == "web"),
            -0.30 + 0.65 * high_value + 0.25 * loyalty_score - 0.15 * inactive,
        ]
    ).T
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    treatment = np.array([rng.choice([0, 1, 2, 3], p=row) for row in probs])

    baseline_logit = (
        -2.25
        + 0.006 * prior_spend
        - 0.014 * recency_days
        + 0.10 * visits_30d
        + 0.85 * loyalty_score
        + 0.12 * (region == "east")
    )
    baseline_prob = 1 / (1 + np.exp(-baseline_logit))
    effect_email = np.clip(0.015 + 0.06 * mobile + 0.04 * recent - 0.03 * inactive, -0.04, 0.18)
    effect_push = np.clip(0.01 + 0.09 * recent + 0.04 * inactive - 0.02 * high_value, -0.03, 0.20)
    effect_coupon = np.clip(0.02 + 0.12 * high_value + 0.05 * loyalty_score - 0.04 * inactive, -0.02, 0.28)
    effects = np.vstack([np.zeros(rows), effect_email, effect_push, effect_coupon]).T
    outcome_prob = np.clip(baseline_prob + effects[np.arange(rows), treatment], 0.01, 0.95)
    outcome = rng.binomial(1, outcome_prob)

    df = pd.DataFrame(
        {
            "age": age.round(2),
            "prior_spend": prior_spend.round(2),
            "recency_days": recency_days.round(2),
            "visits_30d": visits_30d,
            "loyalty_score": loyalty_score.round(4),
            "device": device,
            "region": region,
            "true_effect_email": effect_email.round(4),
            "true_effect_push": effect_push.round(4),
            "true_effect_coupon": effect_coupon.round(4),
            "treatment": treatment,
            "outcome": outcome,
        }
    )
    output = _write_dataset(df, "synthetic_retail_multi_treatment_6k.csv")
    return {
        "id": "synthetic_retail_multi_treatment_6k",
        "name": "Synthetic Retail Multi-Treatment 6k",
        "kind": "synthetic_multi_treatment",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "Synthetic 4-action retail campaign: 0 control, 1 email, 2 push, 3 coupon.",
        "source_url": "",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": ["age", "prior_spend", "recency_days", "visits_30d", "loyalty_score", "device", "region"],
        "task": "classification",
        "recommended_models": ["MultiTLearnerGBM", "MultiDRLearnerGBM", "MultiTLearnerRF", "MultiDRLearnerRF"],
        "quick_train": {"epochs": 1, "batch_size": 128, "learning_rate": 0.001, "max_rows": 6000},
    }


def write_synthetic_continuous_dose_sample() -> dict:
    rng = np.random.default_rng(2027)
    rows = 6000
    age = rng.normal(38, 11, rows).clip(18, 76)
    prior_spend = rng.gamma(shape=2.2, scale=38, size=rows).clip(0, 800)
    recency_days = rng.exponential(24, rows).clip(0, 180)
    visits_30d = rng.poisson(3.2, rows).clip(0, 28)
    loyalty_score = rng.beta(2.1, 3.4, rows)
    device = rng.choice(["ios", "android", "web"], rows, p=[0.36, 0.36, 0.28])
    region = rng.choice(["east", "south", "west", "north"], rows, p=[0.31, 0.25, 0.24, 0.20])

    high_value = prior_spend > np.quantile(prior_spend, 0.68)
    inactive = visits_30d <= 1
    recent = recency_days < 15
    ideal_discount = np.clip(
        0.06
        + 0.08 * high_value
        + 0.06 * inactive
        + 0.04 * (device == "web")
        + 0.04 * (1 - loyalty_score),
        0.02,
        0.36,
    )
    assigned_discount = np.clip(
        ideal_discount
        + rng.normal(0, 0.055, rows)
        + 0.035 * recent
        - 0.025 * (region == "north"),
        0.0,
        0.4,
    )

    baseline_logit = (
        -2.1
        + 0.0055 * prior_spend
        - 0.012 * recency_days
        + 0.08 * visits_30d
        + 0.75 * loyalty_score
        + 0.10 * (region == "east")
    )
    dose_effect = 0.55 * assigned_discount - 4.8 * (assigned_discount - ideal_discount) ** 2
    outcome_prob = np.clip(1 / (1 + np.exp(-(baseline_logit + dose_effect))), 0.01, 0.95)
    outcome = rng.binomial(1, outcome_prob)

    df = pd.DataFrame(
        {
            "age": age.round(2),
            "prior_spend": prior_spend.round(2),
            "recency_days": recency_days.round(2),
            "visits_30d": visits_30d,
            "loyalty_score": loyalty_score.round(4),
            "device": device,
            "region": region,
            "ideal_discount": ideal_discount.round(4),
            "discount_rate": assigned_discount.round(4),
            "outcome": outcome,
        }
    )
    output = _write_dataset(df, "synthetic_retail_continuous_dose_6k.csv")
    return {
        "id": "synthetic_retail_continuous_dose_6k",
        "name": "Synthetic Retail Continuous Dose 6k",
        "kind": "synthetic_continuous_treatment",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "Synthetic discount-intensity campaign with continuous treatment `discount_rate`.",
        "source_url": "",
        "treatment_col": "discount_rate",
        "outcome_col": "outcome",
        "feature_cols": ["age", "prior_spend", "recency_days", "visits_30d", "loyalty_score", "device", "region"],
        "task": "classification",
        "recommended_models": ["DoseResponseGBM", "DoseResponseRF"],
        "quick_train": {"epochs": 1, "batch_size": 128, "learning_rate": 0.001, "max_rows": 6000},
    }


def write_synthetic_llm_routing_sample() -> dict:
    rng = np.random.default_rng(20260517)
    rows = 8000
    task_type = rng.choice(
        ["simple_qa", "coding", "legal_policy", "creative", "math_reasoning", "customer_support"],
        rows,
        p=[0.28, 0.18, 0.13, 0.15, 0.12, 0.14],
    )
    language = rng.choice(["zh", "en", "mixed"], rows, p=[0.48, 0.42, 0.10])
    prompt_tokens = rng.lognormal(mean=5.35, sigma=0.58, size=rows).clip(20, 1800)
    ambiguity_score = rng.beta(2.2, 3.0, rows)
    retrieval_need = rng.beta(1.7, 3.8, rows)
    safety_risk = rng.beta(1.4, 5.2, rows)
    user_value = rng.gamma(shape=2.1, scale=3.2, size=rows).clip(0.2, 30)
    latency_slo_ms = rng.choice([700, 1200, 2000, 3500], rows, p=[0.22, 0.36, 0.30, 0.12])
    prior_fail_rate = rng.beta(1.6, 5.4, rows)
    model_pool_load = rng.beta(2.4, 4.0, rows)

    complex_task = np.isin(task_type, ["coding", "legal_policy", "math_reasoning"])
    creative_task = task_type == "creative"
    support_task = task_type == "customer_support"
    short_prompt = prompt_tokens < 120
    strict_latency = latency_slo_ms <= 1200

    small_quality = (
        0.62
        - 0.18 * complex_task
        - 0.10 * ambiguity_score
        - 0.08 * retrieval_need
        - 0.06 * safety_risk
        - 0.05 * prior_fail_rate
        + 0.04 * support_task
        + rng.normal(0, 0.025, rows)
    )
    strong_gain = (
        0.04
        + 0.17 * complex_task
        + 0.11 * ambiguity_score
        + 0.08 * retrieval_need
        + 0.08 * prior_fail_rate
        + 0.04 * (language == "mixed")
        + 0.03 * creative_task
        - 0.06 * short_prompt
        - 0.05 * strict_latency
        - 0.035 * model_pool_load
    )
    true_uplift = np.clip(strong_gain, -0.06, 0.34)
    small_quality = np.clip(small_quality, 0.05, 0.92)
    strong_quality = np.clip(small_quality + true_uplift, 0.05, 0.98)

    propensity_logit = (
        -0.55
        + 1.05 * complex_task
        + 0.75 * ambiguity_score
        + 0.45 * retrieval_need
        + 0.35 * (user_value > np.quantile(user_value, 0.7))
        - 0.65 * strict_latency
        - 0.35 * model_pool_load
    )
    propensity = 1 / (1 + np.exp(-propensity_logit))
    treatment = rng.binomial(1, propensity)
    outcome_prob = np.where(treatment == 1, strong_quality, small_quality)
    outcome = rng.binomial(1, outcome_prob)

    strong_cost_per_1k = 0.018
    small_cost_per_1k = 0.0022
    incremental_cost = ((strong_cost_per_1k - small_cost_per_1k) * prompt_tokens / 1000.0).clip(0.0003, 0.05)
    latency_penalty = np.where(strict_latency, 0.015 + 0.025 * model_pool_load, 0.006 + 0.010 * model_pool_load)
    policy_value = true_uplift * user_value - incremental_cost - latency_penalty

    df = pd.DataFrame(
        {
            "prompt_tokens": prompt_tokens.round(0).astype(int),
            "ambiguity_score": ambiguity_score.round(4),
            "retrieval_need": retrieval_need.round(4),
            "safety_risk": safety_risk.round(4),
            "user_value": user_value.round(4),
            "latency_slo_ms": latency_slo_ms,
            "prior_fail_rate": prior_fail_rate.round(4),
            "model_pool_load": model_pool_load.round(4),
            "task_type": task_type,
            "language": language,
            "propensity": propensity.round(4),
            "small_model_quality": small_quality.round(4),
            "strong_model_quality": strong_quality.round(4),
            "true_uplift": true_uplift.round(4),
            "incremental_cost": incremental_cost.round(5),
            "latency_penalty": latency_penalty.round(5),
            "oracle_policy_value": policy_value.round(5),
            "treatment": treatment,
            "outcome": outcome,
        }
    )
    output = _write_dataset(df, "synthetic_llm_routing_uplift_8k.csv")
    feature_cols = [
        "prompt_tokens",
        "ambiguity_score",
        "retrieval_need",
        "safety_risk",
        "user_value",
        "latency_slo_ms",
        "prior_fail_rate",
        "model_pool_load",
        "task_type",
        "language",
    ]
    return {
        "id": "synthetic_llm_routing_uplift_8k",
        "name": "Synthetic LLM Routing Uplift 8k",
        "kind": "synthetic_llm_routing",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "Synthetic LLM routing dataset: treatment is strong-model escalation vs cheap model; outcome is task success with known true_uplift and cost columns.",
        "source_url": "docs/UPLIFT_LLM_INTERVIEW_QA.md",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM", "PAVCalibratedDRLearnerLightGBM", "EconMLDRLearner", "CausalForest"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 8000},
    }


def write_synthetic_llm_multi_action_routing_sample() -> dict:
    rng = np.random.default_rng(2026051801)
    rows = 9000
    task_type = rng.choice(
        ["simple_qa", "coding", "legal_policy", "creative", "math_reasoning", "customer_support", "data_analysis"],
        rows,
        p=[0.22, 0.17, 0.12, 0.14, 0.12, 0.14, 0.09],
    )
    language = rng.choice(["zh", "en", "mixed"], rows, p=[0.48, 0.40, 0.12])
    route_action = rng.choice(
        ["strong_model", "rag", "tool", "human_review"],
        rows,
        p=[0.34, 0.28, 0.24, 0.14],
    )
    prompt_tokens = rng.lognormal(mean=5.45, sigma=0.62, size=rows).clip(20, 2200)
    ambiguity_score = rng.beta(2.2, 3.0, rows)
    retrieval_need = rng.beta(1.8, 3.5, rows)
    tool_need = rng.beta(1.7, 3.8, rows)
    safety_risk = rng.beta(1.35, 5.0, rows)
    user_value = rng.gamma(shape=2.0, scale=3.4, size=rows).clip(0.2, 35)
    latency_slo_ms = rng.choice([700, 1200, 2000, 3500], rows, p=[0.22, 0.34, 0.30, 0.14])
    prior_fail_rate = rng.beta(1.7, 5.2, rows)
    model_pool_load = rng.beta(2.5, 4.0, rows)
    evidence_requirement = rng.beta(1.8, 3.1, rows)
    judge_noise_risk = rng.beta(1.6, 4.4, rows)
    budget_pressure = rng.beta(2.0, 3.2, rows)

    complex_task = np.isin(task_type, ["coding", "legal_policy", "math_reasoning", "data_analysis"])
    support_task = task_type == "customer_support"
    creative_task = task_type == "creative"
    strict_latency = latency_slo_ms <= 1200
    short_prompt = prompt_tokens < 120

    cheap_quality = np.clip(
        0.64
        - 0.18 * complex_task
        - 0.09 * ambiguity_score
        - 0.07 * retrieval_need
        - 0.06 * tool_need
        - 0.06 * prior_fail_rate
        + 0.04 * support_task
        + rng.normal(0, 0.025, rows),
        0.05,
        0.94,
    )
    strong_quality = np.clip(
        cheap_quality
        + 0.045
        + 0.16 * complex_task
        + 0.11 * ambiguity_score
        + 0.04 * creative_task
        + 0.04 * (language == "mixed")
        - 0.05 * short_prompt
        - 0.05 * strict_latency
        - 0.04 * model_pool_load,
        0.04,
        0.99,
    )
    rag_quality = np.clip(
        cheap_quality
        + 0.025
        + 0.18 * retrieval_need
        + 0.08 * evidence_requirement
        + 0.04 * (task_type == "legal_policy")
        - 0.04 * strict_latency
        - 0.03 * model_pool_load,
        0.04,
        0.99,
    )
    tool_quality = np.clip(
        cheap_quality
        + 0.020
        + 0.20 * tool_need
        + 0.07 * np.isin(task_type, ["coding", "data_analysis", "math_reasoning"])
        - 0.035 * strict_latency
        - 0.03 * model_pool_load,
        0.04,
        0.99,
    )
    human_quality = np.clip(
        cheap_quality
        + 0.035
        + 0.15 * support_task
        + 0.12 * safety_risk
        + 0.10 * ambiguity_score
        + 0.08 * prior_fail_rate
        - 0.08 * strict_latency
        - 0.02 * model_pool_load,
        0.04,
        0.995,
    )
    action_quality = {
        "strong_model": strong_quality,
        "rag": rag_quality,
        "tool": tool_quality,
        "human_review": human_quality,
    }
    selected_quality = np.choose(
        pd.Categorical(route_action, categories=["strong_model", "rag", "tool", "human_review"]).codes,
        [strong_quality, rag_quality, tool_quality, human_quality],
    )
    true_uplift = np.clip(selected_quality - cheap_quality, -0.12, 0.42)

    action_cost = np.select(
        [
            route_action == "strong_model",
            route_action == "rag",
            route_action == "tool",
            route_action == "human_review",
        ],
        [
            0.0035 + 0.018 * prompt_tokens / 1000.0,
            0.0020 + 0.007 * prompt_tokens / 1000.0 + 0.004 * retrieval_need,
            0.0025 + 0.006 * prompt_tokens / 1000.0 + 0.006 * tool_need,
            0.18 + 0.08 * support_task + 0.05 * safety_risk,
        ],
        default=0.003,
    )
    latency_penalty = np.select(
        [
            route_action == "strong_model",
            route_action == "rag",
            route_action == "tool",
            route_action == "human_review",
        ],
        [
            0.008 + 0.018 * strict_latency + 0.012 * model_pool_load,
            0.010 + 0.018 * strict_latency + 0.015 * retrieval_need,
            0.012 + 0.020 * strict_latency + 0.012 * tool_need,
            0.090 + 0.080 * strict_latency + 0.030 * model_pool_load,
        ],
        default=0.006,
    )
    hallucination_risk = np.clip(
        0.07
        + 0.18 * ambiguity_score
        + 0.16 * safety_risk
        + 0.12 * (route_action == "strong_model")
        - 0.10 * (route_action == "rag")
        - 0.12 * (route_action == "tool")
        - 0.16 * (route_action == "human_review"),
        0.01,
        0.70,
    )
    evidence_failure_risk = np.clip(
        0.08
        + 0.24 * evidence_requirement
        + 0.16 * retrieval_need
        - 0.20 * (route_action == "rag")
        - 0.15 * (route_action == "tool")
        + 0.05 * (route_action == "strong_model"),
        0.01,
        0.75,
    )
    propensity = _sigmoid(
        -0.55
        + 0.95 * complex_task
        + 0.55 * ambiguity_score
        + 0.50 * retrieval_need
        + 0.45 * tool_need
        + 0.55 * safety_risk
        + 0.35 * (user_value > np.quantile(user_value, 0.70))
        - 0.65 * strict_latency
        - 0.45 * budget_pressure
        - 0.25 * model_pool_load
    )
    treatment = rng.binomial(1, propensity)
    outcome_prob = np.where(treatment == 1, selected_quality, cheap_quality)
    outcome = rng.binomial(1, np.clip(outcome_prob, 0.01, 0.99))

    risk_penalty = 0.18 * hallucination_risk + 0.10 * evidence_failure_risk + 0.06 * judge_noise_risk + 0.03 * budget_pressure
    oracle_policy_value = true_uplift * user_value - action_cost - latency_penalty - risk_penalty
    action_values = np.vstack(
        [
            (strong_quality - cheap_quality) * user_value - (0.0035 + 0.018 * prompt_tokens / 1000.0) - (0.008 + 0.018 * strict_latency + 0.012 * model_pool_load),
            (rag_quality - cheap_quality) * user_value - (0.0020 + 0.007 * prompt_tokens / 1000.0 + 0.004 * retrieval_need) - (0.010 + 0.018 * strict_latency + 0.015 * retrieval_need),
            (tool_quality - cheap_quality) * user_value - (0.0025 + 0.006 * prompt_tokens / 1000.0 + 0.006 * tool_need) - (0.012 + 0.020 * strict_latency + 0.012 * tool_need),
            (human_quality - cheap_quality) * user_value - (0.18 + 0.08 * support_task + 0.05 * safety_risk) - (0.090 + 0.080 * strict_latency + 0.030 * model_pool_load),
        ]
    )
    best_codes = np.argmax(action_values, axis=0)
    best_action = np.array(["strong_model", "rag", "tool", "human_review"])[best_codes]
    oracle_best_policy_value = np.max(action_values, axis=0) - risk_penalty

    df = pd.DataFrame(
        {
            "prompt_tokens": prompt_tokens.round(0).astype(int),
            "ambiguity_score": ambiguity_score.round(4),
            "retrieval_need": retrieval_need.round(4),
            "tool_need": tool_need.round(4),
            "safety_risk": safety_risk.round(4),
            "user_value": user_value.round(4),
            "latency_slo_ms": latency_slo_ms,
            "prior_fail_rate": prior_fail_rate.round(4),
            "model_pool_load": model_pool_load.round(4),
            "evidence_requirement": evidence_requirement.round(4),
            "judge_noise_risk": judge_noise_risk.round(4),
            "budget_pressure": budget_pressure.round(4),
            "task_type": task_type,
            "language": language,
            "route_action": route_action,
            "best_action": best_action,
            "propensity": propensity.round(4),
            "cheap_quality": cheap_quality.round(4),
            "strong_quality": strong_quality.round(4),
            "rag_quality": rag_quality.round(4),
            "tool_quality": tool_quality.round(4),
            "human_quality": human_quality.round(4),
            "selected_action_quality": selected_quality.round(4),
            "true_uplift": true_uplift.round(4),
            "incremental_cost": action_cost.round(5),
            "latency_penalty": latency_penalty.round(5),
            "hallucination_risk": hallucination_risk.round(5),
            "evidence_failure_risk": evidence_failure_risk.round(5),
            "oracle_policy_value": oracle_policy_value.round(5),
            "oracle_best_policy_value": oracle_best_policy_value.round(5),
            "treatment": treatment,
            "outcome": outcome,
        }
    )
    output = _write_dataset(df, "synthetic_llm_multi_action_routing_9k.csv")
    feature_cols = [
        "prompt_tokens",
        "ambiguity_score",
        "retrieval_need",
        "tool_need",
        "safety_risk",
        "user_value",
        "latency_slo_ms",
        "prior_fail_rate",
        "model_pool_load",
        "evidence_requirement",
        "judge_noise_risk",
        "budget_pressure",
        "task_type",
        "language",
        "route_action",
    ]
    return {
        "id": "synthetic_llm_multi_action_routing_9k",
        "name": "Synthetic LLM Multi-Action Routing 9k",
        "kind": "synthetic_llm_multi_action_routing",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "Synthetic LLM routing dataset: treatment executes a planned escalation action (strong model, RAG, tool, or human review) versus cheap model; includes cost, latency, hallucination/evidence risk and oracle policy values.",
        "source_url": "docs/LLM_ROUTING_POLICY_PLAYBOOK.md",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": [
            "DRLearnerLightGBM",
            "RLearnerLightGBM",
            "TLearnerLightGBM",
            "PAVCalibratedDRLearnerLightGBM",
            "EconMLDRLearner",
            "CausalForest",
            "ContrastiveUpliftNet",
        ],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 9000},
    }


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    writers = [
        write_criteo_sample,
        write_doubleml_uplift,
        write_lalonde,
        write_ecdat_treatment,
        write_nhefs,
        write_ihdp,
        write_smoke_ban,
        write_thornton_hiv,
        write_synthetic_sample,
        write_synthetic_multi_treatment_sample,
        write_synthetic_continuous_dose_sample,
        write_synthetic_llm_routing_sample,
        write_synthetic_llm_multi_action_routing_sample,
    ]
    datasets = []
    for writer in writers:
        try:
            dataset = writer()
        except Exception as exc:
            print(f"SKIP {writer.__name__}: {exc}")
            continue
        datasets.append(dataset)
        print(f"{dataset['name']}: {dataset['path']} ({dataset['rows']:,} rows)")

    manifest = {"datasets": datasets}
    (DATASET_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(datasets)} dataset entries to {DATASET_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
