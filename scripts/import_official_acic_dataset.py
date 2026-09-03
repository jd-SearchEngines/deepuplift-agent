from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "examples" / "datasets"
MANIFEST_PATH = DATASET_DIR / "manifest.json"


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"datasets": []}


def _update_manifest(entry: dict) -> None:
    manifest = _load_manifest()
    datasets = manifest.get("datasets") or []
    by_id = {row.get("id"): row for row in datasets if isinstance(row, dict)}
    by_id[entry["id"]] = entry
    ordered = []
    seen = set()
    for row in datasets:
        dataset_id = row.get("id")
        if dataset_id in by_id and dataset_id not in seen:
            ordered.append(by_id[dataset_id])
            seen.add(dataset_id)
    if entry["id"] not in seen:
        ordered.append(entry)
    MANIFEST_PATH.write_text(json.dumps({"datasets": ordered}, ensure_ascii=False, indent=2), encoding="utf-8")


def _infer_feature_cols(df: pd.DataFrame, treatment_col: str, outcome_col: str) -> list[str]:
    excluded = {
        treatment_col,
        outcome_col,
        "mu0",
        "mu1",
        "true_uplift",
        "true_effect",
        "ite",
        "cate",
        "tau",
        "ycf",
        "propensity",
        "oracle_policy_value",
    }
    return [col for col in df.columns if col not in excluded]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Guarded importer for official/local ACIC-style files. It never downloads data by default; "
            "pass a local CSV whose redistribution/license you have checked."
        )
    )
    parser.add_argument("--input", help="Local ACIC CSV path. If omitted, the script exits as guarded_skip.")
    parser.add_argument("--dataset-id", default="acic_official_local")
    parser.add_argument("--name", default="ACIC Official Local Import")
    parser.add_argument("--treatment-col", default="treatment")
    parser.add_argument("--outcome-col", default="outcome")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--output-dir", default=str(DATASET_DIR))
    args = parser.parse_args()

    if not args.input:
        print(
            json.dumps(
                {
                    "status": "guarded_skip",
                    "reason": "No --input provided. Download/obtain official ACIC data manually, verify redistribution rights, then import the local CSV.",
                    "source": "https://acic.berkeley.edu/",
                },
                ensure_ascii=False,
            )
        )
        return

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    df = pd.read_csv(input_path)
    required = {args.treatment_col, args.outcome_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    if "true_uplift" not in df.columns:
        if {"mu0", "mu1"}.issubset(df.columns):
            df["true_uplift"] = pd.to_numeric(df["mu1"], errors="coerce") - pd.to_numeric(df["mu0"], errors="coerce")
        elif {"true_effect"}.issubset(df.columns):
            df["true_uplift"] = df["true_effect"]
    feature_cols = _infer_feature_cols(df, args.treatment_col, args.outcome_col)
    if not feature_cols:
        raise ValueError("No feature columns found after excluding treatment/outcome/effect columns.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.dataset_id}.csv"
    df.to_csv(output, index=False)
    entry = {
        "id": args.dataset_id,
        "name": args.name,
        "kind": "paper_level_official_acic_local",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": (
            "Guarded local import of an official ACIC-style CSV. The repository does not redistribute raw ACIC data by default; "
            "this entry is created only when the user provides a local file."
        ),
        "source_url": "https://acic.berkeley.edu/",
        "treatment_col": args.treatment_col,
        "outcome_col": args.outcome_col,
        "feature_cols": feature_cols,
        "task": args.task,
        "recommended_models": ["TLearnerGBM", "DRLearnerGBM", "CFRNet", "DragonNet", "EFIN", "DESCN"],
        "quick_train": {"epochs": 2, "batch_size": 128, "learning_rate": 0.001, "max_rows": min(int(len(df)), 5000)},
        "license_gate": "User-provided local ACIC file; verify data terms before redistribution.",
    }
    _update_manifest(entry)
    print(json.dumps({"status": "ok", "dataset": entry}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
