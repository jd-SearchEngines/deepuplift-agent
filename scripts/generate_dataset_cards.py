from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.artifacts import save_json
from deepuplift.core.dataset_registry import build_dataset_registry, dataset_cards_frame


def write_markdown(path: Path, payload: dict) -> None:
    cards = payload.get("cards") or []
    lines = [
        "# DeepUplift Dataset Registry",
        "",
        f"Generated at: `{time.strftime('%Y-%m-%d %H:%M:%S %z')}`",
        "",
        "## Summary",
        "",
        f"- Dataset cards: {payload.get('dataset_count')}",
        f"- Failures: {len(payload.get('failures') or [])}",
        f"- Benchmark tiers: `{json.dumps(payload.get('tier_counts') or {}, ensure_ascii=False)}`",
        "",
        "## Dataset Cards",
        "",
        "| Dataset | Kind | Tier | Task | Rows | Features | License / Reuse | Validation |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for card in cards:
        profile = card.get("data_profile") or {}
        license_payload = card.get("license") or {}
        lines.append(
            "| {dataset} | {kind} | {tier} | {task} | {rows} | {features} | {license_status} / {reuse_policy} | `{status}` |".format(
                dataset=card.get("dataset_id"),
                kind=card.get("kind"),
                tier=card.get("benchmark_tier"),
                task=card.get("task"),
                rows=profile.get("rows_observed") or "NA",
                features=(card.get("schema") or {}).get("feature_count"),
                license_status=license_payload.get("license_status"),
                reuse_policy=license_payload.get("reuse_policy"),
                status=(card.get("validation") or {}).get("status"),
            )
        )
    lines.extend(
        [
            "",
            "## Causal Review Contract",
            "",
            "Each card records treatment, outcome, feature columns, source URL, reuse policy, file hash, data fingerprint, and causal assumption risks.",
            "For production pilot use, a dataset card must be paired with overlap diagnostics, promotion gate output, and online experiment design.",
            "",
            "## Failure Rows",
            "",
        ]
    )
    failures = payload.get("failures") or []
    if failures:
        for row in failures:
            lines.append(f"- `{row.get('dataset_id')}`: {row}")
    else:
        lines.append("- None.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DeepUplift dataset cards and registry docs.")
    parser.add_argument("--manifest", default="examples/datasets/manifest.json")
    parser.add_argument("--json-output", default="reports/dataset_registry_latest.json")
    parser.add_argument("--csv-output", default="reports/dataset_registry_latest.csv")
    parser.add_argument("--markdown-output", default="docs/DEEPUplift_DATASET_REGISTRY.md")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    payload = build_dataset_registry(args.manifest, root=ROOT)
    json_path = ROOT / args.json_output
    csv_path = ROOT / args.csv_output
    md_path = ROOT / args.markdown_output
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(json_path, payload)
    dataset_cards_frame(payload.get("cards") or []).to_csv(csv_path, index=False)
    write_markdown(md_path, payload)
    result = {
        "status": "fail" if payload.get("failures") else "ok",
        "json": args.json_output,
        "csv": args.csv_output,
        "markdown": args.markdown_output,
        "dataset_count": payload.get("dataset_count"),
        "failures": payload.get("failures"),
    }
    print(json.dumps(result, ensure_ascii=False))
    if args.strict and payload.get("failures"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
