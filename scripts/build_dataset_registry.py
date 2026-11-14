#!/usr/bin/env python3
"""Build a local-only dataset registry from the public benchmark manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepuplift.core.dataset_registry import build_dataset_registry, dataset_cards_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("examples/datasets/manifest.json"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    report = build_dataset_registry(args.manifest, root=args.root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dataset_cards_frame(report["cards"]).to_csv(args.output_csv, index=False)

    validation_pass = all(card["validation"]["status"] == "ok" for card in report["cards"])
    availability = {card["dataset_id"]: card["availability_status"] for card in report["cards"]}
    print(f"status={'PASS' if validation_pass else 'FAIL'} datasets={report['dataset_count']}")
    print(f"validation_failures={len(report['failures'])}")
    print(f"availability={json.dumps(availability, ensure_ascii=False, sort_keys=True)}")
    return 0 if validation_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
