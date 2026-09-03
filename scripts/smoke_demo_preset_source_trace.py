from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.frontier_models import frontier_demo_trace_rows


REQUIRED_DATASETS = {
    "synthetic_ads_full_funnel_ecup_7k": "full_funnel",
    "synthetic_growth_delayed_feedback_7k": "delayed_feedback",
    "synthetic_llm_routing_uplift_8k": "policy",
    "synthetic_llm_multi_action_routing_9k": "multi-action",
}


def main() -> None:
    rows = frontier_demo_trace_rows()
    failures: list[str] = []
    by_dataset = {row.get("dataset_id"): row for row in rows}

    manifest_path = ROOT / "examples" / "datasets" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_ids = {row.get("id") for row in manifest.get("datasets", [])}

    for dataset_id, expected_keyword in REQUIRED_DATASETS.items():
        row = by_dataset.get(dataset_id)
        if row is None:
            failures.append(f"Missing source-trace row for {dataset_id}")
            continue
        if dataset_id not in manifest_ids:
            failures.append(f"Source-trace dataset missing from manifest: {dataset_id}")
        joined = " ".join(str(value) for value in row.values()).lower()
        for required_field in [
            "manifest_source",
            "data_columns",
            "evaluator_contract",
            "metrics_json",
            "compare_history_columns",
            "validation",
        ]:
            if not row.get(required_field):
                failures.append(f"{dataset_id} missing {required_field}")
        if expected_keyword not in joined:
            failures.append(f"{dataset_id} row does not mention expected keyword: {expected_keyword}")

    output = {
        "status": "fail" if failures else "ok",
        "rows": rows,
        "failures": failures,
    }
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    path = reports_dir / "demo_preset_source_trace_latest.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": output["status"], "path": str(path.relative_to(ROOT)), "rows": len(rows), "failures": failures}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
