from __future__ import annotations

import json

import pandas as pd

from deepuplift.core.dataset_registry import build_dataset_card, build_dataset_registry


def test_public_manifest_is_metadata_only_and_does_not_claim_missing_raw_data_is_available():
    report = build_dataset_registry("examples/datasets/manifest.json", root=".")
    assert report["dataset_count"] == 6
    assert report["failures"] == []
    assert all(card["validation"]["status"] == "ok" for card in report["cards"])
    assert all(card["availability_status"] == "NOT_AVAILABLE" for card in report["cards"])
    mrt = next(card for card in report["cards"] if card["dataset_id"] == "mybpmylife_mrt")
    assert mrt["ground_truth_status"] == "assignment_probability_only"
    assert mrt["license"]["reuse_policy"] == "local_only_do_not_redistribute"


def test_registry_records_local_file_hash_schema_and_license(tmp_path):
    frame = pd.DataFrame({"treatment": [0, 1, 0], "outcome": [1.0, 2.0, 1.5], "x1": [0.1, 0.2, 0.3]})
    data_path = tmp_path / "sample.csv"
    frame.to_csv(data_path, index=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "id": "sample",
                        "name": "sample",
                        "kind": "counterfactual_public",
                        "path": "sample.csv",
                        "source_url": "https://example.invalid/sample",
                        "license_status": "upstream_terms_unverified",
                        "reuse_policy": "local_only_do_not_redistribute",
                        "treatment_col": "treatment",
                        "outcome_col": "outcome",
                        "feature_cols": ["x1"],
                        "task": "known_counterfactual_causal",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    card = build_dataset_card(json.loads(manifest_path.read_text(encoding="utf-8"))["datasets"][0], root=tmp_path)
    assert card["validation"]["status"] == "ok"
    assert card["availability_status"] == "PASS"
    assert card["data_profile"]["rows_observed"] == 3
    assert len(card["data_profile"]["file_sha256"]) == 64
    assert card["license"]["license_status"] == "upstream_terms_unverified"
