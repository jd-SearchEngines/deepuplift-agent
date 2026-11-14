import json
from pathlib import Path


def test_experiment_a_summary_preserves_hermes_pilot_status_boundary():
    report_path = Path(__file__).resolve().parents[2] / "reports" / "EXPERIMENT_A_SUMMARY.json"
    report = json.loads(report_path.read_text())

    assert report["overall_status"] == "PARTIAL"
    assert report["data"]["myBPmyLife"]["status"] == "NOT_AVAILABLE"
    assert report["execution"]["raw_data_committed"] is False
