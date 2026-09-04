from __future__ import annotations

from typing import Any


def optional_backend_status(model_name: str, model_info: dict[str, Any], result_rows: list[dict[str, Any]]) -> str:
    if model_info.get("missing_dependencies"):
        return "NOT_INSTALLED"
    matching = [row for row in result_rows if row.get("model") == model_name]
    if matching and matching[0].get("status") == "PASS":
        return "PASS"
    return "FAIL" if model_info.get("runnable") else "NOT_INSTALLED"


def _status(value: bool | str) -> str:
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    return str(value).upper()


def build_release_gate(*, synthetic_pass: bool, observational_pass: bool, application_pass: bool, rct_pass: bool, multi_pass: bool, continuous_pass: bool, scale_results: list[dict[str, Any]], hillstrom_status: str, criteo_status: str, reproducibility_pass: bool, packaging_pass: bool | str, ci_pass: bool | str, optional_statuses: dict[str, str], rf_pass: bool | None = None) -> dict[str, Any]:
    scale_gate = {str(row["rows"]): "PASS" if row.get("prediction_finite") and row.get("policy_generated") and row.get("peak_memory_mb", 0) > 0 else "FAIL" for row in scale_results}
    packaging_status, ci_status = _status(packaging_pass), _status(ci_pass)
    alpha = all([synthetic_pass, observational_pass, application_pass, rct_pass, packaging_status == "PASS", ci_status == "PASS"])
    beta = alpha and multi_pass and hillstrom_status == "PASS" and criteo_status == "PASS" and all(value == "PASS" for value in optional_statuses.values())
    return {"DATA": {"Synthetic Ground Truth": "PASS" if synthetic_pass else "FAIL", "Observational Nuisance": "PASS" if observational_pass else "FAIL", "Observational Application Pipeline": "PASS" if application_pass else "FAIL"}, "MODELS": {"Meta Learners": "PASS" if synthetic_pass else "FAIL", "RF-backed": "PASS" if (rf_pass if rf_pass is not None else synthetic_pass) else "FAIL", **optional_statuses}, "DECISION": {"Binary": "PASS" if rct_pass else "FAIL", "Multi": "PASS" if multi_pass else "FAIL", "Continuous": "PASS" if continuous_pass else "FAIL / EXPERIMENTAL"}, "EVALUATION": {"Synthetic Ground Truth": "PASS" if synthetic_pass else "FAIL", "Hillstrom": hillstrom_status, "Criteo": criteo_status}, "SCALE": scale_gate, "REPRODUCIBILITY": "PASS" if reproducibility_pass else "FAIL", "PACKAGING": packaging_status, "CI": ci_status, "overall": "READY_FOR_BETA_RELEASE" if beta else "READY_FOR_ALPHA_RELEASE" if alpha else "BLOCKED"}
