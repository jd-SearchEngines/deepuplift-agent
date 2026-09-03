from __future__ import annotations

from typing import Any

from deepuplift.contracts import AssignmentType, PolicyResult


def build_experiment_plan(
    policy: PolicyResult,
    *,
    assignment_type: AssignmentType | str,
    randomization_unit: str = "unit_id",
    primary_metric: str = "incremental_profit",
    secondary_metrics: list[str] | None = None,
    guardrails: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "control_policy": "business-as-usual targeting or holdout",
        "candidate_policy": policy.policy_name,
        "randomization_unit": randomization_unit,
        "primary_metric": primary_metric,
        "secondary_metrics": secondary_metrics or ["incremental_orders", "GMV", "coupon_cost", "iROAS"],
        "guardrails": guardrails or ["contact frequency", "complaints/refunds", "margin", "data quality"],
        "recommended_duration_notes": "Requires baseline rate, minimum detectable effect, traffic and seasonality inputs; no power or duration is fabricated by this library.",
        "rollback_conditions": ["negative primary metric", "guardrail breach", "policy/data drift", "tracking failure"],
        "status": "READY_FOR_EXPERIMENT_DESIGN",
        "requires_user_input": ["baseline", "MDE", "traffic", "duration", "approval and rollback owner"],
        "source_assignment_type": str(getattr(assignment_type, "value", assignment_type)),
        "policy_summary": policy.summary,
    }
