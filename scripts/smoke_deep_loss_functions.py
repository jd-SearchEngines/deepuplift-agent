from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.models.deep_losses import (
    contrastive_uplift_representation_loss,
    cost_aware_policy_loss,
    full_funnel_multitask_loss,
    llm_multi_action_routing_loss,
    llm_routing_cost_quality_loss,
    transformed_outcome_proxy,
    uplift_calibration_loss,
    uplift_pairwise_ranking_loss,
)


def _check_grad(name: str, loss: torch.Tensor, parameter: torch.Tensor, failures: list[str], results: dict) -> None:
    parameter.grad = None
    loss.backward(retain_graph=True)
    grad_norm = float(parameter.grad.detach().norm().item()) if parameter.grad is not None else 0.0
    results[name] = {"loss": float(loss.detach().item()), "grad_norm": grad_norm}
    if grad_norm <= 0:
        failures.append(f"{name} produced non-positive gradient norm")


def main() -> None:
    torch.manual_seed(20260517)
    n = 96
    y_true = torch.bernoulli(torch.full((n, 1), 0.28))
    t_true = torch.cat([torch.zeros(n // 2, 1), torch.ones(n - n // 2, 1)], dim=0)
    propensity = torch.full((n, 1), 0.5)
    uplift_score = torch.linspace(-1.2, 1.2, n).reshape(-1, 1).requires_grad_()
    route_logit = torch.randn(n, 1, requires_grad=True)
    action_logits = torch.randn(n, 4, requires_grad=True)
    representations = torch.randn(n, 10, requires_grad=True)
    funnel_logits = torch.randn(n, 3, requires_grad=True)
    funnel = torch.sigmoid(funnel_logits)

    failures: list[str] = []
    results: dict = {}

    proxy = transformed_outcome_proxy(y_true, t_true, propensity)
    if proxy.shape != (n, 1):
        failures.append(f"transformed_outcome_proxy returned shape {tuple(proxy.shape)}")

    _check_grad(
        "uplift_pairwise_ranking_loss",
        uplift_pairwise_ranking_loss(uplift_score, y_true, t_true, propensity),
        uplift_score,
        failures,
        results,
    )
    _check_grad(
        "cost_aware_policy_loss",
        cost_aware_policy_loss(uplift_score, conversion_value=15.0, treatment_cost=1.2, budget_rate=0.35),
        uplift_score,
        failures,
        results,
    )
    _check_grad(
        "uplift_calibration_loss",
        uplift_calibration_loss(uplift_score, y_true, t_true, propensity),
        uplift_score,
        failures,
        results,
    )
    _check_grad(
        "contrastive_uplift_representation_loss",
        contrastive_uplift_representation_loss(representations, y_true, t_true, propensity),
        representations,
        failures,
        results,
    )
    _check_grad(
        "llm_routing_cost_quality_loss",
        llm_routing_cost_quality_loss(
            route_logit,
            cheap_quality=torch.full((n, 1), 0.72),
            strong_quality=torch.full((n, 1), 0.84),
            strong_cost=0.04,
            latency_penalty=0.015,
            budget_rate=0.4,
        ),
        route_logit,
        failures,
        results,
    )
    action_quality = torch.sigmoid(torch.randn(n, 4)) * 0.25 + 0.72
    action_cost = torch.tensor([0.035, 0.022, 0.028, 0.180]).reshape(1, 4).repeat(n, 1)
    action_latency = torch.tensor([0.015, 0.025, 0.030, 0.120]).reshape(1, 4).repeat(n, 1)
    action_risk = torch.tensor([0.035, 0.025, 0.020, 0.015]).reshape(1, 4).repeat(n, 1)
    _check_grad(
        "llm_multi_action_routing_loss",
        llm_multi_action_routing_loss(
            action_logits,
            cheap_quality=torch.full((n, 1), 0.70),
            action_quality=action_quality,
            action_cost=action_cost,
            latency_penalty=action_latency,
            risk_penalty=action_risk,
            route_budget_rate=0.35,
        ),
        action_logits,
        failures,
        results,
    )
    _check_grad(
        "full_funnel_multitask_loss",
        full_funnel_multitask_loss(
            funnel[:, 0],
            funnel[:, 1],
            funnel[:, 2],
            torch.ones(n, 1),
            torch.bernoulli(torch.full((n, 1), 0.35)),
            torch.bernoulli(torch.full((n, 1), 0.12)),
        ),
        funnel_logits,
        failures,
        results,
    )

    out = {
        "status": "fail" if failures else "ok",
        "results": results,
        "failures": failures,
        "evidence": {
            "code_path": "deepuplift/models/deep_losses.py",
            "ui_pages": ["Models -> Loss Catalog", "Evaluation -> Loss-to-Metric Map", "Policy -> ROI simulators"],
        },
    }
    Path("reports").mkdir(exist_ok=True)
    path = Path("reports/deep_loss_functions_smoke_latest.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": out["status"], "path": str(path), "failures": failures}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
