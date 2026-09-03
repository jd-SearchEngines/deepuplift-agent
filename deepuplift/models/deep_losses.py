from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _as_column(values: torch.Tensor | float, reference: Optional[torch.Tensor] = None) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        tensor = values
    elif reference is not None:
        tensor = torch.full_like(reference, float(values))
    else:
        tensor = torch.tensor(float(values))
    return tensor.reshape(-1, 1)


def transformed_outcome_proxy(
    y_true: torch.Tensor,
    t_true: torch.Tensor,
    propensity: torch.Tensor | float = 0.5,
) -> torch.Tensor:
    """IPW transformed outcome proxy for uplift ranking and calibration losses."""
    y = _as_column(y_true)
    t = _as_column(t_true, y)
    p = _as_column(propensity, y).to(dtype=y.dtype, device=y.device).clamp(1e-3, 1.0 - 1e-3)
    return y * (t / p - (1.0 - t) / (1.0 - p))


def uplift_pairwise_ranking_loss(
    uplift_score: torch.Tensor,
    y_true: torch.Tensor,
    t_true: torch.Tensor,
    propensity: torch.Tensor | float = 0.5,
    margin: float = 0.0,
    max_pairs: int = 4096,
) -> torch.Tensor:
    """Differentiable pairwise surrogate for ranking high incremental users above low ones."""
    score = _as_column(uplift_score)
    proxy = transformed_outcome_proxy(y_true, t_true, propensity).detach()
    n = score.shape[0]
    if n < 2:
        return score.sum() * 0.0

    diff_proxy = proxy - proxy.T
    pair_mask = diff_proxy > 0
    if not bool(pair_mask.any()):
        return score.sum() * 0.0
    diff_score = score - score.T
    pair_losses = F.softplus(-(diff_score[pair_mask] - margin))
    if pair_losses.numel() > max_pairs:
        idx = torch.linspace(0, pair_losses.numel() - 1, steps=max_pairs, device=pair_losses.device).long()
        pair_losses = pair_losses[idx]
    return pair_losses.mean()


def cost_aware_policy_loss(
    uplift_score: torch.Tensor,
    conversion_value: torch.Tensor | float = 1.0,
    treatment_cost: torch.Tensor | float = 0.0,
    temperature: float = 1.0,
    budget_rate: Optional[float] = None,
    budget_penalty: float = 1.0,
) -> torch.Tensor:
    """Negative expected incremental profit under a soft treatment policy."""
    score = _as_column(uplift_score)
    temp = max(float(temperature), 1e-3)
    policy_prob = torch.sigmoid(score / temp)
    value = _as_column(conversion_value, score).to(dtype=score.dtype, device=score.device)
    cost = _as_column(treatment_cost, score).to(dtype=score.dtype, device=score.device)
    net_value = score * value - cost
    objective = (policy_prob * net_value).mean()
    if budget_rate is not None:
        budget_gap = torch.relu(policy_prob.mean() - float(budget_rate))
        objective = objective - budget_penalty * budget_gap.pow(2)
    return -objective


def uplift_calibration_loss(
    uplift_score: torch.Tensor,
    y_true: torch.Tensor,
    t_true: torch.Tensor,
    propensity: torch.Tensor | float = 0.5,
) -> torch.Tensor:
    """Brier-style calibration loss against an IPW transformed outcome proxy."""
    score = _as_column(uplift_score)
    proxy = transformed_outcome_proxy(y_true, t_true, propensity).to(dtype=score.dtype, device=score.device)
    centered_proxy = proxy - proxy.mean()
    scale = centered_proxy.detach().std().clamp_min(1e-3)
    return F.mse_loss(score, centered_proxy / scale)


def contrastive_uplift_representation_loss(
    representations: torch.Tensor,
    y_true: torch.Tensor,
    t_true: torch.Tensor,
    propensity: torch.Tensor | float = 0.5,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Supervised contrastive loss grouping examples by high/low uplift proxy and treatment arm.

    Positives share the same proxy sign across opposite treatment arms; negatives include
    opposite proxy signs. This encourages treatment-invariant user similarity while keeping
    uplift-relevant separation.
    """
    z = F.normalize(representations, dim=1)
    proxy = transformed_outcome_proxy(y_true, t_true, propensity).reshape(-1)
    treatment = _as_column(t_true, proxy).reshape(-1) > 0.5
    high_proxy = proxy > proxy.median()
    same_proxy = high_proxy[:, None] == high_proxy[None, :]
    cross_arm = treatment[:, None] != treatment[None, :]
    positive_mask = same_proxy & cross_arm
    eye = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    positive_mask = positive_mask & ~eye
    if not bool(positive_mask.any()):
        return z.sum() * 0.0

    logits = (z @ z.T) / max(float(temperature), 1e-3)
    logits = logits.masked_fill(eye, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = positive_mask.sum(dim=1).clamp_min(1)
    per_anchor = -(log_prob * positive_mask).sum(dim=1) / pos_count
    valid_anchor = positive_mask.any(dim=1)
    return per_anchor[valid_anchor].mean()


def llm_routing_cost_quality_loss(
    route_logit: torch.Tensor,
    cheap_quality: torch.Tensor,
    strong_quality: torch.Tensor,
    strong_cost: torch.Tensor | float,
    latency_penalty: torch.Tensor | float = 0.0,
    budget_rate: Optional[float] = None,
    budget_penalty: float = 1.0,
) -> torch.Tensor:
    """Cost-quality objective for deciding whether to escalate to a stronger LLM."""
    logit = _as_column(route_logit)
    route_prob = torch.sigmoid(logit)
    cheap = _as_column(cheap_quality, logit).to(dtype=logit.dtype, device=logit.device)
    strong = _as_column(strong_quality, logit).to(dtype=logit.dtype, device=logit.device)
    cost = _as_column(strong_cost, logit).to(dtype=logit.dtype, device=logit.device)
    latency = _as_column(latency_penalty, logit).to(dtype=logit.dtype, device=logit.device)
    incremental_value = strong - cheap - cost - latency
    objective = (route_prob * incremental_value).mean()
    if budget_rate is not None:
        budget_gap = torch.relu(route_prob.mean() - float(budget_rate))
        objective = objective - budget_penalty * budget_gap.pow(2)
    return -objective


def llm_multi_action_routing_loss(
    action_logits: torch.Tensor,
    cheap_quality: torch.Tensor,
    action_quality: torch.Tensor,
    action_cost: torch.Tensor | float,
    latency_penalty: torch.Tensor | float = 0.0,
    risk_penalty: torch.Tensor | float = 0.0,
    temperature: float = 1.0,
    include_cheap_action: bool = True,
    route_budget_rate: Optional[float] = None,
    budget_penalty: float = 1.0,
) -> torch.Tensor:
    """Cost-quality objective for multi-action LLM routing.

    Actions can represent strong-model, RAG, tool-use or human-review escalation.
    When `include_cheap_action` is true, the loss adds a zero-value cheap-model
    fallback action so the policy can learn not to escalate.
    """
    logits = action_logits
    if logits.dim() != 2:
        raise ValueError("action_logits must be shaped [batch, actions]")
    cheap = _as_column(cheap_quality, logits[:, :1]).to(dtype=logits.dtype, device=logits.device)
    quality = action_quality.to(dtype=logits.dtype, device=logits.device)
    if quality.shape != logits.shape:
        raise ValueError("action_quality must have the same shape as action_logits")
    cost = torch.as_tensor(action_cost, dtype=logits.dtype, device=logits.device)
    if cost.dim() == 0:
        cost = torch.full_like(logits, float(cost.item()))
    cost = cost.reshape_as(logits)
    latency = torch.as_tensor(latency_penalty, dtype=logits.dtype, device=logits.device)
    if latency.dim() == 0:
        latency = torch.full_like(logits, float(latency.item()))
    latency = latency.reshape_as(logits)
    risk = torch.as_tensor(risk_penalty, dtype=logits.dtype, device=logits.device)
    if risk.dim() == 0:
        risk = torch.full_like(logits, float(risk.item()))
    risk = risk.reshape_as(logits)

    incremental_value = quality - cheap - cost - latency - risk
    if include_cheap_action:
        cheap_logits = torch.zeros(logits.shape[0], 1, dtype=logits.dtype, device=logits.device)
        logits_for_policy = torch.cat([cheap_logits, logits], dim=1)
        values_for_policy = torch.cat([torch.zeros_like(cheap_logits), incremental_value], dim=1)
        escalation_prob = torch.softmax(logits_for_policy / max(float(temperature), 1e-3), dim=1)[:, 1:].sum(dim=1)
    else:
        logits_for_policy = logits
        values_for_policy = incremental_value
        escalation_prob = torch.ones(logits.shape[0], dtype=logits.dtype, device=logits.device)

    policy_prob = torch.softmax(logits_for_policy / max(float(temperature), 1e-3), dim=1)
    objective = (policy_prob * values_for_policy).sum(dim=1).mean()
    if route_budget_rate is not None:
        budget_gap = torch.relu(escalation_prob.mean() - float(route_budget_rate))
        objective = objective - budget_penalty * budget_gap.pow(2)
    return -objective


def full_funnel_multitask_loss(
    impression_pred: torch.Tensor,
    click_pred: torch.Tensor,
    conversion_pred: torch.Tensor,
    impression_true: torch.Tensor,
    click_true: torch.Tensor,
    conversion_true: torch.Tensor,
    click_weight: float = 1.0,
    conversion_weight: float = 2.0,
) -> torch.Tensor:
    """ECUP-style full-funnel BCE objective for impression, click, and conversion signals."""
    imp = F.binary_cross_entropy(_as_column(impression_pred), _as_column(impression_true))
    clk = F.binary_cross_entropy(_as_column(click_pred), _as_column(click_true))
    conv = F.binary_cross_entropy(_as_column(conversion_pred), _as_column(conversion_true))
    return imp + click_weight * clk + conversion_weight * conv
