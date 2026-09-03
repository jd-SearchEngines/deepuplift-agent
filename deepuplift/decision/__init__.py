"""Public decision-layer API for DeepUplift.

Decision functions turn uplift predictions into evaluation evidence and policy
recommendations. They deliberately do not imply that offline estimates are
production causal proof.
"""

from deepuplift.core.evaluator import evaluate_uplift_predictions
from deepuplift.core.ope import clipping_sensitivity, effective_sample_size, evaluate_ope_policy
from deepuplift.core.policy import budget_policy_optimizer, policy_value_curve
from deepuplift.core.predictor import score_top_fraction
from deepuplift.core.readiness import decision_readiness

__all__ = [
    "budget_policy_optimizer",
    "clipping_sensitivity",
    "decision_readiness",
    "effective_sample_size",
    "evaluate_uplift_predictions",
    "evaluate_ope_policy",
    "policy_value_curve",
    "score_top_fraction",
]
