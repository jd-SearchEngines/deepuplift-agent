"""Compatibility namespace for offline policy evaluation."""

from deepuplift.core.ope import clipping_sensitivity, effective_sample_size, evaluate_ope_policy

__all__ = ["clipping_sensitivity", "effective_sample_size", "evaluate_ope_policy"]
