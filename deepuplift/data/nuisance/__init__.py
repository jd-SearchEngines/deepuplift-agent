from .contracts import NuisanceResult
from .overlap import overlap_report
from .propensity import estimate_propensity
from .weighting import apply_weighting, effective_sample_size


def estimate_nuisance(dataset, *, estimator="logistic", cross_fit=True, n_splits=5, weighting="none", trim_threshold=None, clipping_range=None, random_state=42):
    result = estimate_propensity(dataset, estimator=estimator, cross_fit=cross_fit, n_splits=n_splits, clipping_range=clipping_range, random_state=random_state)
    return apply_weighting(dataset, result, strategy=weighting, trim_threshold=trim_threshold)


__all__ = ["NuisanceResult", "estimate_nuisance", "estimate_propensity", "apply_weighting", "effective_sample_size", "overlap_report"]
