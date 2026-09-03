"""Public data-layer API for DeepUplift.

The implementation remains in ``deepuplift.core`` for backwards compatibility.
This package gives open-source users a stable place to discover data contracts,
profiling, preprocessing, and causal-design diagnostics.
"""

from deepuplift.core.data_profile import build_feature_profile, compare_feature_profile
from deepuplift.core.dataset_registry import load_dataset_manifest
from deepuplift.core.diagnostics import diagnose_uplift_data
from deepuplift.core.preprocess import TabularPreprocessor
from deepuplift.core.schema import UpliftConfig

__all__ = [
    "TabularPreprocessor",
    "UpliftConfig",
    "build_feature_profile",
    "compare_feature_profile",
    "diagnose_uplift_data",
    "load_dataset_manifest",
]
