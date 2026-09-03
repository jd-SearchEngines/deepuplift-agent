"""Public data-layer API for DeepUplift."""

from .backends import PandasBackend, PolarsBackend, polars_available
from .diagnostics import diagnose_dataset
from .preprocessing import TabularPreprocessor
from .registry import DatasetRegistry, load_dataset_manifest
from .schema import create_causal_dataset, infer_treatment_type
from .split import split_dataset
from .nuisance import NuisanceResult, apply_weighting, estimate_nuisance, estimate_propensity, effective_sample_size, overlap_report
from .public import controlled_observational_stress_test, load_criteo, load_hillstrom, load_public_dataset, load_retail, synthetic_ground_truth
from .datasets import observational_dataset, randomized_dataset
from deepuplift.contracts import AssignmentType, CausalDataset, DataDiagnostics, TreatmentType

__all__ = [
    "AssignmentType",
    "CausalDataset",
    "DataDiagnostics",
    "DatasetRegistry",
    "PandasBackend",
    "PolarsBackend",
    "TabularPreprocessor",
    "TreatmentType",
    "create_causal_dataset",
    "diagnose_dataset",
    "infer_treatment_type",
    "load_dataset_manifest",
    "polars_available",
    "split_dataset",
    "NuisanceResult",
    "estimate_nuisance",
    "estimate_propensity",
    "apply_weighting",
    "effective_sample_size",
    "overlap_report",
    "synthetic_ground_truth",
    "load_public_dataset",
    "load_hillstrom",
    "load_criteo",
    "load_retail",
    "randomized_dataset",
    "observational_dataset",
    "controlled_observational_stress_test",
]
