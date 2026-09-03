"""Public data-layer API for DeepUplift."""

from .backends import PandasBackend, PolarsBackend, polars_available
from .diagnostics import diagnose_dataset
from .preprocessing import TabularPreprocessor
from .registry import DatasetRegistry, load_dataset_manifest
from .schema import create_causal_dataset, infer_treatment_type
from .split import split_dataset
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
]
