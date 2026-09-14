from __future__ import annotations

from deepuplift.contracts import CausalDataset


class TransTEEContinuousAdapter:
    """Fail-closed entry point for TransTEE, pending a validated backend.

    The official source is MIT-licensed and describes continuous treatment,
    but it is not packaged as an installable dependency. This marker class
    deliberately has no fallback implementation and is not runnable.
    """

    name = "TransTEE"
    maturity = "OPTIONAL_NOT_VALIDATED"
    status = "OPTIONAL_NOT_VALIDATED"
    source_project = "https://github.com/hlzhang109/TransTEE"
    license = "MIT"

    def __init__(self, **_kwargs):
        raise NotImplementedError(
            "TransTEEContinuousAdapter is OPTIONAL_NOT_VALIDATED: the official "
            "repository has no installable package and no fit/predict adapter "
            "has passed DeepUplift validation."
        )

    def fit(self, _dataset: CausalDataset):
        raise NotImplementedError("TransTEE is not runnable in this release.")

    def predict(self, _dataset: CausalDataset):
        raise NotImplementedError("TransTEE is not runnable in this release.")


__all__ = ["TransTEEContinuousAdapter"]
