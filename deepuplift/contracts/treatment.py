from __future__ import annotations

from enum import Enum


class TreatmentType(str, Enum):
    """Supported treatment shapes."""

    BINARY = "binary"
    MULTI_DISCRETE = "multi_discrete"
    CONTINUOUS = "continuous"


class AssignmentType(str, Enum):
    """How treatment was assigned in the input data."""

    RANDOMIZED = "randomized"
    OBSERVATIONAL = "observational"


def coerce_treatment_type(value: TreatmentType | str) -> TreatmentType:
    if isinstance(value, TreatmentType):
        return value
    normalized = str(value).strip().lower()
    aliases = {
        "multi": TreatmentType.MULTI_DISCRETE,
        "multi-treatment": TreatmentType.MULTI_DISCRETE,
        "multi_treatment": TreatmentType.MULTI_DISCRETE,
        "continuous_treatment": TreatmentType.CONTINUOUS,
    }
    return aliases.get(normalized, TreatmentType(normalized))


def coerce_assignment_type(value: AssignmentType | str) -> AssignmentType:
    if isinstance(value, AssignmentType):
        return value
    return AssignmentType(str(value).strip().lower())
