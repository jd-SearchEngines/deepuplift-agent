import importlib.util

import pytest

from deepuplift.models import build_model


@pytest.mark.skipif(importlib.util.find_spec("causalml") is None, reason="causalml not installed")
def test_causalml_random_forest_adapter_is_constructible_when_optional_dependency_exists():
    assert build_model("CausalMLUpliftRandomForest").name == "CausalMLUpliftRandomForest"

