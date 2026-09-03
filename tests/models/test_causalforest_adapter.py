import importlib.util

import pytest

from deepuplift.models import build_model, model_info


@pytest.mark.skipif(importlib.util.find_spec("econml") is None, reason="econml not installed")
def test_causalforest_adapter_is_real_when_optional_dependency_exists():
    model = build_model("CausalForestDML")
    assert model.nuisance_mode == "internal"
    assert model_info("CausalForestDML")["runnable"] is True

