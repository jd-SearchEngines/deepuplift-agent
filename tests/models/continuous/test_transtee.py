import pytest

from deepuplift.models import build_model, model_info
from deepuplift.models.continuous import TransTEEContinuousAdapter


def test_transtee_is_explicitly_optional_not_validated_and_fails_closed():
    info = model_info("TransTEE")
    assert info["maturity"] == "OPTIONAL_NOT_VALIDATED"
    assert info["runnable"] is False
    with pytest.raises(NotImplementedError, match="OPTIONAL_NOT_VALIDATED"):
        TransTEEContinuousAdapter()
    with pytest.raises(NotImplementedError, match="interface-only"):
        build_model("TransTEE")
