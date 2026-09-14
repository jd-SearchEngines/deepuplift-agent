from deepuplift.models import model_info


def test_optional_continuous_backends_expose_install_and_readiness_state():
    ccpfn = model_info("CCPFN")
    assert ccpfn["weights_required"] is True
    assert ccpfn["requires_torch"] is True
    assert model_info("DRNet")["requires_torch"] is True
    assert ccpfn["requires_torch"] is True
    assert ccpfn["weights_available"] in {True, False}
    if not ccpfn["dependency_installed"]:
        assert ccpfn["runnable"] is False
        assert ccpfn["missing_dependencies"] == ["ccpfn"]
    transtee = model_info("TransTEE")
    assert transtee["runnable"] is False
    assert transtee["maturity"] == "OPTIONAL_NOT_VALIDATED"
    assert transtee["dependency_installed"] is None
    assert transtee["requires_torch"] is False
