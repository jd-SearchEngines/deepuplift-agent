from deepuplift.data import controlled_observational_stress_test, synthetic_ground_truth


def test_controlled_stress_test_preserves_assignment_provenance():
    original = synthetic_ground_truth(120, 5)
    stressed = controlled_observational_stress_test(original, seed=5, strength=1.5)
    assert stressed.assignment_type.value == "observational"
    assert "original_randomized_treatment" in stressed.to_pandas()
    assert stressed.metadata["stress_test"]["not_real_observational_ground_truth"] is True
