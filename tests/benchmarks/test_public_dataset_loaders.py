from deepuplift.data import load_public_dataset, synthetic_ground_truth


def test_synthetic_public_loader_has_ground_truth():
    dataset = load_public_dataset("synthetic", rows=80, seed=3)
    assert dataset.metadata["true_effect_col"] == "true_effect"
    assert len(dataset.to_pandas()) == 80

