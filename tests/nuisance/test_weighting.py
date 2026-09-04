from deepuplift.data import apply_weighting, effective_sample_size, estimate_propensity, synthetic_ground_truth


def test_weighting_records_trim_ess_and_balance():
    dataset = synthetic_ground_truth(220, 4)
    nuisance = estimate_propensity(dataset, n_splits=4)
    result = apply_weighting(dataset, nuisance, strategy="stabilized_ipw", trim_threshold=.05)
    weighting = result.diagnostics["weighting"]
    assert weighting["strategy"] == "stabilized_ipw"
    assert "rows_removed" in weighting and "ess_change" in weighting
    assert "balance_before_smd" in weighting and "balance_after_smd" in weighting
    assert effective_sample_size(result.sample_weights) is not None

