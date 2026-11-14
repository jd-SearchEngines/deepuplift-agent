from __future__ import annotations

from pathlib import Path

from deepuplift.benchmarks.push_continuous import generate_push_continuous
from deepuplift.data import load_mybpmylife_mrt
from tests.data.test_mrt_loader import _fixture


def test_push_continuous_generator_has_three_truth_shapes(tmp_path: Path):
    mrt = load_mybpmylife_mrt(_fixture(tmp_path))
    for shape in ("monotone", "saturating", "fatigue"):
        data = generate_push_continuous(mrt, shape=shape, rows=40, seed=2)
        assert data.dataset.metadata["evidence_class"] == "SEMI_SYNTHETIC_KNOWN_TRUTH"
        assert data.dataset.metadata["dose_is_not_randomized_in_real_mrt"] is True
        assert data.true_response_curves.shape == (40, len(data.dose_grid))
        assert len(data.true_optimal_economic_dose) == 40
