from __future__ import annotations

import numpy as np

from deepuplift.benchmarks.push_calibration import laplacian_smooth, monotonic_penalty_calibrate, pav_calibrate


def test_pav_and_penalty_reduce_monotonic_violations_but_laplacian_is_not_a_guarantee():
    grid = np.arange(4, dtype="float64")
    curves = np.array([[0.0, 2.0, 1.0, 3.0], [2.0, 1.0, 1.5, 1.0]])
    assert np.all(np.diff(pav_calibrate(curves, grid), axis=1) >= -1e-9)
    assert np.sum(np.diff(monotonic_penalty_calibrate(curves, strength=1.0, steps=200), axis=1) < -1e-4) < np.sum(np.diff(curves, axis=1) < -1e-4)
    smooth = laplacian_smooth(curves)
    assert smooth.shape == curves.shape
    assert np.isfinite(smooth).all()
