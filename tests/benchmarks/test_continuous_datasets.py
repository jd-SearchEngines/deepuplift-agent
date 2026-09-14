import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deepuplift.benchmarks.continuous import load_giks_continuous_benchmark, load_ihdp, load_news


def test_giks_tensor_and_restricted_numpy_truth_loader(tmp_path):
    n, p, k = 32, 3, 8
    rng = np.random.default_rng(4)
    x = rng.normal(size=(n, p)).astype("float32")
    dose = np.linspace(0.0, 1.0, n, dtype="float32")
    y = (x[:, 0] + dose).astype("float32")
    matrix = np.column_stack([dose, x, y]).astype("float32")
    torch.save(torch.as_tensor(matrix), tmp_path / "data_matrix.pt")
    response = np.repeat((x[:, 0:1] + np.linspace(0.0, 1.0, k)[None, :]), 1, axis=1)
    with (tmp_path / "ihdp_response.pkl").open("wb") as stream:
        pickle.dump(response, stream, protocol=4)
    with (tmp_path / "news_response.pkl").open("wb") as stream:
        pickle.dump(response, stream, protocol=4)
    split = tmp_path / "eval" / "2"
    split.mkdir(parents=True)
    torch.save(torch.arange(0, 24), split / "idx_train.pt")
    torch.save(torch.arange(24, 32), split / "idx_test.pt")

    benchmark = load_giks_continuous_benchmark("ihdp", tmp_path, split_id=2)
    assert benchmark.dataset.metadata["ground_truth_available"] is True
    assert benchmark.fixed_train_indices.shape == (24,)
    assert benchmark.fixed_test_indices.shape == (8,)
    assert benchmark.true_response_curves.shape == (n, k + 1)
    assert load_ihdp(tmp_path, split_id=2).metadata["name"] == "IHDP"
    assert load_news(tmp_path, split_id=2).metadata["name"] == "NEWS"
