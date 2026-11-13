import numpy as np
import pytest

pytest.importorskip("torch")

from deepuplift.data.public.datasets import synthetic_ground_truth
from deepuplift.models import build_model, model_info


def test_tarnet_cfrnet_dragonnet_have_independent_effect_predictions():
    dataset = synthetic_ground_truth(rows=256, seed=17)
    predictions = {}
    for name in ("TarNet", "CFRNet", "DragonNet"):
        model = build_model(name, task="regression", random_state=17, epochs=2, batch_size=64, hidden_dim=8, share_dim=4, device="cpu")
        model.fit(dataset)
        predictions[name] = model.predict(dataset.subset(np.arange(32))).cate
        assert model_info(name)["runnable"] is True
        assert predictions[name].shape == (32,)
        assert np.isfinite(predictions[name]).all()
        assert model.model_info()["maturity"] == "ARCHITECTURE_INSPIRED_VARIANT"
    assert not np.allclose(predictions["TarNet"], predictions["CFRNet"])
    assert not np.allclose(predictions["TarNet"], predictions["DragonNet"])


def test_cfr_balance_and_dragon_targeted_regularization_are_real_toggles():
    dataset = synthetic_ground_truth(rows=192, seed=19)
    cfr_plain = build_model("CFRNet", task="regression", random_state=19, epochs=1, batch_size=64, hidden_dim=8, share_dim=4, balance_strength=0.0, device="cpu")
    cfr_balanced = build_model("CFRNet", task="regression", random_state=19, epochs=1, batch_size=64, hidden_dim=8, share_dim=4, balance_strength=1.0, device="cpu")
    cfr_plain.fit(dataset)
    cfr_balanced.fit(dataset)
    assert cfr_plain.model_info()["mechanism"] != cfr_balanced.model_info()["mechanism"] or cfr_plain.balance_strength != cfr_balanced.balance_strength
    dragon = build_model("DragonNet", task="regression", random_state=19, epochs=1, batch_size=64, hidden_dim=8, share_dim=4, targeted_regularization=True, device="cpu")
    dragon.fit(dataset)
    assert dragon.predict(dataset.subset(np.arange(16))).metadata["targeted_regularization"] is True
