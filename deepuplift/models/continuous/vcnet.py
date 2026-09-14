from __future__ import annotations

import math
from typing import Any

from ._torch_base import TorchContinuousEstimator, _require_torch


class VCNet(TorchContinuousEstimator):
    """Varying-coefficient response model with a continuous dose basis.

    The network learns covariate-dependent coefficients and combines them
    with a Bernstein basis of dose, i.e. ``mu(x,t) = sum_k beta_k(x) B_k(t)``.
    The response is therefore continuous in dose by construction. This native
    port implements the VCNet base architecture; functional targeted
    regularization (VCNet-TR) is not included.
    """

    name = "VCNet"
    architecture_metadata = {
        "implementation_source": "native_port",
        "source_project": "https://github.com/lushleaf/varying-coefficient-net-with-functional-tr",
        "paper": "VCNet and Functional Targeted Regularization for Learning Causal Effects of Continuous Treatments (ICLR 2021)",
        "license": "MIT (DeepUplift implementation); source repository license undeclared; no code copied.",
        "differences_from_reference": "Independent PyTorch base model using a cubic Bernstein varying-coefficient basis; VCNet-TR is not implemented.",
        "functional_targeted_regularization": False,
    }

    def __init__(self, *, basis_degree: int = 3, **kwargs: Any) -> None:
        self.basis_degree = max(1, int(basis_degree))
        super().__init__(**kwargs)

    def _new_network(self, input_dim: int, dose_boundaries):
        torch = _require_torch()
        nn = torch.nn
        hidden = self.hidden_dim
        degree = self.basis_degree
        basis_dim = degree + 1
        binomial = torch.as_tensor([math.comb(degree, k) for k in range(basis_dim)], dtype=torch.float32)

        class VaryingCoefficientNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.representation = nn.Sequential(
                    nn.Linear(input_dim, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden), nn.ReLU(),
                )
                self.coefficients = nn.Sequential(
                    nn.Linear(hidden, hidden), nn.Tanh(),
                    nn.Linear(hidden, basis_dim),
                )
                self.register_buffer("binomial", binomial)

            def forward(self, x, dose):
                representation = self.representation(x)
                coefficients = self.coefficients(representation)
                t = dose.unsqueeze(-1)
                powers = torch.arange(basis_dim, dtype=t.dtype, device=t.device)
                left = (1.0 - t).pow(degree - powers)
                right = t.pow(powers)
                basis = self.binomial * left * right
                return (coefficients * basis).sum(dim=-1)

        return VaryingCoefficientNet()


__all__ = ["VCNet"]
