from __future__ import annotations

from typing import Any

from ._torch_base import TorchContinuousEstimator, _require_torch


class DRNet(TorchContinuousEstimator):
    """Dose-stratified neural response model with a shared representation.

    This is a clean native reimplementation of the DRNet-style architecture
    described alongside VCNet: covariates share a representation network,
    continuous dose selects a stratum-specific response head, and every head
    is still conditioned on the numeric dose. No upstream code is copied.
    """

    name = "DRNet"
    architecture_metadata = {
        "implementation_source": "native_port",
        "source_project": "https://github.com/lushleaf/varying-coefficient-net-with-functional-tr",
        "paper": "VCNet and Functional Targeted Regularization for Learning Causal Effects of Continuous Treatments (ICLR 2021), DRNet baseline architecture",
        "license": "MIT (DeepUplift implementation); source repository license undeclared; no code copied.",
        "differences_from_reference": "Independent PyTorch implementation with quantile dose strata and two dose-conditioned hidden layers; no functional targeted regularization.",
    }

    def __init__(self, *, num_dose_bins: int = 5, **kwargs: Any) -> None:
        self.num_dose_bins = max(1, int(num_dose_bins))
        super().__init__(**kwargs)

    def _new_network(self, input_dim: int, dose_boundaries):
        torch = _require_torch()
        nn = torch.nn
        hidden = self.hidden_dim
        boundary_tensor = torch.as_tensor(dose_boundaries, dtype=torch.float32)
        bins = len(dose_boundaries) + 1

        class DoseHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(hidden + 1, hidden)
                self.fc2 = nn.Linear(hidden + 1, hidden)
                self.out = nn.Linear(hidden, 1)

            def forward(self, representation, dose):
                h = torch.relu(self.fc1(torch.cat([representation, dose[:, None]], dim=1)))
                h = torch.relu(self.fc2(torch.cat([h, dose[:, None]], dim=1)))
                return self.out(h).squeeze(-1)

        class DoseStratifiedNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.representation = nn.Sequential(
                    nn.Linear(input_dim, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden), nn.ReLU(),
                )
                self.heads = nn.ModuleList([DoseHead() for _ in range(bins)])
                self.register_buffer("dose_boundaries", boundary_tensor)

            def forward(self, x, dose):
                representation = self.representation(x)
                ids = torch.bucketize(dose.contiguous(), self.dose_boundaries)
                result = torch.empty_like(dose)
                for index, head in enumerate(self.heads):
                    selected = ids == index
                    if selected.any():
                        result[selected] = head(representation[selected], dose[selected])
                return result

        return DoseStratifiedNet()


__all__ = ["DRNet"]
