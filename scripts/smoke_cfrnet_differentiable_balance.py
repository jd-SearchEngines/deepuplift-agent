from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.models.BaseModel import representation_balance_loss


def main() -> None:
    torch.manual_seed(20260517)
    phi = torch.randn(64, 8, requires_grad=True)
    treatment = torch.cat([torch.zeros(32, 1), torch.ones(32, 1)], dim=0)
    results = {}
    failures = []
    for mode in ["mmd_rbf", "mean_l2", "energy_distance"]:
        phi.grad = None
        loss = representation_balance_loss(phi, treatment, mode=mode)
        loss.backward(retain_graph=True)
        grad_norm = float(phi.grad.detach().norm().item()) if phi.grad is not None else 0.0
        results[mode] = {"loss": float(loss.detach().item()), "grad_norm": grad_norm}
        if grad_norm <= 0:
            failures.append(f"{mode} produced non-positive gradient norm")
    out = {"status": "fail" if failures else "ok", "results": results, "failures": failures}
    Path("reports").mkdir(exist_ok=True)
    path = Path("reports/cfrnet_differentiable_balance_smoke_latest.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": out["status"], "path": str(path), "failures": failures}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
