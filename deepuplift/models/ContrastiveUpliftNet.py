from __future__ import annotations

from .BaseModel import BaseLoss
from .TarNet import TarNet
from .deep_losses import contrastive_uplift_representation_loss


class ContrastiveUpliftNet(TarNet):
    """TARNet-style uplift network with a supervised contrastive representation penalty."""

    pass


def contrastive_uplift_loss(
    t_pred,
    y_preds,
    t_true,
    y_true,
    phi_x=None,
    *_,
    task="regression",
    lambda_contrastive=0.05,
    propensity=0.5,
    temperature=0.2,
):
    factual_loss, outcome_loss, _ = BaseLoss(
        t_pred=t_pred,
        y_preds=y_preds,
        t_true=t_true,
        y_true=y_true,
        loss_type="tarnet",
        IPM=False,
        task=task,
    )
    contrastive_loss = contrastive_uplift_representation_loss(
        phi_x,
        y_true,
        t_true,
        propensity=propensity,
        temperature=temperature,
    )
    total_loss = factual_loss + float(lambda_contrastive) * contrastive_loss
    return total_loss, outcome_loss, contrastive_loss
