"""Reusable training and evaluation entry points for DeepUplift."""

from .schema import UpliftConfig


def train_uplift_model(*args, **kwargs):
    from .trainer import train_uplift_model as _train_uplift_model

    return _train_uplift_model(*args, **kwargs)


def score_uplift_run(*args, **kwargs):
    from .predictor import score_uplift_run as _score_uplift_run

    return _score_uplift_run(*args, **kwargs)


__all__ = ["UpliftConfig", "train_uplift_model", "score_uplift_run"]
