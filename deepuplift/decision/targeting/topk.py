from __future__ import annotations

import pandas as pd

from deepuplift.contracts import EffectPrediction


def rank_users(prediction: EffectPrediction) -> pd.DataFrame:
    frame = prediction.to_frame().sort_values("recommended_effect", ascending=False).reset_index(drop=True)
    frame["uplift_rank"] = frame.index + 1
    frame["uplift_percentile"] = frame["uplift_rank"] / max(len(frame), 1)
    return frame


def select_top_k(prediction: EffectPrediction, fraction: float = 0.1) -> pd.DataFrame:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")
    frame = rank_users(prediction)
    return frame.head(max(1, int(len(frame) * fraction))).copy()
