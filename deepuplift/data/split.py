from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from deepuplift.contracts import CausalDataset, TreatmentType


def split_dataset(dataset: CausalDataset, *, test_size: float = 0.25, random_state: int = 42) -> tuple[CausalDataset, CausalDataset]:
    frame = dataset.to_pandas().reset_index(drop=True)
    stratify = None
    if dataset.treatment_type == TreatmentType.BINARY:
        candidate = frame[dataset.treatment_col].astype(str)
        if candidate.value_counts().min() >= 2:
            stratify = candidate
    train_idx, test_idx = train_test_split(np.arange(len(frame)), test_size=test_size, random_state=random_state, stratify=stratify)
    return dataset.subset(train_idx), dataset.subset(test_idx)
