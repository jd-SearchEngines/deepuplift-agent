from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DatasetRegistry:
    """Small metadata registry; dataset payloads stay outside source control."""

    def __init__(self):
        self._items: dict[str, dict[str, Any]] = {}

    def register(self, dataset_id: str, **metadata: Any) -> dict[str, Any]:
        item = {"id": dataset_id, **metadata}
        self._items[dataset_id] = item
        return dict(item)

    def get(self, dataset_id: str) -> dict[str, Any]:
        if dataset_id not in self._items:
            raise KeyError(dataset_id)
        return dict(self._items[dataset_id])

    def list(self) -> list[dict[str, Any]]:
        return [dict(self._items[key]) for key in sorted(self._items)]


def load_dataset_manifest(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    datasets = payload.get("datasets") if isinstance(payload, dict) else None
    if not isinstance(datasets, list):
        raise ValueError("Dataset manifest must contain a `datasets` list.")
    return [item for item in datasets if isinstance(item, dict)]
