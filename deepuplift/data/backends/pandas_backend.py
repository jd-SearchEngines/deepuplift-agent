from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


class PandasBackend:
    """In-memory backend used by the reference adapters and examples."""

    name = "pandas"
    supports_out_of_core = False

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.copy()

    @classmethod
    def read(cls, path: str, **kwargs: Any) -> "PandasBackend":
        if str(path).lower().endswith((".parquet", ".pq")):
            return cls(pd.read_parquet(path, **kwargs))
        return cls(pd.read_csv(path, **kwargs))

    @property
    def columns(self) -> list[str]:
        return self.frame.columns.tolist()

    def row_count(self) -> int:
        return len(self.frame)

    def select(self, columns: Iterable[str]) -> "PandasBackend":
        return PandasBackend(self.frame[list(columns)])

    def iter_batches(self, batch_size: int = 100_000):
        for start in range(0, len(self.frame), batch_size):
            yield self.frame.iloc[start : start + batch_size].copy()

    def to_pandas(self) -> pd.DataFrame:
        return self.frame.copy()
