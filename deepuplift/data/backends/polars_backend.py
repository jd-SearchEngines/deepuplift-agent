from __future__ import annotations

from typing import Any, Iterable


def polars_available() -> bool:
    try:
        import polars  # noqa: F401
    except ImportError:
        return False
    return True


class PolarsBackend:
    """Optional lazy backend; materialization is explicit and user-controlled."""

    name = "polars"
    supports_out_of_core = True

    def __init__(self, frame: Any):
        if not polars_available():
            raise ImportError("PolarsBackend requires the optional 'polars' dependency.")
        self.frame = frame

    @classmethod
    def scan(cls, path: str, **kwargs: Any) -> "PolarsBackend":
        if not polars_available():
            raise ImportError("Install with `pip install 'deepuplift[polars]'` to use PolarsBackend.")
        import polars as pl

        if str(path).lower().endswith((".parquet", ".pq")):
            return cls(pl.scan_parquet(path, **kwargs))
        return cls(pl.scan_csv(path, **kwargs))

    @property
    def columns(self) -> list[str]:
        return list(self.frame.collect_schema().names())

    def select(self, columns: Iterable[str]) -> "PolarsBackend":
        return PolarsBackend(self.frame.select(list(columns)))

    def to_pandas(self):
        return self.frame.collect().to_pandas()
