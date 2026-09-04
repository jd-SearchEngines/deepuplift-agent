"""Data backend boundaries. Polars is optional."""

from .pandas_backend import PandasBackend
from .polars_backend import PolarsBackend, polars_available

__all__ = ["PandasBackend", "PolarsBackend", "polars_available"]
