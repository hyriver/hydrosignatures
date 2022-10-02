"""Top-level package for HydroSignatures."""
from importlib.metadata import PackageNotFoundError, version

from .exceptions import InputRangeError, InputTypeError, InputValueError
from .print_versions import show_versions
from .hydrosignatures import (
    HydroSignatures,
    compute_exceedance,
    compute_mean_monthly,
    compute_rolling_mean_monthly,
    compute_baseflow,
    compute_bfi,
    compute_si_walsh,
    compute_si_markham,
    extract_exterema,
)

try:
    __version__ = version("hydrosignatures")
except PackageNotFoundError:
    __version__ = "999"

__all__ = [
    "InputRangeError",
    "InputValueError",
    "InputTypeError",
    "HydroSignatures",
    "compute_exceedance",
    "compute_mean_monthly",
    "compute_rolling_mean_monthly",
    "compute_baseflow",
    "compute_bfi",
    "compute_si_walsh",
    "compute_si_markham",
    "extract_exterema",
    "show_versions",
]
