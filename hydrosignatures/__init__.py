"""Top-level package for HydroSignatures."""
from importlib.metadata import PackageNotFoundError, version

from hydrosignatures.exceptions import InputRangeError, InputTypeError, InputValueError
from hydrosignatures.hydrosignatures import (
    HydroSignatures,
    compute_ai,
    compute_baseflow,
    compute_bfi,
    compute_exceedance,
    compute_fdc_slope,
    compute_flood_moments,
    compute_mean_monthly,
    compute_rolling_mean_monthly,
    compute_si_markham,
    compute_si_walsh,
    extract_extrema,
)
from hydrosignatures.print_versions import show_versions

try:
    __version__ = version("hydrosignatures")
except PackageNotFoundError:
    __version__ = "999"

__all__ = [
    "InputRangeError",
    "InputValueError",
    "InputTypeError",
    "HydroSignatures",
    "compute_flood_moments",
    "compute_exceedance",
    "compute_fdc_slope",
    "compute_mean_monthly",
    "compute_rolling_mean_monthly",
    "compute_baseflow",
    "compute_bfi",
    "compute_ai",
    "compute_si_walsh",
    "compute_si_markham",
    "extract_extrema",
    "show_versions",
]
