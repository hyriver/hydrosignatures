"""Top-level package for HydroSignatures."""
from importlib.metadata import PackageNotFoundError, version

from hydrosignatures.exceptions import InputRangeError, InputTypeError, InputValueError
from hydrosignatures.hydrosignatures import (
    HydroSignatures,
    aridity_index,
    baseflow,
    baseflow_index,
    exceedance,
    extract_extrema,
    fdc_slope,
    flashiness_index,
    flood_moments,
    mean_monthly,
    rolling_mean_monthly,
    seasonality_index_markham,
    seasonality_index_walsh,
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
    "flood_moments",
    "exceedance",
    "fdc_slope",
    "flashiness_index",
    "mean_monthly",
    "rolling_mean_monthly",
    "baseflow",
    "baseflow_index",
    "aridity_index",
    "seasonality_index_walsh",
    "seasonality_index_markham",
    "extract_extrema",
    "show_versions",
]
