"""Top-level package for HydroSignatures."""

from importlib.metadata import PackageNotFoundError, version

from hydrosignatures import exceptions
from hydrosignatures.baseflow import baseflow, baseflow_index, baseflow_recession
from hydrosignatures.hydrosignatures import (
    HydroSignatures,
    aridity_index,
    exceedance,
    extract_extrema,
    flashiness_index,
    flood_moments,
    flow_duration_curve_slope,
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
    "HydroSignatures",
    "flood_moments",
    "exceedance",
    "flow_duration_curve_slope",
    "flashiness_index",
    "mean_monthly",
    "rolling_mean_monthly",
    "baseflow_recession",
    "baseflow",
    "baseflow_index",
    "aridity_index",
    "seasonality_index_walsh",
    "seasonality_index_markham",
    "extract_extrema",
    "show_versions",
    "exceptions",
    "__version__",
]
