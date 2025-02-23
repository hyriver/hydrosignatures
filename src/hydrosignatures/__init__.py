"""Top-level package for HydroSignatures."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from hydrosignatures import exceptions, plot
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
    "__version__",
    "aridity_index",
    "baseflow",
    "baseflow_index",
    "baseflow_recession",
    "exceedance",
    "exceptions",
    "extract_extrema",
    "flashiness_index",
    "flood_moments",
    "flow_duration_curve_slope",
    "mean_monthly",
    "plot",
    "rolling_mean_monthly",
    "seasonality_index_markham",
    "seasonality_index_walsh",
    "show_versions",
]
