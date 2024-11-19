"""Function for computing hydrologic signature."""

from __future__ import annotations

import calendar
import json
import warnings
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, TypeVar, Union, cast, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from scipy import signal

from hydrosignatures.baseflow import baseflow_index
from hydrosignatures.exceptions import InputRangeError, InputTypeError, InputValueError

pandas_lt2 = int(pd.__version__.split(".")[0]) < 2
warnings.filterwarnings("ignore", message=".*Converting to PeriodArray/Index.*")
YEAR_END = "Y" if pandas_lt2 else "YE"
MONTH_END = "M" if pandas_lt2 else "ME"


EPS = np.float64(1e-6)
if TYPE_CHECKING:
    DF = TypeVar("DF", pd.DataFrame, pd.Series)
    FloatArray = npt.NDArray[np.float64]
    ArrayVar = TypeVar("ArrayVar", pd.Series, pd.DataFrame, FloatArray, xr.DataArray)
    ArrayLike = Union[pd.Series, pd.DataFrame, FloatArray, xr.DataArray]

__all__ = [
    "HydroSignatures",
    "aridity_index",
    "exceedance",
    "extract_extrema",
    "flashiness_index",
    "flood_moments",
    "flow_duration_curve_slope",
    "mean_monthly",
    "rolling_mean_monthly",
    "seasonality_index_markham",
    "seasonality_index_walsh",
]


def flood_moments(streamflow: pd.DataFrame) -> pd.DataFrame:
    """Compute flood moments (MAF, CV, CS) from streamflow.

    Parameters
    ----------
    streamflow : pandas.DataFrame
        The streamflow data to be processed

    Returns
    -------
    pandas.DataFrame
        Flood moments; mean annual flood (MAF), coefficient
        of variation (CV), and coefficient of skewness (CS).
    """
    maf = streamflow.resample(YEAR_END).max().mean()
    n = streamflow.shape[0]
    s2 = np.power(streamflow - maf, 2).sum() / (n - 1)
    cv = np.sqrt(s2) / maf
    cs = n * np.power(streamflow - maf, 3).sum() / ((n - 1) * (n - 2) * np.power(s2, 3.0 / 2.0))

    fm = pd.concat([maf, cv, cs], axis=1)
    fm.columns = ["MAF", "CV", "CS"]
    return fm


def exceedance(daily: pd.DataFrame | pd.Series, threshold: float = 1e-3) -> pd.DataFrame:
    """Compute exceedance probability from daily data.

    Parameters
    ----------
    daily : pandas.Series or pandas.DataFrame
        The data to be processed
    threshold : float, optional
        The threshold to compute exceedance probability, defaults to 1e-3.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Exceedance probability.
    """
    if isinstance(daily, pd.Series):
        daily = daily.to_frame()
    _daily = daily[daily > threshold].copy()
    ranks = _daily.rank(ascending=False, pct=True) * 100
    return pd.concat(
        (
            pd.DataFrame({c: _daily[c], f"{c}_rank": ranks[c]})
            .sort_values(by=f"{c}_rank")
            .reset_index(drop=True)
            for c in daily
        ),
        axis=1,
    )


def __to_numpy(arr: ArrayLike, no_nan: bool = True) -> FloatArray:
    """Convert array to numpy array."""
    if isinstance(arr, (pd.Series, pd.DataFrame)):
        q = arr.to_numpy("f8")
    elif isinstance(arr, xr.DataArray):
        q = arr.astype("f8").to_numpy()
    elif isinstance(arr, np.ndarray):
        q = arr.astype("f8")
    else:
        raise InputTypeError(
            "discharge", "pandas.Series, pandas.DataFrame, numpy.ndarray or xarray.DataArray"
        )
    q = cast("FloatArray", q)

    if no_nan and np.isnan(q).any():
        raise InputTypeError("discharge", "array/dataframe without NaN values")

    if q.ndim == 1:
        q = np.expand_dims(q, axis=0)
    return q


def flow_duration_curve_slope(discharge: ArrayLike, bins: tuple[int, ...], log: bool) -> FloatArray:
    """Compute FDC slopes between the given lower and upper percentiles.

    Parameters
    ----------
    discharge : pandas.Series or pandas.DataFrame or numpy.ndarray or xarray.DataArray
        The discharge data to be processed.
    bins : tuple of int
        Percentile bins for computing FDC slopes between., e.g., (33, 67)
        returns the slope between the 33rd and 67th percentiles.
    log : bool
        Whether to use log-transformed data.

    Returns
    -------
    numpy.ndarray
        The slopes between the given percentiles.
    """
    if not (
        isinstance(bins, (tuple, list))
        and len(bins) >= 2
        and all(0 <= p <= 100 for p in bins)
        and list(bins) == sorted(bins)
    ):
        raise InputRangeError("bins", "tuple with sorted values between 1 and 100")

    q = __to_numpy(discharge, no_nan=False).squeeze()
    q = np.log(q.clip(1e-3)) if log else q
    return np.diff(np.nanpercentile(q, bins, axis=0), axis=0).T / (np.diff(bins) / 100.0)


def flashiness_index(daily: ArrayLike) -> FloatArray:
    """Compute flashiness index from daily data following Baker et al. (2004).

    Parameters
    ----------
    daily : pandas.Series or pandas.DataFrame or numpy.ndarray or xarray.DataArray
        The data to be processed

    Returns
    -------
    numpy.ndarray
        Flashiness index.

    References
    ----------
    Baker, D.B., Richards, R.P., Loftus, T.T. and Kramer, J.W., 2004. A new
    flashiness index: Characteristics and applications to midwestern rivers
    and streams 1. JAWRA Journal of the American Water Resources
    Association, 40(2), pp.503-522.
    """
    q = __to_numpy(daily)
    q = np.diff(q, axis=1) / q[:, :-1]
    return np.nanmean(np.abs(q), axis=1)


def mean_monthly(daily: DF, index_abbr: bool = False, cms: bool = False) -> DF:
    """Compute mean monthly summary from daily data.

    Parameters
    ----------
    daily : pandas.Series or pandas.DataFrame
        The data to be processed
    index_abbr : bool, optional
        Whether to use abbreviated month names as index instead of
        numbers, defaults to False.
    cms : bool, optional
        Whether the input data is in cubic meters per second (cms),
        defaults to False. If True, the mean monthly summary will be
        computed by taking the mean of the daily data, otherwise the
        sum of the daily data will be used.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Mean monthly summary.
    """
    monthly = daily.resample(MONTH_END).mean() if cms else daily.resample(MONTH_END).sum()
    monthly.index = monthly.index.to_period("M")
    mean_month = monthly.groupby(monthly.index.month).mean()
    mean_month.index.name = "month"
    if index_abbr:
        month_abbr = dict(enumerate(calendar.month_abbr))
        mean_month.index = mean_month.index.map(month_abbr)
    return mean_month


def rolling_mean_monthly(daily: DF) -> DF:
    """Compute rolling mean monthly."""
    dayofyear = daily.index.dayofyear
    return daily.rolling(30).mean().groupby(dayofyear).mean()


@overload
def aridity_index(pet: pd.Series, prcp: pd.Series) -> np.float64: ...


@overload
def aridity_index(pet: pd.DataFrame, prcp: pd.DataFrame) -> pd.Series: ...


@overload
def aridity_index(pet: xr.DataArray, prcp: xr.DataArray) -> xr.DataArray: ...


def aridity_index(
    pet: pd.Series | pd.DataFrame | xr.DataArray,
    prcp: pd.Series | pd.DataFrame | xr.DataArray,
) -> np.float64 | pd.Series | xr.DataArray:
    """Compute (Budyko) aridity index (PET/Prcp).

    Parameters
    ----------
    pet : pandas.DataFrame or pandas.Series or xarray.DataArray
        Potential evapotranspiration time series. Each column can
        correspond to PET a different location. Note that ``pet`` and ``prcp``
        must have the same shape.
    prcp : pandas.DataFrame or pandas.Series or xarray.DataArray
        Precipitation time series. Each column can
        correspond to PET a different location. Note that ``pet`` and ``prcp``
        must have the same shape.

    Returns
    -------
    float or pandas.Series or xarray.DataArray
        The aridity index.
    """
    if pet.shape != prcp.shape:
        raise InputTypeError("pet/prcp", "arrays with the same shape")

    if isinstance(pet, xr.DataArray) and isinstance(prcp, xr.DataArray):
        ai = pet.groupby("time.year").mean() / prcp.groupby("time.year").mean()
        return ai.mean(dim="year")

    if isinstance(pet, pd.Series) and isinstance(prcp, pd.Series):
        ai = pet.resample(YEAR_END).mean() / prcp.resample(YEAR_END).mean()
        return ai.mean().item()

    if isinstance(pet, pd.DataFrame) and isinstance(prcp, pd.DataFrame):
        ai = pet.resample(YEAR_END).mean() / prcp.resample(YEAR_END).mean()
        return ai.mean()

    raise InputTypeError("pet/prcp", "pandas.Series/DataFrame or xarray.DataArray")


def seasonality_index_walsh(data: pd.Series | pd.DataFrame) -> pd.Series:
    """Compute seasonality index based on Walsh and Lawler, 1981 method."""
    annual = data.resample(YEAR_END).sum()
    annual.index = annual.index.to_period("Y")
    monthly = data.resample(MONTH_END).sum().resample(YEAR_END)
    si = pd.DataFrame.from_dict(
        {n: 1 / annual.loc[n] * (g - annual.loc[n] / 12).abs().sum() for n, g in monthly},
        orient="index",
    ).mean()
    si.name = "seasonality_index"
    return si


def seasonality_index_markham(data: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Compute seasonality index based on Markham, 1970."""
    if isinstance(data, pd.Series):
        data = data.to_frame()
    monthly = data.resample(MONTH_END).sum()
    monthly.index = monthly.index.to_period("M")
    pm = monthly.groupby(monthly.index.month).sum()
    pm_norm = pm / len(data.index.year.unique())
    phi_m = pd.Series(
        np.deg2rad(
            [15.8, 44.9, 74.0, 104.1, 134.1, 164.2, 194.3, 224.9, 255.0, 285.0, 315.1, 345.2]
        ),
        range(1, 13),
    )
    s = pm_norm.T * np.sin(phi_m)
    s = s.T.sum()
    c = pm_norm.T * np.cos(phi_m)
    c = c.T.sum()
    pr = np.hypot(s, c)
    phi = np.rad2deg(np.arctan(s / c))
    phi[(s < 0) & (c > 0)] += 360.0
    phi[c < 0] += 180.0
    si = pr / pm.sum() * 12.0
    return pd.DataFrame(
        {"seasonality_magnitude": pr, "seasonality_angle": phi, "seasonality_index": si}
    )


def extract_extrema(ts: pd.Series, var_name: str, n_pts: int) -> pd.DataFrame:
    """Get local extrema in a time series.

    Parameters
    ----------
    ts : pandas.Series
        Variable time series.
    var_name : str
        Variable name.
    n_pts : int
        Number of points to consider for detecting local extrema on both
        sides of each point.

    Returns
    -------
    pandas.DataFrame
        A dataframe with three columns: ``var_name``, ``peak`` (bool)
        and ``trough`` (bool).
    """
    ts_df = ts.to_frame(var_name)
    ilocs_min = signal.argrelextrema(ts_df[var_name].to_numpy(), np.less_equal, order=n_pts)[0]
    ilocs_max = signal.argrelextrema(ts_df[var_name].to_numpy(), np.greater_equal, order=n_pts)[0]

    ts_df["peak"] = False
    ts_df.loc[ts_df.iloc[ilocs_max].index, "peak"] = True
    ts_df["trough"] = False
    ts_df.loc[ts_df.iloc[ilocs_min].index, "trough"] = True
    return ts_df


@dataclass(frozen=True)
class SignaturesBool:
    """Signatures of a time series.

    Parameters
    ----------
    bfi : bool
        Baseflow index
    runoff_ratio : bool
        Runoff Ratio
    fdc_slope : bool
        Flow Duration Curve Slope
    mean_monthly : bool
        Mean Monthly Flow
    streamflow_elasticity : bool
        Streamflow Elasticity
    seasonality_index : bool
        Seasonality Index
    mean_annual_flood : bool
        Mean Annual Flood
    """

    bfi: np.bool_
    runoff_ratio: np.bool_
    fdc_slope: np.bool_
    mean_monthly: np.bool_
    streamflow_elasticity: np.bool_
    seasonality_index: np.bool_
    mean_annual_flood: np.bool_

    @classmethod
    def fields(cls) -> tuple[str, ...]:
        """Return the field names."""
        return tuple(f.name for f in fields(cls))


@dataclass(frozen=True)
class SignaturesFloat:
    """Signatures of a time series.

    Parameters
    ----------
    bfi : float
        Baseflow index
    runoff_ratio : float
        Runoff Ratio
    fdc_slope : float or list of float
        Flow Duration Curve Slope
    mean_monthly : pandas.DataFrame
        Mean Monthly Flow
    streamflow_elasticity : float
        Streamflow Elasticity
    seasonality_index : float
        Seasonality Index
    mean_annual_flood : float
        Mean Annual Flood
    """

    bfi: np.float64
    runoff_ratio: np.float64
    fdc_slope: FloatArray
    mean_monthly: pd.DataFrame
    streamflow_elasticity: np.float64
    seasonality_index: np.float64
    mean_annual_flood: np.float64

    @classmethod
    def fields(cls) -> tuple[str, ...]:
        """Return the field names."""
        return tuple(f.name for f in fields(cls))

    def to_dict(self) -> dict[str, np.float64 | dict[str, dict[str, np.float64]]]:
        """Return a dictionary with the hydrological signatures."""
        sigd = {key.name: getattr(self, key.name) for key in fields(SignaturesFloat)}
        sigd["mean_monthly"] = self.mean_monthly.to_dict()
        sigd["fdc_slope"] = self.fdc_slope.tolist()
        return sigd

    def __getitem__(self, key: str) -> np.float64 | FloatArray | pd.DataFrame:
        """Return attribute value."""
        return getattr(self, key)


@dataclass
class HydroSignatures:
    """Hydrological signatures.

    Parameters
    ----------
    q_mmpt : pandas.Series
        Discharge in mm per unit time (the same timescale as precipitation).
    p_mmpt : pandas.Series
        Precipitation in mm per unit time (the same timescale as discharge).
    si_method : str, optional
        Seasonality index method. Either ``walsh`` or ``markham``. Default is ``walsh``.
    fdc_slope_bins : tuple of int, optional
        The percentage bins between 1-100 to compute the slope of FDC within it,
        defaults to ``(33, 67)``.
    bfi_alpha : float, optional
        Alpha parameter for baseflow separation filter using Lyne and Hollick method.
        Default is ``0.925``.
    """

    q_mmpt: pd.Series
    p_mmpt: pd.Series
    si_method: str = "walsh"
    fdc_slope_bins: tuple[int, ...] = (33, 67)
    bfi_alpha: float = 0.925

    def __post_init__(self) -> None:
        if not isinstance(self.q_mmpt, pd.Series) or not isinstance(self.p_mmpt, pd.Series):
            raise InputTypeError("q_cms/p_mmpd", "pandas.Series")

        if len(self.q_mmpt) != len(self.p_mmpt):
            raise InputTypeError("q_cms/p_mmpd", "dataframes with same length")

        if not (
            isinstance(self.fdc_slope_bins, (tuple, list))
            and len(self.fdc_slope_bins) >= 2
            and all(1 <= p < 100 for p in self.fdc_slope_bins)
        ):
            raise InputRangeError("fdc_slope_bins", "tuple with values between 1 and 100")

        self._values = SignaturesFloat(
            self.bfi(),
            self.runoff_ratio(),
            self.fdc_slope(),
            self.mean_monthly(),
            self.streamflow_elasticity(),
            self.seasonality_index(),
            self.mean_annual_flood(),
        )

    def bfi(self) -> np.float64:
        """Compute Baseflow Index."""
        return baseflow_index(self.q_mmpt.to_numpy("f8"), self.bfi_alpha)

    def runoff_ratio(self) -> np.float64:
        """Compute total runoff ratio."""
        return np.float64(self.q_mmpt.mean() / self.p_mmpt.mean())

    def fdc(self) -> pd.DataFrame:
        """Compute exceedance probability (for flow duration curve)."""
        return exceedance(self.q_mmpt.to_frame("q"))

    def mean_monthly(self) -> pd.DataFrame:
        """Compute mean monthly flow (for regime curve)."""
        return pd.DataFrame(
            {
                "streamflow": mean_monthly(self.q_mmpt, index_abbr=True),
                "precipitation": mean_monthly(self.p_mmpt, index_abbr=True),
            }
        )

    def seasonality_index(self) -> np.float64:
        """Compute seasonality index."""
        if self.si_method == "walsh":
            return np.float64(seasonality_index_walsh(self.q_mmpt).iloc[0])
        if self.si_method == "markham":
            return np.float64(seasonality_index_markham(self.q_mmpt)["seasonality_index"].iloc[0])
        raise InputValueError("method", ["walsh", "markham"])

    def mean_annual_flood(self) -> np.float64:
        """Compute mean annual flood."""
        return np.float64(self.q_mmpt.resample(YEAR_END).max().mean())

    def fdc_slope(self) -> FloatArray:
        """Compute FDC slopes between a list of lower and upper percentiles."""
        return flow_duration_curve_slope(self.q_mmpt, self.fdc_slope_bins, True)

    def streamflow_elasticity(self) -> np.float64:
        """Compute streamflow elasticity."""
        qam = self.q_mmpt.resample(YEAR_END).mean()
        pam = self.p_mmpt.resample(YEAR_END).mean()
        return np.float64(np.nanmedian(qam.diff() / pam.diff() * pam / qam))

    @property
    def values(self) -> SignaturesFloat:
        """Return a dictionary with the hydrological signatures."""
        return self._values

    @property
    def signature_names(self) -> dict[str, str]:
        """Return a dictionary with the hydrological signatures."""
        return {
            "bfi": "Baseflow Index",
            "runoff_ratio": "Runoff Ratio",
            "fdc_slope": "Flow Duration Curve Slope",
            "mean_monthly": "Mean Monthly Flow",
            "streamflow_elasticity": "Streamflow Elasticity",
            "seasonality_index": "Seasonality Index",
            "mean_annual_flood": "Mean Annual Flood",
        }

    def to_dict(self) -> dict[str, np.float64 | dict[str, dict[str, np.float64]]]:
        """Return a dictionary with the hydrological signatures."""
        return self.values.to_dict()

    def to_json(self) -> str:
        """Return a JSON string with the hydrological signatures."""
        return json.dumps(self.to_dict())

    def diff(self, other: HydroSignatures) -> SignaturesFloat:
        """Compute absolute difference between two hydrological signatures."""
        if not isinstance(other, HydroSignatures):
            raise InputTypeError("other", "HydroSignatures")
        this = self.values
        _other = other.values
        return SignaturesFloat(
            **{  # pyright: ignore[reportArgumentType]
                key: abs(this[key] - _other[key]) for key in SignaturesFloat.fields()
            }
        )

    def isclose(self, other: HydroSignatures) -> SignaturesBool:
        """Check if the signatures are close between with a tolerance of 1e-3."""
        if not isinstance(other, HydroSignatures):
            raise InputTypeError("other", "HydroSignatures")
        this = self.values
        _other = other.values
        return SignaturesBool(
            **{  # pyright: ignore[reportArgumentType]
                key: np.isclose(this[key], _other[key], rtol=1.0e-3)
                for key in SignaturesBool.fields()
            }
        )

    def __sub__(self, other: HydroSignatures) -> SignaturesFloat:
        """Subtract two hydrological signatures."""
        if not isinstance(other, HydroSignatures):
            raise InputTypeError("other", "HydroSignatures")
        this = self.values
        _other = other.values
        return SignaturesFloat(
            **{  # pyright: ignore[reportArgumentType]
                key: this[key] - _other[key] for key in SignaturesFloat.fields()
            }
        )

    def __lt__(self, other: HydroSignatures) -> SignaturesBool:
        """Less than two hydrological signatures."""
        if not isinstance(other, HydroSignatures):
            raise InputTypeError("other", "HydroSignatures")
        this = self.values
        _other = other.values
        return SignaturesBool(
            **{key: np.array(this[key] < _other[key]).all() for key in SignaturesBool.fields()}
        )

    def __le__(self, other: HydroSignatures) -> SignaturesBool:
        """Less than or equal to two hydrological signatures."""
        if not isinstance(other, HydroSignatures):
            raise InputTypeError("other", "HydroSignatures")
        this = self.values
        _other = other.values
        return SignaturesBool(
            **{key: np.array(this[key] <= _other[key]).all() for key in SignaturesBool.fields()}
        )

    def __gt__(self, other: HydroSignatures) -> SignaturesBool:
        """Greater than two hydrological signatures."""
        if not isinstance(other, HydroSignatures):
            raise InputTypeError("other", "HydroSignatures")
        this = self.values
        _other = other.values
        return SignaturesBool(
            **{key: np.array(this[key] > _other[key]).all() for key in SignaturesBool.fields()}
        )

    def __ge__(self, other: HydroSignatures) -> SignaturesBool:
        """Greater than or equal to two hydrological signatures."""
        if not isinstance(other, HydroSignatures):
            raise InputTypeError("other", "HydroSignatures")
        this = self.values
        _other = other.values
        return SignaturesBool(
            **{key: np.array(this[key] >= _other[key]).all() for key in SignaturesBool.fields()}
        )
