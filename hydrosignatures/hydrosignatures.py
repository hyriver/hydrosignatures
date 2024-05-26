"""Function for computing hydrologic signature."""

# pyright: reportGeneralTypeIssues=false
from __future__ import annotations

import calendar
import functools
import json
import warnings
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, cast, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from scipy import signal

from hydrosignatures.exceptions import InputRangeError, InputTypeError, InputValueError

pandas_lt2 = int(pd.__version__.split(".")[0]) < 2
warnings.filterwarnings("ignore", message=".*Converting to PeriodArray/Index.*")
YEAR_END = "Y" if pandas_lt2 else "YE"
MONTH_END = "M" if pandas_lt2 else "ME"

try:
    from numba import config as numba_config
    from numba import njit, prange

    ngjit = functools.partial(njit, nogil=True)  # pyright: ignore[reportAssignmentType]
    numba_config.THREADING_LAYER = "workqueue"
    has_numba = True
except ImportError:
    has_numba = False
    prange = range

    T = TypeVar("T")
    Func = Callable[..., T]

    def ngjit(
        signature_or_function: str | Func[T], parallel: bool = False
    ) -> Callable[[Func[T]], Func[T]]:
        def decorator_njit(func: Func[T]) -> Func[T]:
            @functools.wraps(func)
            def wrapper_decorator(*args: tuple[Any, ...], **kwargs: dict[str, Any]) -> T:
                return func(*args, **kwargs)

            return wrapper_decorator

        return decorator_njit


EPS = np.float64(1e-6)
if TYPE_CHECKING:
    DF = TypeVar("DF", pd.DataFrame, pd.Series)
    FloatArray = npt.NDArray[np.float64]
    ArrayVar = TypeVar("ArrayVar", pd.Series, pd.DataFrame, FloatArray, xr.DataArray)
    ArrayLike = Union[pd.Series, pd.DataFrame, FloatArray, xr.DataArray]

__all__ = [
    "HydroSignatures",
    "flood_moments",
    "exceedance",
    "flow_duration_curve_slope",
    "flashiness_index",
    "mean_monthly",
    "rolling_mean_monthly",
    "baseflow",
    "baseflow_index",
    "aridity_index",
    "seasonality_index_walsh",
    "seasonality_index_markham",
    "extract_extrema",
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


@ngjit("f8[::1](f8[::1], f8)")
def __forward_pass(q: FloatArray, alpha: float) -> FloatArray:
    """Perform forward pass of the Lyne and Hollick filter."""
    qb = np.zeros_like(q)
    qb[0] = q[0]
    for i in range(1, q.size):
        qb[i] = alpha * qb[i - 1] + 0.5 * (1 + alpha) * (q[i] - q[i - 1])

    for i in range(q.size):
        qb[i] = q[i] - qb[i] if qb[i] > 0 else q[i]
    return qb


@ngjit("f8[:,::1](f8[:,::1], f8)", parallel=True)
def __batch_forward(q: FloatArray, alpha: float) -> FloatArray:
    """Perform forward pass of the Lyne and Hollick filter."""
    qb = np.zeros_like(q)
    for i in prange(q.shape[0]):
        qb[i] = __forward_pass(q[i], alpha)
    return qb


@ngjit("f8[::1](f8[::1], f8)")
def __backward_pass(q: FloatArray, alpha: float) -> FloatArray:
    """Perform backward pass of the Lyne and Hollick filter."""
    qf = np.zeros_like(q)
    qf[-1] = q[-1]
    for i in range(q.size - 2, -1, -1):
        qf[i] = alpha * qf[i + 1] + 0.5 * (1 + alpha) * (q[i] - q[i + 1])

    for i in prange(qf.shape[0]):
        qf[i] = q[i] - qf[i] if qf[i] > 0 else q[i]
    return qf


@ngjit("f8[:,::1](f8[:,::1], f8)", parallel=True)
def __batch_backward(q: FloatArray, alpha: float) -> FloatArray:
    """Perform forward pass of the Lyne and Hollick filter."""
    qb = np.zeros_like(q)
    for i in prange(q.shape[0]):
        qb[i] = __backward_pass(q[i], alpha)
    return qb


def baseflow(
    discharge: ArrayVar, alpha: float = 0.925, n_passes: int = 3, pad_width: int = 10
) -> ArrayVar:
    """Extract baseflow using the Lyne and Hollick filter (Ladson et al., 2013).

    Parameters
    ----------
    discharge : numpy.ndarray or pandas.DataFrame or pandas.Series or xarray.DataArray
        Discharge time series that must not have any missing values. It can also be a 2D array
        where each row is a time series.
    n_passes : int, optional
        Number of filter passes, defaults to 3. It must be an odd number greater than 3.
    alpha : float, optional
        Filter parameter that must be between 0 and 1, defaults to 0.925.
    pad_width : int, optional
        Padding width for extending the data from both ends to address the warm up issue.

    Returns
    -------
    numpy.ndarray or pandas.DataFrame or pandas.Series or xarray.DataArray
        Same discharge input array-like but values replaced with computed baseflow values.
    """
    if not has_numba:
        warnings.warn(
            "Numba not installed. Using slow pure python version.", UserWarning, stacklevel=2
        )

    if n_passes < 3 or n_passes % 2 == 0:
        raise InputRangeError("n_passes", "odd numbers greater than 2")

    if not (0 < alpha < 1):
        raise InputRangeError("alpha", "between zero and one")

    if pad_width < 1:
        raise InputRangeError("pad_width", "greater than or equal to 1")

    q = __to_numpy(discharge)
    q = np.apply_along_axis(np.pad, 1, q, pad_width, "edge")
    q = cast("FloatArray", q)
    qb = __batch_forward(q, alpha)
    passes = int(round(0.5 * (n_passes - 1)))
    for _ in range(passes):
        qb = __batch_forward(__batch_backward(qb, alpha), alpha)
    qb = np.ascontiguousarray(qb[:, pad_width:-pad_width])
    qb[qb < 0] = 0.0
    qb = qb.squeeze()
    if isinstance(discharge, pd.Series):
        return pd.Series(qb, index=discharge.index)
    if isinstance(discharge, pd.DataFrame):
        return pd.DataFrame(qb, index=discharge.index, columns=discharge.columns)
    if isinstance(discharge, xr.DataArray):
        return discharge.copy(data=qb)
    return qb


def baseflow_index(
    discharge: ArrayLike, alpha: float = 0.925, n_passes: int = 3, pad_width: int = 10
) -> np.float64:
    """Compute the baseflow index using the Lyne and Hollick filter (Ladson et al., 2013).

    Parameters
    ----------
    discharge : numpy.ndarray or pandas.DataFrame or pandas.Series or xarray.DataArray
        Discharge time series that must not have any missing values. It can also be a 2D array
        where each row is a time series.
    n_passes : int, optional
        Number of filter passes, defaults to 3. It must be an odd number greater than 3.
    alpha : float, optional
        Filter parameter that must be between 0 and 1, defaults to 0.925.
    pad_width : int, optional
        Padding width for extending the data from both ends to address the warm up issue.

    Returns
    -------
    numpy.float64
        The baseflow index.
    """
    qsum = discharge.sum()
    if qsum < EPS:
        return np.float64(0.0)
    qb = baseflow(discharge, alpha, n_passes, pad_width)
    return qb.sum() / qsum


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
    monthly = data.resample(MONTH_END, kind="period").sum()
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

class BaseflowRecession:
    def __init__(self, streamflow, ndmin=7, ndmax=30, nregmx=300):
        self.streamflow = streamflow
        self.ndmin = ndmin
        self.ndmax = ndmax
        self.ndays = len(streamflow)
        self.nregmx = nregmx

        self.alpha = np.zeros(self.ndays)
        self.ndreg = np.zeros(self.ndays, dtype=int)
        self.q0 = np.zeros(self.ndays)
        self.q10 = np.zeros(self.ndays)
        self.bfdd = np.zeros(self.nregmx)
        self.qaveln = np.zeros(self.nregmx)
        self.florec = np.zeros((self.nregmx, 200))
        self.aveflo = np.zeros(self.nregmx)
        self.npreg = np.zeros(self.nregmx, dtype=int)
        self.idone = np.zeros(self.nregmx, dtype=int)
        self.icount = np.zeros(200, dtype=int)

    def process_streamflow_data(self):
        self.calculate_alpha()
        self.process_alpha_data()
        self.estimate_master_recession_curve(self.npr)

    def calculate_alpha(self):
        """Calculate the recession constant alpha and baseflow days."""
        nd = 0
        self.npr = 0

        for i in range(1, self.ndays):
            if np.isnan(self.streamflow[i]) or np.isnan(self.streamflow[i - 1]):
                continue
            if self.streamflow[i] <= 0:
                nd = 0
            else:
                if self.streamflow[i] / self.streamflow[i - 1] < 0.99:
                    if nd >= self.ndmin:
                        if not np.isnan(self.streamflow[i - nd]) and not np.isnan(self.streamflow[i - 1]):
                            self.alpha[i] = np.log(self.streamflow[i - nd] / self.streamflow[i - 1]) / nd
                            self.ndreg[i] = nd
                    nd = 0
                else:
                    nd += 1
                    if nd >= self.ndmax:
                        if not np.isnan(self.streamflow[i - nd + 1]) and not np.isnan(self.streamflow[i]):
                            self.alpha[i] = np.log(self.streamflow[i - nd + 1] / self.streamflow[i]) / nd
                            self.ndreg[i] = nd
                        nd = 0

    def process_alpha_data(self):
        """Process the alpha values to compute the recession constant."""
        for i in range(1, self.ndays):
            if self.alpha[i] > 0:
                if self.npr < self.nregmx - 1:  
                    self.npr += 1
                    if not np.isnan(self.streamflow[i - 1]) and not np.isnan(self.streamflow[i - self.ndreg[i]]):
                        self.q10[self.npr] = self.streamflow[i - 1]
                        self.q0[self.npr] = self.streamflow[i - self.ndreg[i]]
                        if self.q0[self.npr] - self.q10[self.npr] > 0.001:
                            self.bfdd[self.npr] = self.ndreg[i] / (np.log(self.q0[self.npr]) - np.log(self.q10[self.npr]))
                            self.qaveln[self.npr] = np.log((self.q0[self.npr] + self.q10[self.npr]) / 2)
                            self.fill_florec(self.npr, i)
                    else:
                        self.npr -= 1

    def fill_florec(self, npr, i):
        """Fill the florec array with log-transformed flow data."""
        kk = 0
        for k in range(self.ndreg[i]):
            if not np.isnan(self.streamflow[i - k]):
                x = np.log(self.streamflow[i - k])
                if x > 0:
                    kk += 1
                    if kk < self.florec.shape[0] and npr < self.florec.shape[1]:  
                        self.florec[kk, npr] = x
        if kk == 0 and npr > 0:
            npr -= 1

    def estimate_master_recession_curve(self, npr):
        """Estimate the master recession curve."""
        if npr > 1:
            n_points, sumx, sumy, sumxy, sumx2 = 0, 0, 0, 0, 0
            for i in range(1, npr + 1):
                if not np.isnan(self.qaveln[i]) and not np.isnan(self.bfdd[i]):
                    n_points += 1
                    x = self.qaveln[i]
                    sumx += x
                    sumy += self.bfdd[i]
                    sumxy += x * self.bfdd[i]
                    sumx2 += x * x

            if n_points > 1:
                ssxy = sumxy - (sumx * sumy) / n_points
                ssxx = sumx2 - (sumx * sumx) / n_points
                slope_ = ssxy / ssxx
                yint = sumy / n_points - slope_ * sumx / n_points

                self.fill_aveflo(slope_, yint, npr)
                self.calculate_final_alf(n_points, sumx, sumy, sumxy, sumx2, npr)
            else:
                self.alf = np.nan
                self.bfd = np.nan
        else:
            self.alf = np.nan
            self.bfd = np.nan

    def linear_regression_fill(self, slope_, yint, now):
        """Perform linear regression to fill aveflo array."""
        n_points, sumx, sumy, sumxy, sumx2 = 0, 0, 0, 0, 0
        for i in range(1, self.nregmx + 1):
            if i < self.aveflo.size and not np.isnan(self.aveflo[i]) and self.aveflo[i] > 0:  
                n_points += 1
                x = float(i)
                sumx += x
                sumy += self.aveflo[i]
                sumxy += x * self.aveflo[i]
                sumx2 += x * x

        if n_points > 1:
            ssxy = sumxy - (sumx * sumy) / n_points
            ssxx = sumx2 - (sumx * sumx) / n_points
            if ssxx != 0:
                slope_ = ssxy / ssxx
            else:
                slope_ = np.nan

            if np.isfinite(slope_):
                yint = sumy / n_points - slope_ * sumx / n_points
                if now < self.florec.shape[1]:  
                    if slope_ != 0:
                        self.icount[now] = int((self.florec[1, now] - yint) / slope)
                    else:
                        self.icount[now] = 0  
            else:
                self.icount[now] = 0  
        else:
            self.icount[now] = 0  

    def fill_aveflo(self, slope_, yint, npr):
        """Fill the aveflo array and adjust counts."""
        for j in range(1, npr + 1):
            amn = np.inf
            for i in range(1, npr + 1):
                if self.idone[i] == 0 and i < self.florec.shape[1]:  
                    if self.florec[1, i] < amn:
                        amn = self.florec[1, i]
                        now = i
            if now >= self.florec.shape[1]:  
                continue
            self.idone[now] = 1

            igap = 0
            if j == 1:
                self.icount[now] = 1
                igap = 1
            else:
                for i in range(1, self.nregmx + 1):
                    if i < self.aveflo.size and self.florec[1, now] <= self.aveflo[i]:
                        self.icount[now] = i
                        igap = 1
                        break

            if igap == 0:
                self.linear_regression_fill(slope_, yint, now)

            for i in range(1, self.ndmax + 1):
                k = self.icount[now] + i - 1
                if k < 0 or k >= self.aveflo.size:  
                    continue
                if self.florec[i, now] > 0.0001:  
                    self.aveflo[k] = (self.aveflo[k] * self.npreg[k] + self.florec[i, now]) / (self.npreg[k] + 1)
                    if self.aveflo[k] <= 0:
                        self.aveflo[k] = slope_ * i + yint
                    self.npreg[k] += 1

    def calculate_final_alf(self, n_points, sumx, sumy, sumxy, sumx2, npr):
        """Calculate the final alpha factor (alf) and baseflow days (bfd)."""
        n_points, sumx, sumy, sumxy, sumx2 = 0, 0, 0, 0, 0
        for j in range(1, min(npr + 1, self.florec.shape[1])): 
            for i in range(1, min(self.ndmax + 1, self.florec.shape[0])):  
                if not np.isnan(self.florec[i, j]) and self.florec[i, j] > 0:
                    n_points += 1
                    x = float(self.icount[j] + i)
                    sumx += x
                    sumy += self.florec[i, j]
                    sumxy += x * self.florec[i, j]
                    sumx2 += x * x

        if n_points > 1:
            ssxy = sumxy - (sumx * sumy) / n_points
            ssxx = sumx2 - (sumx * sumx) / n_points
            if ssxx != 0:
                self.alf = ssxy / ssxx
                self.bfd = 2.3 / self.alf
            else:
                self.alf = np.nan
                self.bfd = np.nan
        else:
            self.alf = np.nan
            self.bfd = np.nan

    def get_results(self):
        """Return the results as a dictionary."""
        # Extract the data points for the master recession curve
        days = []
        flow_rates = []
        for i in range(self.florec.shape[1]):
            if np.any(self.florec[:, i] > 0):
                valid_indices = np.where(self.florec[:, i] > 0)[0]
                days.extend(valid_indices + 1)
                flow_rates.extend(self.florec[valid_indices, i])

        return {
            'alpha': self.alpha,
            'baseflow_days': self.bfd,
            'alf': self.alf,
            'npr': self.npr,
            'master_rc_days': days,
            'master_rc_flow_rates': flow_rates
        }


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
            **{key: abs(this[key] - _other[key]) for key in SignaturesFloat.fields()}
        )

    def isclose(self, other: HydroSignatures) -> SignaturesBool:
        """Check if the signatures are close between with a tolerance of 1e-3."""
        if not isinstance(other, HydroSignatures):
            raise InputTypeError("other", "HydroSignatures")
        this = self.values
        _other = other.values
        return SignaturesBool(
            **{
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
        return SignaturesFloat(**{key: this[key] - _other[key] for key in SignaturesFloat.fields()})

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
