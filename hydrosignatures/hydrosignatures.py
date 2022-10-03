"""Function for computing hydrologic signature."""
import calendar
import functools
import json
import warnings
from dataclasses import dataclass
from typing import Dict, NamedTuple, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import signal

from .exceptions import InputRangeError, InputTypeError, InputValueError

try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    from numba import njit, prange

    ngjit = functools.partial(njit, cache=True, nogil=True)
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    prange = range

    def ngjit(ntypes, parallel=None):  # type: ignore
        def decorator_njit(func):  # type: ignore
            @functools.wraps(func)
            def wrapper_decorator(*args, **kwargs):  # type: ignore
                return func(*args, **kwargs)

            return wrapper_decorator

        return decorator_njit


EPS = np.float64(1e-6)
DF = TypeVar("DF", pd.DataFrame, pd.Series)
if HAS_XARRAY:
    ARRAY = TypeVar("ARRAY", pd.Series, pd.DataFrame, npt.NDArray[np.float64], xr.DataArray)
else:
    ARRAY = TypeVar("ARRAY", pd.Series, pd.DataFrame, npt.NDArray[np.float64])  # type: ignore

__all__ = [
    "HydroSignatures",
    "compute_exceedance",
    "compute_mean_monthly",
    "compute_rolling_mean_monthly",
    "compute_baseflow",
    "compute_bfi",
    "compute_si_walsh",
    "compute_si_markham",
    "extract_exterema",
]


def compute_exceedance(
    daily: Union[pd.DataFrame, pd.Series], threshold: float = 1e-3
) -> pd.DataFrame:
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
    fdc = [
        pd.DataFrame({c: _daily[c], f"{c}_rank": ranks[c]})
        .sort_values(by=f"{c}_rank")
        .reset_index(drop=True)
        for c in daily
    ]
    return pd.concat(fdc, axis=1)


def compute_mean_monthly(daily: DF, index_abbr: bool = False) -> DF:
    """Compute mean monthly summary from daily data.

    Parameters
    ----------
    daily : pandas.Series or pandas.DataFrame
        The data to be processed

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Mean monthly summary.
    """
    monthly = daily.resample("M", kind="period").sum()
    mean_month = monthly.groupby(monthly.index.month).mean()
    mean_month.index.name = "month"
    if index_abbr:
        month_abbr = dict(enumerate(calendar.month_abbr))
        mean_month.index = mean_month.index.map(month_abbr)
    return mean_month


def compute_rolling_mean_monthly(daily: DF) -> DF:
    """Compute rolling mean monthly."""
    dayofyear = daily.index.dayofyear
    return daily.rolling(30).mean().groupby(dayofyear).mean()


@ngjit("f8[::1](f8[::1], f8)")
def __forward_pass(q: npt.NDArray[np.float64], alpha: float) -> npt.NDArray[np.float64]:
    """Perform forward pass of the Lyne and Hollick filter."""
    qf = np.zeros_like(q)
    qf[0] = q[0]
    for i in range(1, q.size):
        qf[i] = alpha * qf[i - 1] + 0.5 * (1 + alpha) * (q[i] - q[i - 1])
    qb = np.where(qf > 0, q - qf, q)
    return qb


@ngjit("f8[:,::1](f8[:,::1], f8)", parallel=True)
def __batch_forward(q: npt.NDArray[np.float64], alpha: float) -> npt.NDArray[np.float64]:
    """Perform forward pass of the Lyne and Hollick filter."""
    qb = np.zeros_like(q)
    for i in prange(q.shape[0]):
        qb[i] = __forward_pass(q[i], alpha)
    return qb


@ngjit("f8[::1](f8[::1], f8)")
def __backward_pass(q: npt.NDArray[np.float64], alpha: float) -> npt.NDArray[np.float64]:
    """Perform backward pass of the Lyne and Hollick filter."""
    qf = np.zeros_like(q)
    qf[-1] = q[-1]
    for i in range(q.size - 2, -1, -1):
        qf[i] = alpha * qf[i + 1] + 0.5 * (1 + alpha) * (q[i] - q[i + 1])
    qb = np.where(qf > 0, q - qf, q)
    return qb


@ngjit("f8[:,::1](f8[:,::1], f8)", parallel=True)
def __batch_backward(q: npt.NDArray[np.float64], alpha: float) -> npt.NDArray[np.float64]:
    """Perform forward pass of the Lyne and Hollick filter."""
    qb = np.zeros_like(q)
    for i in prange(q.shape[0]):
        qb[i] = __backward_pass(q[i], alpha)
    return qb


def __to_numpy(arr: ARRAY) -> npt.NDArray[np.float64]:
    """Convert array to numpy array."""
    if isinstance(arr, (pd.Series, pd.DataFrame)):
        q = arr.to_numpy("f8")
    elif HAS_XARRAY and isinstance(arr, xr.DataArray):
        q = arr.astype("f8").to_numpy()
    else:
        q = arr.astype("f8")

    if np.isnan(q).any():
        raise InputTypeError("discharge", "array/dataframe without NaN values")

    if q.ndim == 1:
        q = np.expand_dims(q, axis=0)
    return q


def compute_baseflow(
    discharge: ARRAY, alpha: float = 0.925, n_passes: int = 3, pad_width: int = 10
) -> ARRAY:
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
    if not HAS_NUMBA:
        warnings.warn("Numba not installed. Using slow pure python version.", UserWarning)

    if n_passes < 3 or n_passes % 2 == 0:
        raise InputRangeError("n_passes", "odd numbers greater than 2")

    if not (0 < alpha < 1):
        raise InputRangeError("alpha", "between zero and one")

    if pad_width < 1:
        raise InputRangeError("pad_width", "greater than or equal to 1")

    q = __to_numpy(discharge)
    q = np.apply_along_axis(np.pad, 1, q, pad_width, "edge")
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
    if HAS_XARRAY and isinstance(discharge, xr.DataArray):
        return discharge.copy(data=qb)
    return qb


def compute_bfi(
    discharge: ARRAY, alpha: float = 0.925, n_passes: int = 3, pad_width: int = 10
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
    qb = compute_baseflow(discharge, alpha, n_passes, pad_width)
    return qb.sum() / qsum


def compute_si_walsh(data: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """Compute seasonality index based on Walsh and Lawler, 1981 method."""
    annual = data.resample("Y", kind="period").sum()
    monthly = data.resample("M", kind="period").sum()
    si = pd.DataFrame.from_dict(
        {
            n: 1 / annual.loc[n] * (g - annual.loc[n] / 12).abs().sum()
            for n, g in monthly.resample("Y", kind="period")
        },
        orient="index",
    ).mean()
    si.name = "seasonality_index"
    return si


def compute_si_markham(data: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """Compute seasonality index based on Markham, 1970."""
    if isinstance(data, pd.Series):
        data = data.to_frame()
    monthly = data.resample("M", kind="period").sum()
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


def extract_exterema(ts: pd.Series, var_name: str, n_pts: int) -> pd.DataFrame:
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


class SignaturesBool(NamedTuple):
    """Signatures of a time series.

    Parameters
    ----------
    bfi : bool
        Baseflow index
    runoff_ratio : bool
        Runoff Ratio
    runoff_ratio_annual : bool
        Annual Runoff Ratio
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
    runoff_ratio_anual: np.bool_
    fdc_slope: np.bool_
    mean_monthly: np.bool_
    streamflow_elasticity: np.bool_
    seasonality_index: np.bool_
    mean_annual_flood: np.bool_


class SignaturesFloat(NamedTuple):
    """Signatures of a time series.

    Parameters
    ----------
    bfi : float
        Baseflow index
    runoff_ratio : float
        Runoff Ratio
    runoff_ratio_annual : float
        Annual Runoff Ratio
    fdc_slope : float
        Flow Duration Curve Slope
    mean_monthly : float
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
    runoff_ratio_anual: np.float64
    fdc_slope: np.float64
    mean_monthly: pd.DataFrame
    streamflow_elasticity: np.float64
    seasonality_index: np.float64
    mean_annual_flood: np.float64


@dataclass
class HydroSignatures:
    """Hydrologic signatures.

    Parameters
    ----------
    q_mmpd : pandas.Series
        Discharge in mm per unit time (the same time scale as prcipitation).
    p_mmpd : pandas.Series
        Prcipitation in mm per unit time (the same time scale as discharge).
    si_method : str, optional
        Seasonality index method. Either ``walsh`` or ``markham``. Default is ``walsh``.
    fdc_slope_limits : tuple or list of tuples, optional
        The lower or upper bound percentages to compute the slope of FDC within it,
        defaults to ``(33, 66)``.
    bfi_alpha : float, optional
        Alpha parameter for baseflow separation filter using Lyne and Hollick method.
        Default is ``0.925``.
    """

    q_mmpt: pd.Series
    p_mmpt: pd.Series
    si_method: str = "walsh"
    fdc_slope_limits: Tuple[float, float] = (33, 66)
    bfi_alpha: float = 0.925

    def __post_init__(self) -> None:
        if not isinstance(self.q_mmpt, pd.Series) or not isinstance(self.p_mmpt, pd.Series):
            raise InputTypeError("q_cms/p_mmpd", "pandas.Series")

        if len(self.q_mmpt) != len(self.p_mmpt):
            raise InputTypeError("q_cms/p_mmpd", "dataframes with same length")

        if not all(0 < p < 100 for p in self.fdc_slope_limits):
            raise InputRangeError("fdc_slope_limits", "between 0 and 100")

        self._values = SignaturesFloat(
            self.bfi(),
            self.runoff_ratio(),
            self.runoff_ratio_annual(),
            self.fdc_slope(),
            self.mean_monthy(),
            self.streamflow_elasticity(),
            self.seasonality_index(),
            self.mean_annual_flood(),
        )

    def bfi(self) -> np.float64:
        """Compute Baseflow Index."""
        return compute_bfi(self.q_mmpt.to_numpy("f8"), self.bfi_alpha)

    def runoff_ratio(self) -> np.float64:
        """Compute total runoff ratio."""
        return np.float64(self.q_mmpt.mean() / self.p_mmpt.mean())

    def runoff_ratio_annual(self) -> np.float64:
        """Compute annual runoff ratio."""
        rqp = self.q_mmpt / self.p_mmpt[self.p_mmpt > 0]
        return np.float64(rqp.resample("AS-OCT").sum().mean())

    def fdc(self) -> pd.DataFrame:
        """Compute exceedance probability (for flow duration curve)."""
        return compute_exceedance(self.q_mmpt.to_frame("q"))

    def mean_monthy(self) -> pd.DataFrame:
        """Compute mean monthly flow (for regime curve)."""
        return pd.DataFrame(
            {
                "streamflow": compute_mean_monthly(self.q_mmpt, index_abbr=True),
                "precipitation": compute_mean_monthly(self.p_mmpt, index_abbr=True),
            }
        )

    def seasonality_index(self) -> np.float64:
        """Compute seasonality index."""
        if self.si_method == "walsh":
            return np.float64(compute_si_walsh(self.q_mmpt))
        if self.si_method == "markham":
            return np.float64(compute_si_markham(self.q_mmpt).seasonality_index.iloc[0])
        raise InputValueError("method", ["walsh", "markham"])

    def mean_annual_flood(self) -> np.float64:
        """Compute mean annual flood."""
        return np.float64(self.q_mmpt.resample("Y").max().mean())

    def fdc_slope(self) -> np.float64:
        """Compute FDC slopes between a list of lower and upper percentiles."""
        if not isinstance(self.fdc_slope_limits, (tuple, list)) or len(self.fdc_slope_limits) != 2:
            raise InputTypeError("fdc_slope_limits", "tuple of length 2", "(33, 66)")
        lo, hi = self.fdc_slope_limits
        q_ranked = self.fdc()
        q_hi = q_ranked[q_ranked.q_rank >= hi].iloc[0].q
        q_lo = q_ranked[q_ranked.q_rank >= lo].iloc[0].q
        return np.float64((np.log(q_lo) - np.log(q_hi)) / (hi - lo))

    def streamflow_elasticity(self) -> np.float64:
        """Compute streamflow elasticity."""
        qam = self.q_mmpt.resample("Y").mean()
        pam = self.p_mmpt.resample("Y").mean()
        return np.nanmedian(qam.diff() / pam.diff() * pam / qam)

    @property
    def values(self) -> SignaturesFloat:
        """Return a dictionary with the hydrologic signatures."""
        return self._values

    @property
    def signature_names(self) -> Dict[str, str]:
        """Return a dictionary with the hydrologic signatures."""
        return {
            "bfi": "Baseflow Index",
            "runoff_ratio": "Runoff Ratio",
            "runoff_ratio_annual": "Annual Runoff Ratio",
            "fdc_slope": "Flow Duration Curve Slope",
            "mean_monthly": "Mean Monthly Flow",
            "streamflow_elasticity": "Streamflow Elasticity",
            "seasonality_index": "Seasonality Index",
            "mean_annual_flood": "Mean Annual Flood",
        }

    def to_dict(self) -> Dict[str, Union[np.float64, Dict[str, Dict[str, np.float64]]]]:
        """Return a dictionary with the hydrologic signatures."""
        sigd = self.values._asdict()
        sigd["mean_monthly"] = self.values.mean_monthly.to_dict()
        return sigd

    def to_json(self) -> str:
        """Return a JSON string with the hydrologic signatures."""
        return json.dumps(self.to_dict())

    def diff(self, other: "HydroSignatures") -> SignaturesFloat:
        """Compute absolute difference between two hydrologic signatures."""
        if not isinstance(other, HydroSignatures):
            raise InputTypeError("other", "HydroSignatures")
        this = self.values._asdict()
        _other = other.values._asdict()
        return SignaturesFloat(
            **{key: abs(this[key] - _other[key]) for key in SignaturesFloat._fields}
        )

    def isclose(self, other: "HydroSignatures") -> SignaturesBool:
        """Check if the signatures are close between with a tolerance of 1e-3."""
        if not isinstance(other, HydroSignatures):
            raise InputTypeError("other", "HydroSignatures")
        this = self.values._asdict()
        _other = other.values._asdict()
        return SignaturesBool(
            **{
                key: np.isclose(this[key], _other[key], rtol=1.0e-3)
                for key in SignaturesBool._fields
            }
        )

    def __sub__(self, other: "HydroSignatures") -> SignaturesFloat:
        """Subtract two hydrologic signatures."""
        if not isinstance(other, HydroSignatures):
            raise InputTypeError("other", "HydroSignatures")
        this = self.values._asdict()
        _other = other.values._asdict()
        return SignaturesFloat(**{key: this[key] - _other[key] for key in SignaturesFloat._fields})

    def __lt__(self, other: "HydroSignatures") -> SignaturesBool:
        """Less than two hydrologic signatures."""
        if not isinstance(other, HydroSignatures):
            raise InputTypeError("other", "HydroSignatures")
        this = self.values._asdict()
        _other = other.values._asdict()
        return SignaturesBool(**{key: this[key] < _other[key] for key in SignaturesBool._fields})

    def __le__(self, other: "HydroSignatures") -> SignaturesBool:
        """Less than or equal to two hydrologic signatures."""
        if not isinstance(other, HydroSignatures):
            raise InputTypeError("other", "HydroSignatures")
        this = self.values._asdict()
        _other = other.values._asdict()
        return SignaturesBool(**{key: this[key] <= _other[key] for key in SignaturesBool._fields})

    def __gt__(self, other: "HydroSignatures") -> SignaturesBool:
        """Greater than two hydrologic signatures."""
        if not isinstance(other, HydroSignatures):
            raise InputTypeError("other", "HydroSignatures")
        this = self.values._asdict()
        _other = other.values._asdict()
        return SignaturesBool(**{key: this[key] > _other[key] for key in SignaturesBool._fields})

    def __ge__(self, other: "HydroSignatures") -> SignaturesBool:
        """Greater than or equal to two hydrologic signatures."""
        if not isinstance(other, HydroSignatures):
            raise InputTypeError("other", "HydroSignatures")
        this = self.values._asdict()
        _other = other.values._asdict()
        return SignaturesBool(**{key: this[key] >= _other[key] for key in SignaturesBool._fields})
