"""Function for computing hydrologic signature."""

# pyright: reportGeneralTypeIssues=false
from __future__ import annotations

import calendar
import functools
import json
import warnings
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar, Union, cast, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from scipy import interpolate, optimize, signal, sparse, stats

from hydrosignatures.exceptions import InputRangeError, InputTypeError, InputValueError

pandas_lt2 = int(pd.__version__.split(".")[0]) < 2
warnings.filterwarnings("ignore", message=".*Converting to PeriodArray/Index.*")
YEAR_END = "Y" if pandas_lt2 else "YE"
MONTH_END = "M" if pandas_lt2 else "ME"

try:
    from numba import config as numba_config
    from numba import njit, prange

    ngjit = functools.partial(njit, nogil=True)  # pyright: ignore[reportAssignmentType]
    numba_config.THREADING_LAYER = "workqueue"  # pyright: ignore[reportAttributeAccessIssue]
    has_numba = True
except ImportError:
    has_numba = False
    prange = range

    T = TypeVar("T")
    Func = Callable[..., T]

    def ngjit(parallel: bool = False) -> Callable[[Func[T]], Func[T]]:
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
    "baseflow_recession",
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


@ngjit()
def _pad_array(arr: FloatArray, pad_width: int) -> FloatArray:
    padded_arr = np.empty(len(arr) + 2 * pad_width, dtype=arr.dtype)
    padded_arr[:pad_width] = arr[0]
    padded_arr[pad_width:-pad_width] = arr
    padded_arr[-pad_width:] = arr[-1]
    return padded_arr


@ngjit()
def __forward_pass(q: FloatArray, alpha: float) -> FloatArray:
    """Perform forward pass of the Lyne and Hollick filter."""
    qb = np.zeros_like(q)
    qb[0] = q[0] - np.min(q)  # initial condition, see Su et al. (2016)
    for i in range(1, q.size):
        qb[i] = alpha * qb[i - 1] + 0.5 * (1 + alpha) * (q[i] - q[i - 1])

    for i in range(q.size):
        qb[i] = q[i] - qb[i] if qb[i] > 0 else q[i]
    return qb


@ngjit(parallel=True)
def __batch_forward(q: FloatArray, alpha: float) -> FloatArray:
    """Perform forward pass of the Lyne and Hollick filter."""
    qb = np.zeros_like(q)
    for i in prange(q.shape[0]):
        qb[i] = __forward_pass(q[i], alpha)
    return qb


@ngjit()
def _recession_segments(
    streamflow: FloatArray,
    freq: np.float64,
    recession_length: np.int64,
    eps: np.float64,
    lyne_hollick_smoothing: np.float64,
    start_of_recession: str,
    n_start: np.int64,
) -> npt.NDArray[np.int64]:
    """Identify all individual recession segments.

    Parameters
    ----------
    streamflow : numpy.ndarray
        Streamflow data.
    freq : float
        Frequency of streamflow data.
    recession_length : int
        Minimum length of recession segments.
    eps : float
        Allowed increase in flow during recession period.
    lyne_hollick_smoothing : float
        Smoothing parameter for Lyne-Hollick filter.
    start_of_recession : str
        Method to determine start of recession ('baseflow' or 'peak').
    n_start : int
        Days to be removed after start of recession.

    Returns
    -------
    numpy.ndarray
        Array of recession segments start and end indices.
    """
    len_decrease = recession_length / freq
    decreasing_flow = streamflow[1:] < (streamflow[:-1] + eps)
    start_point = np.where(~decreasing_flow)[0][0]
    decreasing_flow = decreasing_flow[start_point:]

    flow_change = np.where(decreasing_flow[:-1] != decreasing_flow[1:])[0]
    flow_change = flow_change[: 2 * (len(flow_change) // 2)].reshape(-1, 2)

    flow_section = flow_change[(flow_change[:, 1] - flow_change[:, 0]) >= len_decrease + n_start]
    flow_section += start_point
    flow_section[:, 0] += n_start

    if start_of_recession == "baseflow":
        pad_width = 10
        q_bf = _pad_array(streamflow, pad_width)
        q_bf = __batch_forward(np.atleast_2d(q_bf), lyne_hollick_smoothing)
        q_bf = np.ascontiguousarray(q_bf[0, pad_width:-pad_width])
        is_baseflow = np.isclose(q_bf, streamflow)
        for i, (start, end) in enumerate(flow_section):
            is_b_section = is_baseflow[start : end + 1]
            if not np.any(is_b_section):
                flow_section[i] = -1
            else:
                flow_section[i, 0] += np.argmax(is_b_section)

        flow_section = flow_section[flow_section[:, 0] >= 0]
        flow_section = flow_section[(flow_section[:, 1] - flow_section[:, 0]) >= 3]

    return flow_section


def _exponential_mrc(streamflow: FloatArray, flow_section: npt.NDArray[np.int64]) -> FloatArray:
    """Compute exponential method for master recession curve."""
    start_values = streamflow[flow_section[:, 0]]
    sort_indices = np.argsort(start_values)[::-1]

    mrc = np.column_stack(
        (
            np.arange(1, flow_section[sort_indices[0], 1] - flow_section[sort_indices[0], 0] + 2),
            streamflow[flow_section[sort_indices[0], 0] : flow_section[sort_indices[0], 1] + 1],
        )
    )

    for i in range(1, flow_section.shape[0]):
        mdl = np.polyfit(mrc[:, 0], np.log(mrc[:, 1]), 1)
        timeshift = (np.log(start_values[sort_indices[i]]) - mdl[1]) / mdl[0]
        new_segment = np.column_stack(
            (
                timeshift
                + np.arange(
                    1, flow_section[sort_indices[i], 1] - flow_section[sort_indices[i], 0] + 2
                ),
                streamflow[flow_section[sort_indices[i], 0] : flow_section[sort_indices[i], 1] + 1],
            )
        )
        mrc = np.vstack((mrc, new_segment))

    return mrc


def _get_nonparam_matrix(
    segments: list[FloatArray], flow_vals: FloatArray, numflows: int
) -> tuple[sparse.coo_matrix, FloatArray, list[int]]:
    """Get matrix for non-parametric analytic method."""
    msp_matrix, b_matrix, mcount, mspcount, bad_segs = [], [], 0, 0, []

    for i, segment in enumerate(segments):
        fmax_index = np.argmax(segment[0] >= flow_vals)
        fmin_index = (
            numflows if segment[-1] <= flow_vals[-1] else np.argmax(segment[-1] > flow_vals) - 1
        )

        interp_func = interpolate.interp1d(
            segment, np.arange(len(segment)), bounds_error=False, fill_value="extrapolate"
        )
        interp_segment = interp_func(flow_vals[fmax_index:fmin_index])

        nf = fmin_index - fmax_index

        if nf == 0:
            bad_segs.append(i)
            continue

        if i == 0:
            msp_matrix.extend(
                [
                    (j, len(segments) + fmax_index + k - 1, -1)
                    for j, k in enumerate(range(nf), start=mcount)
                ]
            )
            b_matrix.extend(interp_segment)
        else:
            msp_matrix.extend([(j, i - 1, 1) for j in range(mcount, mcount + nf)])
            msp_matrix.extend(
                [
                    (j, len(segments) + fmax_index + k - 1, -1)
                    for j, k in enumerate(range(nf), start=mcount)
                ]
            )
            b_matrix.extend(interp_segment)

        mcount += nf
        mspcount += nf if i == 0 else 2 * nf

    rows, cols, data = zip(*msp_matrix)
    m_sparse = sparse.coo_matrix((data, (rows, cols)), shape=(mcount, len(segments) - 1 + numflows))
    b_mat = -np.array(b_matrix)
    return m_sparse, b_mat, bad_segs


def _nonparametric_analytic_mrc(
    streamflow: FloatArray,
    flow_section: npt.NDArray[np.int64],
    match_method: Literal["linear", "log"] = "log",
) -> FloatArray:
    """Compute non-parametric analytic method for master recession curve."""
    jitter_size, numflows = 1e-8, 500
    rng = np.random.default_rng(42)

    segments = [streamflow[start : end + 1] for start, end in flow_section]

    total_jitter_length = sum(len(segment) - 1 for segment in segments)
    all_jitter = rng.normal(0, jitter_size, total_jitter_length)

    jitter_index = 0
    for i, segment in enumerate(segments):
        segment_length = len(segment)
        segment_jitter = all_jitter[jitter_index : jitter_index + segment_length - 1]
        jitter_index += segment_length - 1

        segment[1:] += segment_jitter
        # avoid negative segment values and sort the segment with jitter, in case
        # eps parameter was used and so thereare small increases during the recessions
        segments[i] = np.sort(np.abs(segment) + 1e-20)[::-1]

    max_flow = max(seg[0] for seg in segments)
    min_flow = max(min(seg[-1] for seg in segments), jitter_size)

    if match_method == "linear":
        flow_vals = np.linspace(max_flow, min_flow, numflows)
    elif match_method == "log":
        frac_log = 0.2
        gridspace = (max_flow - min_flow) / numflows
        flow_vals = np.sort(
            np.concatenate(
                [
                    np.linspace(
                        max_flow - gridspace / 2,
                        min_flow + gridspace / 2,
                        numflows - int(frac_log * numflows),
                    ),
                    np.logspace(np.log10(max_flow), np.log10(min_flow), int(frac_log * numflows)),
                ]
            )
        )[::-1]
        flow_vals[-1] = min_flow
        flow_vals[0] = max_flow
        flow_vals = np.sort(np.unique(flow_vals))[::-1]
        numflows = len(flow_vals)

    m_sparse, b_mat, bad_segs = _get_nonparam_matrix(segments, flow_vals, numflows)

    for i in sorted(bad_segs, reverse=True):
        m_sparse = m_sparse.tocsc()
        m_sparse = m_sparse[:, list(range(i - 1)) + list(range(i, m_sparse.shape[1]))]

    mrc_solve = optimize.lsq_linear(m_sparse, b_mat).x

    lags = np.concatenate(([0], mrc_solve[: len(segments) - 1 - len(bad_segs)]))
    mrc_time = mrc_solve[len(segments) - 1 - len(bad_segs) :]
    mrc_time = np.sort(mrc_time)
    offset = np.min(mrc_time)
    mrc_time -= offset
    lags -= offset

    result = np.column_stack((mrc_time, flow_vals))

    return result


def baseflow_recession(
    streamflow: FloatArray | pd.Series,
    freq: float = 1.0,
    recession_length: int = 15,
    n_start: int = 0,
    eps: float = 0,
    start_of_recession: Literal["baseflow", "peak"] = "baseflow",
    fit_method: Literal["nonparametric_analytic", "exponential"] = "nonparametric_analytic",
    lyne_hollick_smoothing: float = 0.925,
) -> tuple[FloatArray, float]:
    """Calculate baseflow recession constant and master recession curve.

    Notes
    -----
    This function is ported from the TOSSH Matlab toolbox, which is based on the
    following publication:

    Gnann, S.J., Coxon, G., Woods, R.A., Howden, N.J.K., McMillan H.K., 2021.
    TOSSH: A Toolbox for Streamflow Signatures in Hydrology.
    Environmental Modelling & Software.
    https://doi.org/10.1016/j.envsoft.2021.104983

    This function calculates baseflow recession constant assuming exponential
    recession behaviour (Safeeq et al., 2013). Master recession curve (MRC) is
    constructed using the adapted matching strip method (Posavec et al.,
    2006).

    According to Safeeq et al. (2013), :math:`K < 0.065` represent groundwater
    dominated slow-draining systems, :math:`K >= 0.065` represent shallow subsurface
    flow dominated fast draining systems.

    Parameters
    ----------
    streamflow : numpy.ndarray
        Streamflow as a 1D array.
    freq : float, optional
        Frequency of steamflow in number of days. Default is 1, i.e., daily streamflow.
    recession_length : int, optional
        Minimum length of recessions [days]. Default is 15.
    n_start : int, optional
        Days to be removed after start of recession. Default is 0.
    eps : float, optional
        Allowed increase in flow during recession period. Default is 0.
    start_of_recession : {'baseflow', 'peak'}, optional
        Define start of recession. Default is 'baseflow'.
    fit_method : {'nonparametric_analytic', 'exponential'}, optional
        Method to fit mrc. Default is 'nonparametric_analytic'.
    lyne_hollick_smoothing : float, optional
        Smoothing parameter of Lyne-Hollick filter. Default is 0.925.

    Returns
    -------
    mrc : numpy.ndarray
        Master Recession Curve as 2D array of [time, flow].
    bf_recession_k : float
        Baseflow Recession Constant [1/day].

    Raises
    ------
    ValueError
        If no recession segments are found or if a complex BaseflowRecessionK is calculated.
    """
    # TO DO: remove snow-affected sections of the time series
    streamflow = np.array(streamflow, "f8").squeeze()

    if streamflow.ndim != 1:
        raise InputTypeError("streamflow", "1D numpy.ndarray or pandas.Series")

    valid_fit = ("nonparametric_analytic", "exponential")
    if fit_method not in valid_fit:
        raise InputValueError("fit_method", valid_fit)

    valid_start = ("baseflow", "peak")
    if start_of_recession not in valid_start:
        raise InputValueError("start_of_recession", valid_start)

    if eps > np.nanmedian(streamflow) / 100:
        msg = ". ".join(
            (
                "eps set to a value larger than 1 percent of median(Q)",
                "High eps values can lead to problematic recession selection.",
            )
        )
        warnings.warn(msg, UserWarning, stacklevel=2)

    flow_section = _recession_segments(
        streamflow,
        np.float64(freq),
        np.int64(recession_length),
        np.float64(eps),
        np.float64(lyne_hollick_smoothing),
        start_of_recession,
        np.int64(n_start),
    )

    if len(flow_section) == 0:
        raise ValueError("No recession segments found.")  # noqa: TRY003
    elif len(flow_section) < 10:
        warnings.warn(
            "Fewer than 10 recession segments extracted, results might not be robust.",
            UserWarning,
            stacklevel=2,
        )

    if fit_method == "nonparametric_analytic":
        mrc = _nonparametric_analytic_mrc(streamflow, flow_section)
    else:
        mrc = _exponential_mrc(streamflow, flow_section)

    slope, *_ = stats.linregress(mrc[:, 0], np.log(mrc[:, 1]))
    slope = cast("float", slope)
    bf_recession_k = -slope / freq
    return mrc, bf_recession_k


@ngjit()
def __backward_pass(q: FloatArray, alpha: float) -> FloatArray:
    """Perform backward pass of the Lyne and Hollick filter."""
    qf = np.zeros_like(q)
    qf[-1] = q[-1] - np.min(q)  # initial condition, see Su et al. (2016)
    for i in range(q.size - 2, -1, -1):
        qf[i] = alpha * qf[i + 1] + 0.5 * (1 + alpha) * (q[i] - q[i + 1])

    for i in range(qf.shape[0]):
        qf[i] = q[i] - qf[i] if qf[i] > 0 else q[i]
    return qf


@ngjit(parallel=True)
def __batch_backward(q: FloatArray, alpha: float) -> FloatArray:
    """Perform forward pass of the Lyne and Hollick filter."""
    qb = np.zeros_like(q)
    for i in prange(q.shape[0]):
        qb[i] = __backward_pass(q[i], alpha)
    return qb


def baseflow(
    discharge: ArrayVar, alpha: float = 0.925, n_passes: int = 1, pad_width: int = 10
) -> ArrayVar:
    """Extract baseflow using the Lyne and Hollick filter (Ladson et al., 2013).

    Parameters
    ----------
    discharge : numpy.ndarray or pandas.DataFrame or pandas.Series or xarray.DataArray
        Discharge time series that must not have any missing values. It can also be a 2D array
        where each row is a time series.
    n_passes : int, optional
        Number of filter passes, defaults to 1.
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

    if n_passes < 1 or n_passes % 2 == 0:
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
