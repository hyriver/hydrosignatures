"""Function for computing hydrologic signature."""

from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from scipy import interpolate, optimize, sparse, stats

from hydrosignatures.exceptions import InputRangeError, InputTypeError, InputValueError

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
    FloatArray = npt.NDArray[np.floating]
    ArrayVar = TypeVar("ArrayVar", pd.Series, pd.DataFrame, FloatArray, xr.DataArray)
    ArrayLike = Union[pd.Series, pd.DataFrame, FloatArray, xr.DataArray]

__all__ = [
    "baseflow",
    "baseflow_index",
    "baseflow_recession",
]


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


@ngjit()
def _pad_array(arr: FloatArray, pad_width: int) -> FloatArray:
    padded_arr = np.empty(len(arr) + 2 * pad_width, dtype=arr.dtype)
    padded_arr[:pad_width] = arr[0]
    padded_arr[pad_width:-pad_width] = arr
    padded_arr[-pad_width:] = arr[-1]
    return padded_arr


@ngjit()
def __forward_pass(q: FloatArray, alpha: np.float64) -> FloatArray:
    """Perform forward pass of the Lyne and Hollick filter."""
    qb = np.zeros_like(q)
    qb[0] = q[0] - np.min(q)  # initial condition, see Su et al. (2016)
    for i in range(1, q.size):
        qb[i] = alpha * qb[i - 1] + 0.5 * (1 + alpha) * (q[i] - q[i - 1])

    for i in range(q.size):
        qb[i] = q[i] - qb[i] if qb[i] > 0 else q[i]
    return qb


@ngjit(parallel=True)
def __batch_forward(q: FloatArray, alpha: np.float64) -> FloatArray:
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
) -> tuple[sparse.coo_array, FloatArray, list[int]]:
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
    m_sparse = sparse.coo_array((data, (rows, cols)), shape=(mcount, len(segments) - 1 + numflows))
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
        col_idx = list(range(i - 1)) + list(range(i, m_sparse.shape[1]))
        m_sparse = m_sparse[:, col_idx]  # pyright: ignore[reportOptionalSubscript]

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
        raise ValueError("No recession segments found.")
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
def __backward_pass(q: FloatArray, alpha: np.float64) -> FloatArray:
    """Perform backward pass of the Lyne and Hollick filter."""
    qf = np.zeros_like(q)
    qf[-1] = q[-1] - np.min(q)  # initial condition, see Su et al. (2016)
    for i in range(q.size - 2, -1, -1):
        qf[i] = alpha * qf[i + 1] + 0.5 * (1 + alpha) * (q[i] - q[i + 1])

    for i in range(qf.shape[0]):
        qf[i] = q[i] - qf[i] if qf[i] > 0 else q[i]
    return qf


@ngjit(parallel=True)
def __batch_backward(q: FloatArray, alpha: np.float64) -> FloatArray:
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
    alpha_ = np.float64(alpha)
    if pad_width < 1:
        raise InputRangeError("pad_width", "greater than or equal to 1")

    q = __to_numpy(discharge)
    q = np.apply_along_axis(np.pad, 1, q, pad_width, "edge")
    q = cast("FloatArray", q)
    qb = __batch_forward(q, alpha_)
    passes = int(round(0.5 * (n_passes - 1)))
    for _ in range(passes):
        qb = __batch_forward(__batch_backward(qb, alpha_), alpha_)
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
