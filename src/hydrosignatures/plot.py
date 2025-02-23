"""Plot hydrological signatures.

Plots include daily, monthly and annual hydrograph as well as regime
curve (monthly mean) and flow duration curve.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from hydrosignatures.exceptions import InputTypeError
from hydrosignatures.hydrosignatures import exceedance, mean_monthly

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes


__all__ = ["prepare_plot_data", "signatures"]


@dataclass(frozen=True)
class _PlotDataType:
    """Data structure for plotting hydrologic signatures."""

    daily: pd.DataFrame
    mean_monthly: pd.DataFrame
    ranked: pd.DataFrame
    titles: dict[str, str]
    units: dict[str, str]

    @classmethod
    def fields(cls) -> tuple[str, ...]:
        """Return the field names of the dataclass."""
        return tuple(field.name for field in fields(cls))


def prepare_plot_data(daily: pd.DataFrame | pd.Series) -> _PlotDataType:
    """Generate a structured data for plotting hydrologic signatures.

    Parameters
    ----------
    daily : pandas.Series or pandas.DataFrame
        The data to be processed

    Returns
    -------
    PlotDataType
        Containing ``daily, ``mean_monthly``, ``ranked``, ``titles``,
        and ``units`` fields.
    """
    if isinstance(daily, pd.Series):
        daily = daily.to_frame()
    mean_month = mean_monthly(daily, True)
    ranked = exceedance(daily)

    _titles = [
        "Total Hydrograph (daily)",
        "Regime Curve (monthly mean)",
        "Flow Duration Curve",
    ]
    _units = [
        "mm/day",
        "mm/month",
        "mm/day",
    ]
    _fields = _PlotDataType.fields()
    titles = dict(zip(_fields[:-1], _titles))
    units = dict(zip(_fields[:-1], _units))
    return _PlotDataType(daily, mean_month, ranked, titles, units)


def _prepare_plot_data(
    daily: pd.DataFrame | pd.Series,
    prcp: pd.DataFrame | pd.Series | None = None,
) -> tuple[_PlotDataType, _PlotDataType | None]:
    if not isinstance(daily, (pd.DataFrame, pd.Series)):
        raise InputTypeError("daily", "pandas.DataFrame or pandas.Series")

    discharge = prepare_plot_data(daily)
    if prcp is not None:
        if isinstance(prcp, pd.DataFrame) and prcp.shape[1] == 1:
            prcp = prepare_plot_data(prcp.squeeze())  # pyright: ignore[reportArgumentType]
        elif isinstance(prcp, pd.Series):
            prcp = prepare_plot_data(prcp)
        else:
            raise InputTypeError(
                "precipitation", "pandas.Series or pandas.DataFrame with one column"
            )
    else:
        prcp = None

    return discharge, prcp


def _add_prcp(
    prcp: pd.DataFrame | pd.Series,
    ax: Axes,
    daily: pd.DataFrame,
    lines: list[Artist],
    labels: list[Any],
) -> tuple[list[Artist], list[Any]]:
    """Add precipitation to the plot."""
    _prcp = prcp.daily.squeeze()
    _prcp = _prcp.loc[daily.index[0] : daily.index[-1]]
    label = "$P$ (mm/day)"
    ax_p = ax.twinx()
    ax_p.grid(False)
    if _prcp.shape[0] > 1000:
        ax_p.plot(
            _prcp,
            alpha=0.7,
            color="g",
            label=label,
        )
    else:
        ax_p.bar(
            _prcp.index,
            _prcp.to_numpy().ravel(),
            alpha=0.7,
            width=1,
            color="g",
            align="edge",
            label=label,
        )
    ax_p.set_ylim(_prcp.max() * 2.5, 0)
    ax_p.set_ylabel(label)
    ax_p.set_xmargin(0)
    lines_p, labels_p = ax_p.get_legend_handles_labels()
    lines.extend(lines_p)
    labels.extend(labels_p)
    return lines, labels


def signatures(
    discharge: pd.DataFrame | pd.Series,
    precipitation: pd.Series | None = None,
    title: str | None = None,
    figsize: tuple[int, int] | None = None,
    output: str | Path | None = None,
    close: bool = False,
) -> None:
    """Plot hydrological signatures w/ or w/o precipitation.

    Plots includes daily hydrograph, regime curve (mean monthly) and
    flow duration curve. The input discharges are converted from cms
    to mm/day based on the watershed area, if provided.

    Parameters
    ----------
    discharge : pd.DataFrame or pd.Series
        The streamflows in mm/day. The column names are used as labels
        on the plot and the column values should be daily streamflow.
    precipitation : pd.Series, optional
        Daily precipitation time series in mm/day. If given, the data is
        plotted on the second x-axis at the top.
    title : str, optional
        The plot supertitle.
    figsize : tuple, optional
        The figure size in inches, defaults to (9, 5).
    output : str, optional
        Path to save the plot as png, defaults to ``None`` which means
        the plot is not saved to a file.
    close : bool, optional
        Whether to close the figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting. Please install it.") from e
    qdaily, prcp = _prepare_plot_data(discharge, precipitation)

    daily = qdaily.daily
    figsize = (9, 5) if figsize is None else figsize
    fig = plt.figure(constrained_layout=True, figsize=figsize, facecolor="w")
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, :])
    for c, q in daily.items():
        ax.plot(q, label=c)
    lines, labels = ax.get_legend_handles_labels()
    ax.set_ylabel("$Q$ (mm/day)")
    ax.text(0.02, 0.9, "(a)", transform=ax.transAxes, ha="left", va="center", fontweight="bold")

    if prcp is not None:
        lines, labels = _add_prcp(prcp, ax, daily, lines, labels)

    ax.set_xmargin(0)
    ax.set_xlabel("")
    ax.legend(
        lines,
        labels,
        bbox_to_anchor=(0.5, -0.2),
        loc="upper center",
        ncol=len(lines),
    )

    ax = fig.add_subplot(gs[1, :-1])
    ax.plot(qdaily.mean_monthly)
    ax.set_xmargin(0)
    ax.set_ylabel("$Q$ (mm/month)")
    ax.text(0.02, 0.9, "(b)", transform=ax.transAxes, ha="left", va="center", fontweight="bold")

    ax = fig.add_subplot(gs[1, 2])
    for col in daily:
        dc = qdaily.ranked[[col, f"{col}_rank"]]
        ax.plot(dc[f"{col}_rank"], dc[col], label=col)

    ax.set_yscale("log")
    ax.set_xlim(0, 100)
    ax.set_xlabel("% Exceedance")
    ax.set_ylabel(rf"$\log(Q)$ ({qdaily.units['ranked']})")
    ax.text(0.02, 0.9, "(c)", transform=ax.transAxes, ha="left", va="center", fontweight="bold")

    if title:
        fig.suptitle(title)

    if output is not None:
        Path(output).parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(output, dpi=300)  # pyright: ignore[reportGeneralTypeIssues]

    if close:
        plt.close(fig)
