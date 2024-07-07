"""Tests for HydroSignatures package."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import hydrosignatures as hs


def assert_close(a: float, b: float) -> None:
    np.testing.assert_allclose(a, b, rtol=1e-3)


@pytest.fixture()
def datasets() -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    df = pd.read_csv(Path("tests", "test_data.csv"), index_col=0, parse_dates=True)
    with Path("tests", "test_data.json").open("r") as f:
        sig_expected = json.load(f)
    return df.q_mmpd, df.p_mmpd, sig_expected


@pytest.fixture()
def streamflow() -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    return pd.read_csv(Path("tests", "test_streamflow.csv"), index_col=0).squeeze()


@pytest.mark.speedup()
def test_signatures(datasets):
    q_mmpd, p_mmpd, sig_expected = datasets
    sig = hs.HydroSignatures(q_mmpd, p_mmpd)
    sig_dict = sig.to_dict()
    mm = sig_expected.pop("mean_monthly")
    assert all(np.isclose(sig_dict[key], val, rtol=1.0e-3) for key, val in sig_expected.items())
    assert np.allclose(pd.DataFrame(mm), sig.values.mean_monthly)


@pytest.mark.speedup()
def test_recession(streamflow):
    mrc, k = hs.baseflow_recession(streamflow)
    assert_close(mrc.max(), 70.7921)
    assert_close(k, 0.0560)


def test_show_versions():
    f = io.StringIO()
    hs.show_versions(file=f)
    assert "SYS INFO" in f.getvalue()
