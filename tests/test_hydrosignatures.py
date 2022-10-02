"""Tests for PyNHD package."""
import io
import os
from pathlib import Path

import hydrosignatures as hs
import numpy as np
from pygeohydro import NWIS
import pytest

try:
    import typeguard  # noqa: F401
except ImportError:
    has_typeguard = False
else:
    has_typeguard = True

is_ci = os.environ.get("GH_CI") == "true"
STA_ID = "01031500"
station_id = f"USGS-{STA_ID}"
site = "nwissite"
UM = "upstreamMain"
UT = "upstreamTributaries"


def assert_close(a: float, b: float) -> bool:
    assert np.isclose(a, b, rtol=1e-3).all()


@pytest.fixture
def streamflow():
    return NWIS().get_streamflow("12304500", ("2019-01-01", "2019-12-31"))


def test_baseflow(streamflow):
    bf = hs.compute_baseflow(streamflow.squeeze(), alpha=0.93)
    assert_close(bf.sum(), 2489.142)


def test_show_versions():
    f = io.StringIO()
    hs.show_versions(file=f)
    assert "INSTALLED VERSIONS" in f.getvalue()
