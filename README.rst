.. image:: https://raw.githubusercontent.com/hyriver/HyRiver-examples/main/notebooks/_static/hydrosignatures_logo.png
    :target: https://github.com/hyriver/HyRiver

|

.. image:: https://joss.theoj.org/papers/b0df2f6192f0a18b9e622a3edff52e77/status.svg
    :target: https://joss.theoj.org/papers/b0df2f6192f0a18b9e622a3edff52e77
    :alt: JOSS

|

.. |pygeohydro| image:: https://github.com/hyriver/pygeohydro/actions/workflows/test.yml/badge.svg
    :target: https://github.com/hyriver/pygeohydro/actions/workflows/test.yml
    :alt: Github Actions

.. |pygeoogc| image:: https://github.com/hyriver/pygeoogc/actions/workflows/test.yml/badge.svg
    :target: https://github.com/hyriver/pygeoogc/actions/workflows/test.yml
    :alt: Github Actions

.. |pygeoutils| image:: https://github.com/hyriver/pygeoutils/actions/workflows/test.yml/badge.svg
    :target: https://github.com/hyriver/pygeoutils/actions/workflows/test.yml
    :alt: Github Actions

.. |pynhd| image:: https://github.com/hyriver/pynhd/actions/workflows/test.yml/badge.svg
    :target: https://github.com/hyriver/pynhd/actions/workflows/test.yml
    :alt: Github Actions

.. |py3dep| image:: https://github.com/hyriver/py3dep/actions/workflows/test.yml/badge.svg
    :target: https://github.com/hyriver/py3dep/actions/workflows/test.yml
    :alt: Github Actions

.. |pydaymet| image:: https://github.com/hyriver/pydaymet/actions/workflows/test.yml/badge.svg
    :target: https://github.com/hyriver/pydaymet/actions/workflows/test.yml
    :alt: Github Actions

.. |pygridmet| image:: https://github.com/hyriver/pygridmet/actions/workflows/test.yml/badge.svg
    :target: https://github.com/hyriver/pygridmet/actions/workflows/test.yml
    :alt: Github Actions

.. |pynldas2| image:: https://github.com/hyriver/pynldas2/actions/workflows/test.yml/badge.svg
    :target: https://github.com/hyriver/pynldas2/actions/workflows/test.yml
    :alt: Github Actions

.. |async| image:: https://github.com/hyriver/async-retriever/actions/workflows/test.yml/badge.svg
    :target: https://github.com/hyriver/async-retriever/actions/workflows/test.yml
    :alt: Github Actions

.. |signatures| image:: https://github.com/hyriver/hydrosignatures/actions/workflows/test.yml/badge.svg
    :target: https://github.com/hyriver/hydrosignatures/actions/workflows/test.yml
    :alt: Github Actions

================ ==================================================================== ============
Package          Description                                                          Status
================ ==================================================================== ============
PyNHD_           Navigate and subset NHDPlus (MR and HR) using web services           |pynhd|
Py3DEP_          Access topographic data through National Map's 3DEP web service      |py3dep|
PyGeoHydro_      Access NWIS, NID, WQP, eHydro, NLCD, CAMELS, and SSEBop databases    |pygeohydro|
PyDaymet_        Access daily, monthly, and annual climate data via Daymet            |pydaymet|
PyGridMET_       Access daily climate data via GridMET                                |pygridmet|
PyNLDAS2_        Access hourly NLDAS-2 data via web services                          |pynldas2|
HydroSignatures_ A collection of tools for computing hydrological signatures          |signatures|
AsyncRetriever_  High-level API for asynchronous requests with persistent caching     |async|
PyGeoOGC_        Send queries to any ArcGIS RESTful-, WMS-, and WFS-based services    |pygeoogc|
PyGeoUtils_      Utilities for manipulating geospatial, (Geo)JSON, and (Geo)TIFF data |pygeoutils|
================ ==================================================================== ============

.. _PyGeoHydro: https://github.com/hyriver/pygeohydro
.. _AsyncRetriever: https://github.com/hyriver/async-retriever
.. _PyGeoOGC: https://github.com/hyriver/pygeoogc
.. _PyGeoUtils: https://github.com/hyriver/pygeoutils
.. _PyNHD: https://github.com/hyriver/pynhd
.. _Py3DEP: https://github.com/hyriver/py3dep
.. _PyDaymet: https://github.com/hyriver/pydaymet
.. _PyGridMET: https://github.com/hyriver/pygridmet
.. _PyNLDAS2: https://github.com/hyriver/pynldas2
.. _HydroSignatures: https://github.com/hyriver/hydrosignatures

HydroSignatures: Tools for computing hydrological signatures
------------------------------------------------------------

.. image:: https://img.shields.io/pypi/v/hydrosignatures.svg
    :target: https://pypi.python.org/pypi/hydrosignatures
    :alt: PyPi

.. image:: https://img.shields.io/conda/vn/conda-forge/hydrosignatures.svg
    :target: https://anaconda.org/conda-forge/hydrosignatures
    :alt: Conda Version

.. image:: https://codecov.io/gh/hyriver/hydrosignatures/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/hyriver/hydrosignatures
    :alt: CodeCov

.. image:: https://img.shields.io/pypi/pyversions/hydrosignatures.svg
    :target: https://pypi.python.org/pypi/hydrosignatures
    :alt: Python Versions

.. image:: https://static.pepy.tech/badge/hydrosignatures
    :target: https://pepy.tech/project/hydrosignatures
    :alt: Downloads

|

.. image:: https://www.codefactor.io/repository/github/hyriver/hydrosignatures/badge
   :target: https://www.codefactor.io/repository/github/hyriver/hydrosignatures
   :alt: CodeFactor

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
    :target: https://github.com/pre-commit/pre-commit
    :alt: pre-commit

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/hyriver/HyRiver-examples/main?urlpath=lab/tree/notebooks
    :alt: Binder

|

Features
--------

HydroSignatures is a suite of tools for computing hydrological signatures
and a part of `HyRiver <https://github.com/hyriver/HyRiver>`__ software stack.
This package includes the following functions:

- ``exceedance``: Exceedance probability that can be used plotting flow
  duration curves;
- ``flow_duration_curve_slope``: Slope of flow duration curve;
- ``flashiness_index``: Flashiness index;
- ``mean_monthly``: Mean monthly summary of a time series that can be used
  for plotting regime curves;
- ``rolling_mean_monthly``: Rolling mean monthly summary of a time series
  that can be used for plotting smoothed regime curves;
- ``baseflow``: Extracting baseflow from a streamflow time series using the
  Lyne and Hollick digital filter (Ladson et al., 2013);
- ``baseflow_recession``: Baseflow recession analysis using the nonparametric
  analytic (Posavec et al., 2006) and exponential fit methods;
- ``baseflow_index``: Baseflow index;
- ``aridity_index``: Aridity index;
- ``seasonality_index_walsh``: Seasonality index (Walsh and Lawler, 1981);
- ``seasonality_index_markham``: Seasonality index (Markham, 1970);
- ``extract_extrema``: Determining the location of local maxima and minima in a
  time series;

Moreover, the package has a class called ``HydroSignatures`` that can be used to compute
all these signatures by passing a streamflow and a precipitation time series, both
in millimeters per day (or any other unit of time). This class supports subtraction
and inequality operators, which can be used to compare two ``HydroSignatures`` objects.
You can serialize the class to a JSON object using the ``to_json`` method or convert it
to a dictionary using the ``to_dict`` method.

Moreover, ``numba`` is an optional dependency for the ``baseflow`` function.
Installing ``numba`` will speed up the computation of baseflow significantly.
For more efficient handling of NaN values, you can also install ``numbagg``.

You can also try using HydroSignatures without installing
it on your system by clicking on the binder badge. A Jupyter Lab
instance with the HyRiver stack pre-installed will be launched in your web browser, and you
can start coding!

Moreover, requests for additional functionalities can be submitted via
`issue tracker <https://github.com/hyriver/hydrosignatures/issues>`__.

Citation
--------
If you use any of HyRiver packages in your research, we appreciate citations:

.. code-block:: bibtex

    @article{Chegini_2021,
        author = {Chegini, Taher and Li, Hong-Yi and Leung, L. Ruby},
        doi = {10.21105/joss.03175},
        journal = {Journal of Open Source Software},
        month = {10},
        number = {66},
        pages = {1--3},
        title = {{HyRiver: Hydroclimate Data Retriever}},
        volume = {6},
        year = {2021}
    }

Installation
------------

You can install HydroSignatures using ``pip``:

.. code-block:: console

    $ pip install hydrosignatures

or from the ``conda-forge`` repository using `Conda <https://docs.conda.io/en/latest/>`__
or `Mamba <https://github.com/conda-forge/miniforge>`__:

.. code-block:: console

    $ conda install -c conda-forge hydrosignatures

Quick start
-----------

Let's explore the capabilities of ``HydroSignatures`` by getting streamflow
using PyGeoHydro, basin geometry using PyNHD and precipitation using PyDaymet.
In this example, we select West Branch Herring Run At Idlewylde, MD, as the
watershed of interest and compute the hydrological signatures for the period
from 2010 to 2020.

.. code-block:: python

    import pydaymet as daymet
    import hydrosignatures as hs
    import pygeohydro as gh
    from hydrosignatures import HydroSignatures
    from pygeohydro import NWIS
    from pynhd import WaterData

    site = "01585200"
    start = "2010-01-01"
    end = "2020-12-31"

First, we get the basin geometry of the watershed using ``gagesii_basins`` layer of
the USGS's WaterData web service.

.. code-block:: python

    wd = WaterData("gagesii_basins")
    geometry = wd.byid("gage_id", site).geometry[0]

Then, we obtain the station's info and streamflow data using NWIS. Note that
we should convert the streamflow from cms to mm/day.

.. code-block:: python

    nwis = NWIS()
    info = nwis.get_info({"site": site})
    area_sqm = info.drain_sqkm.values[0] * 1e6
    q_cms = nwis.get_streamflow(site, (start, end))
    q_mmpd = q_cms * (24.0 * 60.0 * 60.0) / area_sqm * 1e3
    q_mmpd.index = pd.to_datetime(q_mmpd.index.date)

Next, we retrieve the precipitation data using PyDaymet over the whole basin
using the basin geometry and take its mean as the basin's precipitation.

.. code-block:: python

    prcp = daymet.get_bygeom(geometry, (start, end), variables="prcp")
    p_mmpd = prcp.prcp.mean(dim=["x", "y"]).to_pandas()
    p_mmpd.index = pd.to_datetime(p_mmpd.index.date)
    q_mmpd = q_mmpd.loc[p_mmpd.index]

Now, we can pass these two to the ``HydroSignatures`` class:

.. code-block:: python

    sig = HydroSignatures(q_mmpd, p_mmpd)

The ``values`` property of this class contains the computed signatures. For example,
let's plot the regime curves:

.. code-block:: python

    sig.values.mean_monthly.plot()


.. image:: https://raw.githubusercontent.com/hyriver/HyRiver-examples/main/notebooks/_static/signatures_rc.png
    :target: https://docs.hyriver.io/examples/notebooks/signatures.ipynb
    :align: center

Note that, you can also use the functions directly. For example, let's get
streamflow observations for another station and separate the baseflow using
various filter parameters and compare them:

.. code-block:: python

    import numpy as np
    import pandas as pd

    q = nwis.get_streamflow("12304500", ("2019-01-01", "2019-12-31"))
    alpha = np.arange(0.9, 1, 0.01)
    qb = pd.DataFrame({a: hs.baseflow(q.squeeze(), alpha=a) for a in alpha})


.. image:: https://raw.githubusercontent.com/hyriver/HyRiver-examples/main/notebooks/_static/signatures_bf.png
    :target: https://docs.hyriver.io/examples/notebooks/signatures.ipynb
    :align: center

We can also carry out a baseflow recession analysis using the ``baseflow_recession``
function. For this we need to get streamflow data for a longer period.

.. code-block:: python

    q = nwis.get_streamflow("12304500", ("2000-01-01", "2019-12-31"))
    mrc_np, bfr_k_np = hs.baseflow_recession(q, fit_method="nonparametric_analytic")
    mrc_exp, bfr_k_exp = hs.baseflow_recession(q, fit_method="exponential")

According to Safeeq et al. (2013), $K$ value of $0.065$ is the threshold between groundwater
dominated slow-draining systems and shallow subsurface flow dominated fast draining systems.
In this example, since $K= 0.056$, the watershed is groundwater dominated.

.. image:: https://raw.githubusercontent.com/hyriver/HyRiver-examples/main/notebooks/_static/recession.png
    :target: https://docs.hyriver.io/examples/notebooks/signatures.ipynb
    :align: center

Lastly, let's compute Markham's seasonality index for all streamflow time series of
the stations in the CAMELS dataset. We retrieve the CAMELS dataset using PyGeoHydro:

.. code-block:: python

    import xarray as xr

    _, camels_qobs = gh.get_camels()
    discharge = camels_qobs.discharge.dropna("station_id")
    discharge = xr.where(discharge < 0, 0, discharge)
    si = hs.seasonality_index_markham(discharge.to_pandas())

More examples can be found `here <https://docs.hyriver.io/examples.html>`__.

Contributing
------------

Contributions are very welcomed. Please read
`CONTRIBUTING.rst <https://github.com/hyriver/hydrosignatures/blob/main/CONTRIBUTING.rst>`__
file for instructions.
