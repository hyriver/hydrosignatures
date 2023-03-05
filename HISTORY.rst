=======
History
=======

0.14.0 (2023-03-05)
-------------------

Bug Fixes
~~~~~~~~~
- Address an issue in ``compute_fdc_slope`` where if the input
  includes NANs, it returns NAN. Now, the function correctly
  handles NAN values. Also, this function now works with any
  array-like input, i.e., ``pandas.Series``, ``pandas.DataFrame``,
  ``numpy.ndarray``, and ``xarray.DataArray``. Also, the denominator
  should have been divided by 100 since the input bins are
  percentiles.
- Fix a bug in ``compute_ai`` where instead of using mean annual
  average values, daily values was being used. Also, this function
  now accepts ``xarray.DataArray`` too.

Internal Changes
~~~~~~~~~~~~~~~~
- Sync all minor versions of HyRiver packages to 0.14.0.

0.1.12 (2023-02-10)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Fully migrate ``setup.cfg`` and ``setup.py`` to ``pyproject.toml``.
- Convert relative imports to absolute with ``absolufy-imports``.
- Sync all patch versions of HyRiver packages to x.x.12.

0.1.2 (2023-01-08)
------------------

New Features
~~~~~~~~~~~~
- Refactor the ``show_versions`` function to improve performance and
  print the output in a nicer table-like format.

Internal Changes
~~~~~~~~~~~~~~~~
- Use ``pyright`` for type checking and fix all typing issues that it raised.
- Add ``xarray`` as a dependency.

0.1.1 (2022-11-04)
------------------

New Features
~~~~~~~~~~~~
- Add a new function called ``compute_ai`` for computing the aridity index.
- Add a new function called ``compute_flood_moments`` for computing
  flood moments: Mean annual flood, coefficient of variation, and
  coefficient of skewness.
- Add a stand-alone function for computing the FDC slope, called ``compute_fdc_slope``.

Breaking Changes
~~~~~~~~~~~~~~~~
- Remove the ``runoff_ratio_annual`` function.

0.1.0 (2022-10-03)
------------------

- First release on PyPI.
