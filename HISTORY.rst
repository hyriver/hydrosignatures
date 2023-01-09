=======
History
=======

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
