"""Configuration for pytest."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _add_standard_imports(doctest_namespace):
    """Add hydrosignatures namespace for doctest."""
    import hydrosignatures as hs

    doctest_namespace["hs"] = hs
